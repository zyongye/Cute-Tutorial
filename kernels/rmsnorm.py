import operator

import torch
import triton

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import make_fake_compact_tensor as fake_tensor

class RMSNorm:
    
    def __init__(
        self,
        dtype: type[cute.Numeric],
        D: int,
        reduction_dtype=cutlass.Float32,
    ):
        assert D == 4096

        self.dtype = dtype
        self.D = D
        self.reduction_dtype = reduction_dtype


    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        eps: cute.Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        shape = mO.shape

        # setup copy atom
        tiler_mn = (4, 4096)
        tv_layout = cute.make_layout(
            ((32, 4), (4, 32)),
            stride=((16, 1), (4, 512))
        )
        # expand 1D tensors
        mW_expand_layout = cute.make_layout((shape[0], mW.shape[0]), stride=(0, mW.stride[0]))
        mW_expand = cute.make_tensor(mW.iterator, mW_expand_layout)

        # call kernel
        idX = cute.make_identity_tensor(mX.shape)


        gO, gX, gW, cX = [
            cute.zipped_divide(mT, tiler_mn)
            for mT in (mO, mX, mW_expand, idX)
        ]

        print(f"gX: {gX.type}")

        self.kernel(
            gO, gX, gW, cX, mX.shape, tiler_mn, tv_layout, eps,
        ).launch(
            grid=[cute.ceil_div(shape[0], 4), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gO: cute.Tensor,
        gX: cute.Tensor, 
        gW: cute.Tensor,
        cX: cute.Tensor,
        shape: cute.Shape,
        tiler_mn: cute.Shape,
        tv_layout: cute.Layout,
        eps: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        blkO, blkX, blkW, blkCrd = [
            gT[(None, None), bidx] 
            for gT in (gO, gX, gW, cX)
        ]

        print(f"[DSL INFO]  blkW = {blkW}")

        copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=64)
        copy_atom_load_W = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gW.element_type, num_bits_per_copy=64)
        copy_atom_store_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=64)

        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(blkX)
        tXrX = cute.make_fragment_like(tXgX)
        tXgW = thr_copy_W.partition_S(blkW)
        tXrW = cute.make_fragment_like(tXgW)
        tXgO = thr_copy_O.partition_D(blkO)
        tXrO = cute.make_fragment_like(tXgO)
        tXcX = thr_copy_X.partition_S(blkCrd)[(0, None), None, None]
        row = tXcX[0][0]

        print(f"[DSL INFO]  tXgW = {tXgW.type}")
        print(f"[DSL INFO]  tXgW = {tXrW.type}")
        
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXrX)
            cute.copy(copy_atom_load_W, tXgW, tXrW)
        
        x = tXrX.load().to(self.reduction_dtype)
        square_x = x * x

        square_x = square_x.reduce(
            cute.ReductionOp.ADD, 0.0, reduction_profile=0,
        )

        sum_square_x = cute.arch.warp_reduction(
            square_x,
            operator.add,
        )

        rstd = cute.math.rsqrt(sum_square_x / shape[1] + eps, fastmath=True)
        
        y = x * rstd

        y *= tXrW.load().to(self.reduction_dtype)
        
        # store the result
        tXrO.store(y.to(tXrO.element_type))

        if row < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO)


def _rms_norm_fwd(
    out: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-5,
    benchmark: bool = False,
):
    # compile rmsnorm
    batch_sym = cute.sym_int()
    dim = x.shape[-1]
    x_fake_tensor = fake_tensor(cute.BFloat16, (batch_sym, dim), stride_order=(1, 0), assumed_align=16)
    w_fake_tensor = fake_tensor(cute.BFloat16, (dim, ), assumed_align=16)

    compiled_kernel = cute.compile(
        RMSNorm(x_fake_tensor.element_type, dim),
        x_fake_tensor,
        x_fake_tensor,
        w_fake_tensor,
        cutlass.Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )

    # call
    compiled_kernel(
        out, x, scale, eps,
    )

    if benchmark:
        fn = lambda: compiled_kernel(out, x, scale, eps)
        M, N = x.shape
        avg_time = triton.testing.do_bench(fn, warmup=2, rep=200)
        mem_bw = ((M * N * 2 + N) * x.element_size() // 8) / (avg_time / 1000) / 1e9
        print(f"Kernel execution time: {avg_time:.4f} ms")
        print(f"Mem throughput: {mem_bw:.2f} GB/s")


def cute_rms_norm(
    x: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-5,
):
    out = torch.empty_like(x)

    _rms_norm_fwd(
        out, 
        x, 
        scale,
        eps,
        benchmark=True,
    )

    return out
    
@torch.compile
def torch_rms_norm(
    x: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-5,
):
    d_model = x.shape[-1]
    d_model_ = scale.shape[0]
    assert d_model == d_model_
    
    t, dtype = x.float(), x.dtype
    t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
    return (t * scale).to(dtype)


def main():
    L = 16384
    d = 4096
    device = "cuda"
    x = torch.randn(L, d, dtype=torch.bfloat16, device=device)
    scale = torch.randn(d, dtype=torch.bfloat16, device=device)
    
    torch_result = torch_rms_norm(x, scale)

    cutedsl_result = cute_rms_norm(x, scale)
    
    torch.testing.assert_close(torch_result, cutedsl_result)


if __name__ == "__main__":
    main()
