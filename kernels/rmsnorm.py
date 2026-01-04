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
        mRes: cute.Tensor | None,
        mW: cute.Tensor,
        eps: cute.Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        shape = mO.shape

        # setup copy atom
        tiler_mn = (4, 4096)
        tv_layout = cute.make_layout(
            ((32, 4), (8, 16)), 
            stride=((32, 1), (4, 1024)),
        )
        # expand 1D tensors
        mW_expand_layout = cute.make_layout((shape[0], mW.shape[0]), stride=(0, mW.stride[0]))
        mW_expand = cute.make_tensor(mW.iterator, mW_expand_layout)
        
        idX = cute.make_identity_tensor(mX.shape)

        # call kernel
        self.kernel(
            mO, mX, mRes, mW_expand, idX, eps, tiler_mn, tv_layout,
        ).launch(
            grid=[cute.ceil_div(shape[0], 4), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mX: cute.Tensor,
        mRes: cute.Tensor | None, 
        mW: cute.Tensor,
        idX: cute.Tensor,
        eps: cutlass.Float32,
        tiler_mn: cute.Shape,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        shape = mX.shape

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        if cutlass.const_expr(mRes is not None):
            sRes = smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        gO, gX, gRes, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cutlass.const_expr(0))) if cutlass.const_expr(mT is not None) else None
            for mT in [mO, mX, mRes, idX]
        ]

        gW = cute.local_tile(mW, tiler_mn, (0, cutlass.const_expr(0)))

        print(f"[DSL INFO]  gW = {gW}")
        print(f"[DSL INFO]  mW = {mW}")

        copy_atom_async = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=128)
        
        # We reuse this atom for loading scalar and store result, since it can go directly from rmem to gmem
        copy_atom_sync = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)

        thr_copy_X = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXrX = cute.make_fragment_like(tXgX)
        if cutlass.const_expr(gRes is not None):
            tXgRes = thr_copy_X.partition_S(gRes)
            tXsRes = thr_copy_X.partition_D(sRes)
            tXrRes = cute.make_fragment_like(tXgRes)
        else:
            tXgRes, tXrRes = None, None
        tXgW = thr_copy_X.partition_S(gW)
        tXrW = cute.make_fragment_like(tXgW)
        tXgO = thr_copy_X.partition_D(gO)
        tXrO = cute.make_fragment_like(tXgO)
        tXcX = thr_copy_X.partition_S(cX)
        row = tXcX[0][0]

        tXpX = cute.make_rmem_tensor(
            cute.make_layout(
                (cute.size(tXcX, mode=[0, 1]), cute.size(tXcX, mode=[1]), cute.size(tXcX, mode=[2])),
                stride=(cute.size(tXcX, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )

        for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
            for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
                tXpX[rest_v, 0, rest_k] = cute.elem_less(
                    tXcX[(0, rest_v), 0, rest_k][1], shape[1]
                )

        print(f"[DSL INFO]  tXgX = {tXgX.type}")
        print(f"[DSL INFO]  tXrX = {tXrX.type}")
        print(f"[DSL INFO]  tXpX = {tXpX.type}")
        
        if row < shape[0]:
            cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            if cutlass.const_expr(tXgRes is not None):
                cute.copy(copy_atom_async, tXgRes, tXsRes, pred=tXpX)
            cute.copy(copy_atom_sync, tXgW, tXrW, pred=tXpX)
        
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tXsX, tXrX)
        if cutlass.const_expr(tXgRes is not None):
            cute.autovec_copy(tXsRes, tXrRes)
        
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
        
        if cutlass.const_expr(tXrRes is not None):
            y += tXrRes.load().to(self.reduction_dtype)
        
        # store the result
        tXrO.store(y.to(tXrO.element_type))

        if row < shape[0]:
            cute.copy(copy_atom_sync, tXrO, tXgO, pred=tXpX)


def _rms_norm_fwd(
    out: torch.Tensor,
    x: torch.Tensor,
    res: torch.Tensor | None,
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
        x_fake_tensor if res is not None else None,
        w_fake_tensor,
        cutlass.Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )

    # call
    compiled_kernel(
        out, x, res, scale, eps,
    )

    if benchmark:
        fn = lambda: compiled_kernel(out, x, res, scale, eps)
        M, N = x.shape
        avg_time = triton.testing.do_bench(fn, warmup=2, rep=200)
        mem_bw = ((M * N * (2 + (1 if res is not None else 0)) + N) * x.element_size()) / (avg_time / 1000) / 1e9
        print(f"Kernel execution time: {avg_time:.4f} ms")
        print(f"Mem throughput: {mem_bw:.2f} GB/s")


def cute_rms_norm(
    x: torch.Tensor,
    res: torch.Tensor | None,
    scale: torch.Tensor,
    eps: float = 1e-5,
):
    out = torch.empty_like(x)

    _rms_norm_fwd(
        out, 
        x, 
        res,
        scale,
        eps,
        benchmark=True,
    )

    return out
    
@torch.compile
def torch_rms_norm(
    x: torch.Tensor,
    res: torch.Tensor | None,
    scale: torch.Tensor,
    eps: float = 1e-5,
):
    d_model = x.shape[-1]
    d_model_ = scale.shape[0]
    assert d_model == d_model_
    
    t, dtype = x.float(), x.dtype
    t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
    return (t * scale + res).to(dtype) if res is not None else (t * scale).to(dtype)


def main():
    L = 16384
    d = 4096
    device = "cuda"
    res = torch.randn(L, d, dtype=torch.bfloat16, device=device)
    x = torch.randn(L, d, dtype=torch.bfloat16, device=device)
    scale = torch.randn(d, dtype=torch.bfloat16, device=device)
    
    torch_result = torch_rms_norm(x, res, scale)

    cutedsl_result = cute_rms_norm(x, res, scale)
    
    torch.testing.assert_close(torch_result, cutedsl_result)


if __name__ == "__main__":
    main()
