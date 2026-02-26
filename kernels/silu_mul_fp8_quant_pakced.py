import functools
import math
from typing import Callable

import torch

import cuda.bindings.driver as cuda


import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
from cutlass.cute.typing import AddressSpace
from cutlass import Float32, Int32, BFloat16
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


FP8_MAX = 448
FP8_MIN = -448
COPY_BITS = 128

@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][1], limit
            )
    return tXpX

@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def fma(a: float | Float32, b: float | Float32, c: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def silu(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """
    silu(a) = a * sigmoid(a) = a * (1 + tanh(a / 2)) / 2 = (0.5 * a) * tanh(0.5 * a) + (0.5 * a)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * a, MUFU.TANH, and FFMA.
    """
    # return a / (1.0 + cute.arch.exp2(-a * math.log2(math.e)))
    a_half = 0.5 * a
    # return a_half * self.tanh(a_half) + a_half
    return fma(a_half, tanh(a_half), a_half)


class SiluMulFP8QuantPackedKernel:

    def __init__(
        self,
        dtype: cute.Numeric,
        hidden_size: int,
        block_size: int,
        is_ue8m0: bool,
        is_input_interleaved: bool = False,
    ):
        assert hidden_size % 256 == 0, "Hidden size must be multiple of 256 for this kernel"
        assert not is_input_interleaved, "Interleaved input layout is not supported in this kernel version"
        assert is_ue8m0, "Only UE8M0 format is supported in this kernel version"
        
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.is_ue8m0 = is_ue8m0

        self.reduction_dtype = Float32

        self.H_per_cta = hidden_size

        self.threads_per_row = 32
        self.num_threads = 128
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        elem_bytes = dtype.width // 8
        self.vec_size = COPY_BITS // 8 // elem_bytes
        self.num_vec_blocks = max(
            1, 
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row
        self.num_sf_blocks_per_row = self.hidden_size // self.block_size

        print(f"hidden_size: {hidden_size}, block_size: {block_size}")
        print(f"threads_per_row: {self.threads_per_row}, rows_per_block:{self.rows_per_block}, warps_per_row: {self.warps_per_row}")
        print(f"vec_size: {self.vec_size}, num_vec_blocks: {self.num_vec_blocks}, cols_per_tile: {self.cols_per_tile}")

    @staticmethod
    def _make_tv_layout(
        threads_per_row: int,
        rows_per_block: int,
        vec_size: int,
        num_vec_blocks: int,
    ) -> tuple:
        """Create Thread-Value layout for coalesced vectorized memory access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    @cute.jit
    def __call__(
        self,
        X_ptr: cute.Pointer,
        G_ptr: cute.Pointer,
        O_ptr: cute.Pointer,
        S_ptr: cute.Pointer,
        M: Int32,
        eps: Float32,
        stream,
    ):
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )

        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        print(tv_layout)
        print(tiler_mn)

        N = self.hidden_size
        input_layout = cute.make_layout((M, N), stride=(N * 2, 1))
        output_layout = cute.make_layout((M, N), stride=(N, 1))
        tma_aligned_M = ((M + 3) // 4) * 4

        mX = cute.make_tensor(X_ptr, input_layout)
        mG = cute.make_tensor(G_ptr, input_layout)
        mO = cute.make_tensor(O_ptr, output_layout)
        mS = cute.make_tensor(S_ptr, cute.make_layout((M, self.num_sf_blocks_per_row), stride=(1, tma_aligned_M)))

        self.kernel(
            mX, mG, mO, mS, M, eps, tv_layout, tiler_mn
        ).launch(
            grid=(cute.ceil_div(M, self.rows_per_block), 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )
    
    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mG: cute.Tensor,
        mO: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        token_idx = tidx // cute.arch.WARP_SIZE + bidx * (self.num_threads // cute.arch.WARP_SIZE)

        H = self.hidden_size

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        sG = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cutlass.const_expr(0)))
        gG = cute.local_tile(mG, tiler_mn, (bidx, cutlass.const_expr(0)))
        gO = cute.local_tile(mO, tiler_mn, (bidx, cutlass.const_expr(0)))

        cX = cute.local_tile(idX, tiler_mn, (bidx, cutlass.const_expr(0)))

        print(f"[DSL INFO]  gO = {gO.type}")
        print(f"[DSL INFO]  gX = {gX.type}")
        print(f"[DSL INFO]  gG = {gG.type}")

        copy_atom_load_async = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=COPY_BITS)

        copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=COPY_BITS // 2)

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)
        
        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgG = thr_copy_X.partition_S(gG)
        tXsG = thr_copy_X.partition_D(sG)
        tXcX = thr_copy_X.partition_S(cX)

        tXgO = thr_copy_O.partition_D(gO)

        tXrX = cute.make_fragment_like(tXgX)
        tXrG = cute.make_fragment_like(tXgG)
        tXrO = cute.make_fragment_like(tXgO)
        
        # scale = cute.make_rmem_tensor()

        tXpX = predicate_k(tXcX, limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        print(f"[DSL INFO]  tXgX = {tXgX.type}")
        print(f"[DSL INFO]  tXrX = {tXrX.type}")

        print(f"[DSL INFO]  tXgG = {tXgG.type}")
        print(f"[DSL INFO]  tXsG = {tXsG.type}")

        print(f"[DSL INFO]  tXgO = {tXgO.type}")
        print(f"[DSL INFO]  tXrO = {tXrO.type}")


        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)
            cute.copy(copy_atom_load_async, tXgG, tXsG, pred=tXpX)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tXsX, tXrX)
        cute.autovec_copy(tXsG, tXrG)

        print(f"[DSL INFO]  tXrX = {tXrX.shape[0][0]}")

        # Loop Reduce
        for j in cutlass.range_constexpr(tXrX.shape[0][1]):
            for i in cutlass.range_constexpr(tXrX.shape[0][0]):
                tXrO[i, j] = silu(tXrX[i, j]) * tXrG[i, j]

            

                

                






        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)


def fused_silu_mul_fp8_quant_packed(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    group_size: int = 128,
    eps: float = 1e-10,
):
    assert input.dim() == 2, "Input must be 2D tensor"
    assert input.is_contiguous(), "Input must be contiguous"
    assert input.dtype == torch.bfloat16, "Input must be bfloat16"

    M, N = input.shape
    N_2 = N // 2  # Output hidden dimension

    assert N_2 % group_size == 0, f"N//2 ({N_2}) must be divisible by group_size ({group_size})"

    # Get FP8 info
    fp8_dtype = torch.float8_e4m3fn

    # Compute dimensions
    num_groups_per_row = N_2 // group_size
    num_packed_groups = (num_groups_per_row + 3) // 4
    tma_aligned_M = ((M + 3) // 4) * 4

    # Allocate output tensors
    if output is None:
        output_q = torch.empty((M, N_2), dtype=fp8_dtype, device=input.device)
    else:
        output_q = output

    # Packed scales with TMA-aligned stride
    output_scale_packed = torch.zeros(
        (num_packed_groups, tma_aligned_M),
        dtype=torch.int32,
        device=input.device,
    ).T  # View as [tma_aligned_M, num_packed_groups] with stride (1, tma_aligned_M)

    kernel = SiluMulFP8QuantPackedKernel(
        dtype=BFloat16,
        hidden_size=N // 2,
        block_size=group_size,
        is_ue8m0=True,
        is_input_interleaved=False,
    )

    x_ptr = cute_rt.make_ptr(cutlass.BFloat16, input.data_ptr(), mem_space=AddressSpace.gmem, assumed_align=16)
    g_ptr = cute_rt.make_ptr(cutlass.BFloat16, input[..., N_2:].data_ptr(), mem_space=AddressSpace.gmem, assumed_align=16)
    o_ptr = cute_rt.make_ptr(cutlass.Float8E4M3FN, output_q.data_ptr(), mem_space=AddressSpace.gmem, assumed_align=16)
    s_ptr = cute_rt.make_ptr(cutlass.Int32, output_scale_packed.data_ptr(), mem_space=AddressSpace.gmem, assumed_align=16)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel(
        X_ptr=x_ptr,
        G_ptr=g_ptr,
        O_ptr=o_ptr,
        S_ptr=s_ptr,
        M=M,
        eps=eps,
        stream=current_stream,
    )

    return output_q, output_scale_packed[:M, :]


def main():
    L = 4
    d = 4096
    device = "cuda"
    x = torch.randn(L, d, dtype=torch.bfloat16, device=device)

    output_q, output_scale_packed = fused_silu_mul_fp8_quant_packed(x)
    # print("Output (quantized):", output_q)
    # print("Output scales (packed):", output_scale_packed)

if __name__ == "__main__":
    main()

