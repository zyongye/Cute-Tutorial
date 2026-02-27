"""
CUTE DSL SiLU+Mul+FP8 Quantization Kernel.

This module provides a fused kernel for:
1. SiLU activation + element-wise multiplication (gated activation)
2. Per-token-group FP8 quantization with UE8M0 (power-of-2) scales
3. Pack 4 UE8M0 exponents into int32 for DeepGEMM

Usage:
    from silu_mul_fp8_quant_pakced import fused_silu_mul_fp8_quant_packed

    # Input: [M, N] bf16 tensor where N = 2 * hidden_dim
    output_q, output_scale = fused_silu_mul_fp8_quant_packed(input, group_size=128)

For tests and benchmarks, see test_silu_mul_fp8.py
"""
import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
from cutlass.cute.typing import AddressSpace
from cutlass import Float32, Int32, BFloat16, Uint32, Uint8

from .utils import (
    silu,
    fabs_f32,
    cvt_f32_to_ue8m0,
    ue8m0_to_output_scale,
    fp8_quant_and_clamp,
    cvt_f32_to_e4m3,
    rcp_approx_ftz,
)


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

        self.NUM_CHUNK = 2

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
            self.num_vec_blocks // self.NUM_CHUNK,
        )

        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile // self.NUM_CHUNK)

        N = self.hidden_size
        input_layout = cute.make_layout((M, N), stride=(N * 2, 1))
        output_layout = cute.make_layout((M, N), stride=(N, 1))
        tma_aligned_M = ((M + 3) // 4) * 4

        mX = cute.make_tensor(X_ptr, input_layout)
        mG = cute.make_tensor(G_ptr, input_layout)
        mO = cute.make_tensor(O_ptr, output_layout)
        mS = cute.make_tensor(
            S_ptr,
            cute.make_layout(
                ((4, M), self.num_sf_blocks_per_row // 4),
                stride=((1, 4), tma_aligned_M * 4),
            )
        )

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

        lane_id = tidx % cute.arch.WARP_SIZE

        token_idx = tidx // cute.arch.WARP_SIZE + bidx * (self.num_threads // cute.arch.WARP_SIZE)

        H = self.hidden_size
        fp8_max_rcp = rcp_approx_ftz(Float32(FP8_MAX))

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

        for chunk_ids in range(self.NUM_CHUNK):

            gX = cute.local_tile(mX, tiler_mn, (bidx, chunk_ids))
            gG = cute.local_tile(mG, tiler_mn, (bidx, chunk_ids))
            gO = cute.local_tile(mO, tiler_mn, (bidx, chunk_ids))

            cX = cute.local_tile(idX, tiler_mn, (bidx, chunk_ids))

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

            # act_result = cute.make_fragment_like(tXgO, dtype=Float32)

            tXpX = predicate_k(tXcX, limit=H)
            row_coord = tXcX[(0, 0), 0, 0]
            row_in_bounds = row_coord[0] < M

            if row_in_bounds:
                cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_load_async, tXgG, tXsG, pred=tXpX)

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tXsG, tXrG)

            scales_per_chunk = tXrX.shape[0][1] // 2
            # Loop Reduce
            for scale_idx in range(scales_per_chunk):
                global_scale_idx = chunk_ids * scales_per_chunk + scale_idx
                max_val: Float32 = 0.0
                # one thread has 8 element, 16 threads got one scale value
                # one output int32 has 4 ue8m0 scale
                # so 2 thread loop will produce 1 int
                for pos_in_one_scale in cutlass.range_constexpr(2):
                    j = scale_idx * 2 + pos_in_one_scale
                    for i in cutlass.range_constexpr(tXrX.shape[0][0]):
                        act_result = silu(tXrX[(i, j), 0, 0]) * tXrG[(i, j), 0, 0]
                        # Use fabs only for max computation, keep signed value in act_result
                        max_val = max(fabs_f32(act_result), max_val)

                    max_val = cute.arch.warp_reduction(
                        max_val,
                        max,
                        threads_in_group=16,    # need to be determined in program
                    )

                    # change division to rcp
                    scale_raw = max(max_val * fp8_max_rcp, eps)
                    scale_ue8m0 = cvt_f32_to_ue8m0(scale_raw)
                    scale_u8 = Uint8(scale_ue8m0 & Uint32(0xFF))
                    inv_scale = ue8m0_to_output_scale(scale_ue8m0)

                    # quant_each_value
                    for i in cutlass.range_constexpr(tXrX.shape[0][0]):
                        act_result = silu(tXrX[(i, j), 0, 0]) * tXrG[(i, j), 0, 0]
                        quant_out_f32 = fp8_quant_and_clamp(act_result, inv_scale)
                        # output e4m3 as uint8
                        quant_out_fp8 = cvt_f32_to_e4m3(quant_out_f32)

                        tXrO[(i, j), 0, 0] = quant_out_fp8

                    # handling scale
                    if lane_id == 0 or lane_id == 16:
                        pos = pos_in_one_scale * 2 + lane_id // 16
                        mS[(pos, token_idx), global_scale_idx] = scale_u8

            if row_in_bounds:
                cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)


def fused_silu_mul_fp8_quant_packed(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    group_size: int = 128,
    eps: float = 1e-10,
):
    """
    Fused SiLU+Mul activation and FP8 quantization with packed UE8M0 scales.

    This kernel performs:
    1. SiLU activation on the first half of the input
    2. Element-wise multiplication with the second half
    3. Per-group FP8 quantization with UE8M0 scales
    4. Pack 4 UE8M0 exponents into int32 for DeepGEMM

    Args:
        input: Input tensor of shape [M, N] where N = 2 * hidden_dim, dtype=bfloat16
        output: Optional pre-allocated output tensor for quantized values
        group_size: Quantization group size (default 128)
        eps: Small value to avoid division by zero

    Returns:
        (output_q, output_scale_packed):
            output_q: FP8 tensor of shape [M, N // 2]
            output_scale_packed: Int32 tensor with packed UE8M0 scales
                                 Shape: [M, ceil(num_groups_per_row / 4)]
    """
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

    x_ptr = cute_rt.make_ptr(cutlass.BFloat16, input.data_ptr(), mem_space=AddressSpace.gmem, assumed_align=16)
    g_ptr = cute_rt.make_ptr(cutlass.BFloat16, input[..., N_2:].data_ptr(), mem_space=AddressSpace.gmem, assumed_align=16)
    o_ptr = cute_rt.make_ptr(cutlass.Uint8, output_q.data_ptr(), mem_space=AddressSpace.gmem, assumed_align=16)
    s_ptr = cute_rt.make_ptr(cutlass.Uint8, output_scale_packed.data_ptr(), mem_space=AddressSpace.gmem, assumed_align=16)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        N,
        group_size,
    )
    if compile_key not in fused_silu_mul_fp8_quant_packed.compile_cache:
        kernel = SiluMulFP8QuantPackedKernel(
            dtype=BFloat16,
            hidden_size=N // 2,
            block_size=group_size,
            is_ue8m0=True,
            is_input_interleaved=False,
        )
        fused_silu_mul_fp8_quant_packed.compile_cache[compile_key] = cute.compile(
            kernel,
            x_ptr,
            g_ptr,
            o_ptr,
            s_ptr,
            M,
            eps,
            current_stream,
            options="--enable-tvm-ffi",
        )

    fused_silu_mul_fp8_quant_packed.compile_cache[compile_key](
        x_ptr,
        g_ptr,
        o_ptr,
        s_ptr,
        M,
        eps,
        current_stream,
    )

    return output_q, output_scale_packed[:M, :]

fused_silu_mul_fp8_quant_packed.compile_cache = {}
