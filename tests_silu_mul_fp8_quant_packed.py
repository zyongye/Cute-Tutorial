"""
Tests for fused silu_mul + per_token_group_quant_fp8_packed kernel.

This kernel fuses:
1. SiLU activation + element-wise multiplication (gated activation)
2. Per-token-group FP8 quantization with UE8M0 (power-of-2) scales
3. Pack 4 UE8M0 exponents into int32 for DeepGEMM
"""

import pytest
import torch
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
DTYPES = [torch.bfloat16, torch.float16]
# Shapes are (num_tokens, 2 * hidden_dim) - input has gate and up concatenated
SHAPES = [
    (1, 256),      # Single token, small hidden
    (4, 512),      # Small batch
    (16, 1024),    # Medium batch
    (32, 2048),    # Larger batch
    (64, 4096),    # Large hidden dim
    (128, 8192),   # Very large hidden dim
]
GROUP_SIZES = [128]  # DeepGEMM uses group size 128
SEEDS = [42]


def get_fp8_info():
    """Get FP8 min/max values."""
    finfo = torch.finfo(FP8_DTYPE)
    return finfo.min, finfo.max


# ---------------------------------------------------------------------------
# Triton reference kernel
# ---------------------------------------------------------------------------

@triton.jit
def _silu_mul_quant_fp8_packed_kernel(
    # Input tensor [M, N] where N = 2 * hidden_dim
    input_ptr,
    # Output quantized tensor [M, N // 2]
    output_q_ptr,
    # Output packed scales [M, num_packed_groups] with TMA-aligned stride
    output_scale_ptr,
    # Dimensions
    M,  # Number of tokens
    # Strides
    input_stride_m,
    output_q_stride_m,
    output_scale_stride_k,  # TMA-aligned stride for packed scales
    # Quantization parameters
    eps,
    # Compile-time constants
    N: tl.constexpr,  # Input hidden dimension (2 * output hidden dimension)
    NUM_GROUPS: tl.constexpr,  # Number of groups per row (N // 2 // GROUP_SIZE)
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Fused Triton kernel for:
    1. SiLU activation + element-wise multiplication (gated activation)
    2. Per-token-group FP8 quantization with UE8M0 (power-of-2) scales
    3. Pack 4 UE8M0 exponents into int32 for DeepGEMM

    Grid: (num_packed_groups, ceil(M / BLOCK_M))
    Each thread block processes BLOCK_M rows and 4 groups (one packed int32).
    """
    # Compile-time constant for output hidden dimension
    N_2: tl.constexpr = N // 2

    # Program IDs
    pid_pack = tl.program_id(0)  # Which packed group (0, 1, 2, ...)
    pid_m = tl.program_id(1)     # Which block of rows
    m_offset = pid_m * BLOCK_M

    # Early exit if out of bounds
    if m_offset >= M:
        return

    # Precompute offsets (reused across groups)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, GROUP_SIZE)
    row_mask = (m_offset + offs_m) < M

    # Base pointers for this row block
    base_row_offset = (m_offset + offs_m[:, None]) * input_stride_m
    base_out_offset = (m_offset + offs_m[:, None]) * output_q_stride_m

    # Initialize packed scale value
    packed_scale = tl.zeros((BLOCK_M,), dtype=tl.int32)

    # Process 4 groups for this packed int32
    # Use tl.static_range for compile-time unrolling
    for pack_idx in tl.static_range(4):
        group_id = pid_pack * 4 + pack_idx

        # Check if this group exists (compile-time check when possible)
        if group_id < NUM_GROUPS:
            n_offset = group_id * GROUP_SIZE

            # Load input for SiLU part: input[:, :N_2]
            act_ptrs = input_ptr + base_row_offset + n_offset + offs_n[None, :]
            act_in = tl.load(act_ptrs, mask=row_mask[:, None], other=0.0)

            # Load input for mul part: input[:, N_2:]
            mul_ptrs = act_ptrs + N_2
            mul_in = tl.load(mul_ptrs, mask=row_mask[:, None], other=0.0)

            # SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
            # Fused computation to reduce register pressure
            act_f32 = act_in.to(tl.float32)
            y = (act_f32 / (1.0 + tl.exp(-act_f32))) * mul_in.to(tl.float32)

            # Per-group quantization with UE8M0 scales
            # Find max absolute value in each row for this group
            absmax = tl.max(tl.abs(y), axis=1)  # [BLOCK_M]

            # Compute UE8M0 scale: power-of-2 scale
            # scale = 2^ceil(log2(max(absmax/fp8_max, 1e-10)))
            scale_raw = tl.maximum(absmax / fp8_max, 1e-10)
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)

            # Quantize: y_q = clamp(y / scale, fp8_min, fp8_max)
            y_q = tl.clamp(y / scale[:, None], fp8_min, fp8_max)

            # Store quantized output
            out_q_ptrs = output_q_ptr + base_out_offset + n_offset + offs_n[None, :]
            tl.store(out_q_ptrs, y_q.to(output_q_ptr.dtype.element_ty), mask=row_mask[:, None])

            # Compute UE8M0 exponent (biased by 127) and pack
            # Clamp in float32 before converting to int32 (tl.clamp only supports float)
            exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            packed_scale = packed_scale | (exponent_biased << (pack_idx * 8))

    # Store the packed scale value
    # Scale layout: [M, num_packed_groups] with stride (1, tma_aligned_mn)
    scale_ptrs = output_scale_ptr + pid_pack * output_scale_stride_k + m_offset + offs_m
    tl.store(scale_ptrs, packed_scale, mask=row_mask)


def silu_mul_quant_fp8_packed_triton(
    input: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    output_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused SiLU+mul activation and FP8 quantization with packed UE8M0 scales.

    This Triton kernel is equivalent to:
        act_out = silu_and_mul(input)
        a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(act_out, group_size)

    Args:
        input: Input tensor of shape [M, N] where N = 2 * hidden_dim
        group_size: Quantization group size (default 128)
        eps: Small value to avoid division by zero
        output_q: Optional pre-allocated output tensor for quantized values

    Returns:
        (output_q, output_scale_packed):
            output_q: FP8 tensor of shape [M, N // 2]
            output_scale_packed: Int32 tensor with packed UE8M0 scales
                                 Shape: [M, ceil(num_groups_per_row / 4)]
                                 Stride: (1, tma_aligned_M)
    """
    assert input.dim() == 2, "Input must be 2D tensor"
    assert input.is_contiguous(), "Input must be contiguous"

    M, N = input.shape
    N_2 = N // 2  # Output hidden dimension

    assert N_2 % group_size == 0, f"N//2 ({N_2}) must be divisible by group_size ({group_size})"

    # Get FP8 info
    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    # Compute dimensions
    num_groups_per_row = N_2 // group_size
    num_packed_groups = (num_groups_per_row + 3) // 4
    tma_aligned_M = ((M + 3) // 4) * 4

    # Allocate output tensors
    if output_q is None:
        output_q = torch.empty((M, N_2), dtype=fp8_dtype, device=input.device)

    # Packed scales with TMA-aligned stride
    output_scale_packed = torch.zeros(
        (num_packed_groups, tma_aligned_M),
        dtype=torch.int32,
        device=input.device,
    ).T[:M, :]  # View as [M, num_packed_groups] with stride (1, tma_aligned_M)

    # Launch kernel with 2D grid: (num_packed_groups, ceil(M / BLOCK_M))
    BLOCK_M = 8
    grid = (num_packed_groups, (M + BLOCK_M - 1) // BLOCK_M)

    # Tuning parameters
    # num_warps: 4 warps (128 threads) is good for GROUP_SIZE=128
    # num_stages: software pipelining for memory latency hiding
    num_warps = max(4, group_size // 32)
    num_stages = 2

    _silu_mul_quant_fp8_packed_kernel[grid](
        input,
        output_q,
        output_scale_packed,
        M,
        input.stride(0),
        output_q.stride(0),
        output_scale_packed.stride(1),
        eps,
        # Compile-time constants
        N=N,
        NUM_GROUPS=num_groups_per_row,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output_q, output_scale_packed


# ---------------------------------------------------------------------------
# CuTe DSL kernel wrapper
# ---------------------------------------------------------------------------

def cutedsl_silu_mul_fp8_quant_packed(
    input: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call the CuTe DSL fused kernel.
    """
    from kernels.silu_mul_fp8_quant_pakced import (
        fused_silu_mul_fp8_quant_packed as cutedsl_fused,
    )

    output_q, output_s_packed = cutedsl_fused(
        input, group_size=group_size, eps=eps
    )
    return output_q, output_s_packed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unpack_ue8m0_scales(packed_scales: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Unpack UE8M0 exponents from int32 to float32 scales.

    Each int32 contains 4 packed 8-bit exponents.
    The scale is reconstructed as: scale = 2^(exponent - 127)
    """
    M = packed_scales.shape[0]
    device = packed_scales.device

    # Create output tensor
    scales = torch.zeros((M, num_groups), dtype=torch.float32, device=device)

    # Unpack each int32 into 4 exponents
    for pack_idx in range(packed_scales.shape[1]):
        packed = packed_scales[:, pack_idx].int()
        for i in range(4):
            group_idx = pack_idx * 4 + i
            if group_idx < num_groups:
                exponent = (packed >> (i * 8)) & 0xFF
                # Reconstruct scale from biased exponent
                # scale = 2^(exponent - 127)
                scales[:, group_idx] = torch.pow(
                    2.0, exponent.float() - 127.0
                )

    return scales


def dequantize_fp8_with_scales(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize FP8 tensor using per-group scales.
    """
    M, N = quantized.shape
    num_groups = N // group_size

    # Expand scales to match hidden dimension
    scales_expanded = scales.unsqueeze(-1).expand(M, num_groups, group_size)
    scales_expanded = scales_expanded.reshape(M, N)

    # Dequantize
    dequantized = quantized.to(torch.float32) * scales_expanded

    return dequantized.to(dtype)


# ---------------------------------------------------------------------------
# Tests: Triton kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_triton_silu_mul_fp8_quant_packed_correctness(
    dtype: torch.dtype,
    shape: tuple[int, int],
    group_size: int,
    seed: int,
) -> None:
    """
    Test that the Triton kernel produces mathematically correct output
    by comparing against a pure-Python silu_mul reference.
    """
    torch.manual_seed(seed)
    device = "cuda:0"
    torch.set_default_device(device)

    num_tokens, hidden_2x = shape
    hidden_dim = hidden_2x // 2
    if hidden_dim % group_size != 0:
        pytest.skip(f"hidden_dim {hidden_dim} not divisible by group_size {group_size}")

    # Create input tensor
    input_tensor = torch.randn(num_tokens, hidden_2x, dtype=dtype, device=device)

    # Compute expected silu_mul output (pure Python reference)
    gate = input_tensor[:, :hidden_dim]
    up = input_tensor[:, hidden_dim:]
    silu_gate = gate * torch.sigmoid(gate.float()).to(dtype)
    expected_silu_mul = silu_gate * up

    # Triton kernel
    triton_output_q, triton_output_s_packed = silu_mul_quant_fp8_packed_triton(
        input_tensor, group_size
    )

    # Check dtypes
    assert triton_output_q.dtype == FP8_DTYPE
    assert triton_output_s_packed.dtype == torch.int32

    # Unpack scales
    num_groups = hidden_dim // group_size
    triton_scales = unpack_ue8m0_scales(triton_output_s_packed, num_groups)

    # Dequantize
    triton_dequant = dequantize_fp8_with_scales(
        triton_output_q, triton_scales, group_size, dtype
    )

    # Compare with expected silu_mul output
    # Allow for quantization error
    torch.testing.assert_close(
        expected_silu_mul.float(),
        triton_dequant.float(),
        atol=2.0,
        rtol=0.15,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_triton_silu_mul_fp8_quant_packed_scale_values(
    dtype: torch.dtype,
) -> None:
    """
    Test that scales are correctly computed as UE8M0 (power-of-2) values.
    """
    torch.manual_seed(42)
    device = "cuda:0"
    group_size = 128

    # Create input with known range
    num_tokens = 8
    hidden_dim = 256
    input_tensor = torch.randn(num_tokens, hidden_dim * 2, dtype=dtype, device=device)

    triton_output_q, triton_output_s_packed = silu_mul_quant_fp8_packed_triton(
        input_tensor, group_size
    )

    # Unpack scales
    num_groups = hidden_dim // group_size
    scales = unpack_ue8m0_scales(triton_output_s_packed, num_groups)

    # All scales should be powers of 2
    # log2(scale) should be an integer
    log2_scales = torch.log2(scales)
    rounded_log2 = torch.round(log2_scales)

    # Check that log2 of scales are integers (within floating point tolerance)
    torch.testing.assert_close(
        log2_scales, rounded_log2, atol=1e-5, rtol=1e-5
    )

    # Scales should be positive
    assert (scales > 0).all(), "All scales should be positive"


# ---------------------------------------------------------------------------
# Tests: CuTe DSL kernel vs Triton reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_cutedsl_silu_mul_fp8_quant_packed(
    shape: tuple[int, int],
    group_size: int,
    seed: int,
) -> None:
    """
    Test that the CuTe DSL fused kernel produces the same output as the
    Triton reference kernel.
    """
    dtype = torch.bfloat16  # CuTe DSL kernel only supports bfloat16
    torch.manual_seed(seed)
    device = "cuda:0"
    torch.set_default_device(device)

    num_tokens, hidden_2x = shape
    hidden_dim = hidden_2x // 2
    if hidden_dim % group_size != 0:
        pytest.skip(f"hidden_dim {hidden_dim} not divisible by group_size {group_size}")

    # Create input tensor
    input_tensor = torch.randn(num_tokens, hidden_2x, dtype=dtype, device=device)

    # Triton reference
    ref_output_q, ref_output_s_packed = silu_mul_quant_fp8_packed_triton(
        input_tensor, group_size
    )

    # CuTe DSL implementation
    cutedsl_output_q, cutedsl_output_s_packed = cutedsl_silu_mul_fp8_quant_packed(
        input_tensor, group_size
    )

    # Check dtypes
    assert cutedsl_output_q.dtype == FP8_DTYPE
    assert cutedsl_output_s_packed.dtype == torch.int32

    # Check shapes
    assert ref_output_q.shape == cutedsl_output_q.shape
    assert ref_output_s_packed.shape == cutedsl_output_s_packed.shape

    # Unpack scales for comparison
    num_groups = hidden_dim // group_size
    ref_scales = unpack_ue8m0_scales(ref_output_s_packed, num_groups)
    cutedsl_scales = unpack_ue8m0_scales(cutedsl_output_s_packed, num_groups)

    # Check scales are close (UE8M0 should be exact since they're power-of-2)
    torch.testing.assert_close(ref_scales, cutedsl_scales, atol=1e-6, rtol=1e-6)

    # Check quantized values
    ref_q_float = ref_output_q.to(torch.float32)
    cutedsl_q_float = cutedsl_output_q.to(torch.float32)

    # Allow for small differences due to rounding
    torch.testing.assert_close(ref_q_float, cutedsl_q_float, atol=1.0, rtol=0.1)

    # Dequantize and compare
    ref_dequant = dequantize_fp8_with_scales(ref_output_q, ref_scales, group_size, dtype)
    cutedsl_dequant = dequantize_fp8_with_scales(
        cutedsl_output_q, cutedsl_scales, group_size, dtype
    )

    # Dequantized values should be close
    torch.testing.assert_close(ref_dequant, cutedsl_dequant, atol=0.5, rtol=0.1)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@torch.inference_mode()
def test_cutedsl_silu_mul_fp8_quant_packed_correctness(
    shape: tuple[int, int],
    group_size: int,
) -> None:
    """
    Test that the CuTe DSL fused kernel produces mathematically correct output
    by comparing against a pure-Python silu_mul reference.
    """
    dtype = torch.bfloat16  # CuTe DSL kernel only supports bfloat16
    torch.manual_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    num_tokens, hidden_2x = shape
    hidden_dim = hidden_2x // 2
    if hidden_dim % group_size != 0:
        pytest.skip(f"hidden_dim {hidden_dim} not divisible by group_size {group_size}")

    # Create input tensor
    input_tensor = torch.randn(num_tokens, hidden_2x, dtype=dtype, device=device)

    # Compute expected silu_mul output
    gate = input_tensor[:, :hidden_dim]
    up = input_tensor[:, hidden_dim:]
    silu_gate = gate * torch.sigmoid(gate.float()).to(dtype)
    expected_silu_mul = silu_gate * up

    # Get CuTe DSL output
    cutedsl_output_q, cutedsl_output_s_packed = cutedsl_silu_mul_fp8_quant_packed(
        input_tensor, group_size
    )

    # Unpack scales
    num_groups = hidden_dim // group_size
    cutedsl_scales = unpack_ue8m0_scales(cutedsl_output_s_packed, num_groups)

    # Dequantize
    cutedsl_dequant = dequantize_fp8_with_scales(
        cutedsl_output_q, cutedsl_scales, group_size, dtype
    )

    # Compare with expected silu_mul output
    torch.testing.assert_close(
        expected_silu_mul.float(),
        cutedsl_dequant.float(),
        atol=2.0,
        rtol=0.15,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
