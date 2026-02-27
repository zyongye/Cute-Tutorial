"""
CUTE DSL utility operations for GPU kernels.

This module provides common DSL operations using PTX inline assembly:
- Math ops: tanh, fma, silu, fabs_f32
- FP8/UE8M0 quantization ops: cvt_f32_to_ue8m0, ue8m0_to_output_scale, fp8_quant_and_clamp, cvt_f32_to_e4m3
"""
from cutlass import Float32, Uint32, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """Fast reciprocal using PTX rcp.approx.ftz.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """Compute tanh using PTX tanh.approx.f32."""
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
    """Fused multiply-add: a * b + c using PTX fma.rn.f32."""
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
    SiLU activation: silu(a) = a * sigmoid(a)

    Computed as: (0.5 * a) * tanh(0.5 * a) + (0.5 * a)
    This compiles down to 3 SASS instructions: FMUL, MUFU.TANH, and FFMA.
    """
    a_half = 0.5 * a
    return fma(a_half, tanh(a_half), a_half)


@dsl_user_op
def silu_mul(x: Float32, g: Float32, *, loc=None, ip=None) -> Float32:
    """silu(x) * g using 3 instructions instead of 4."""
    a_half = Float32(0.5) * x
    a_half_g = a_half * g
    return fma(a_half_g, tanh(a_half), a_half_g)


@dsl_user_op
def fabs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """Compute absolute value of float32 using PTX abs.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_f32_to_ue8m0(max_val: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Convert float32 max value to UE8M0 scale factor.

    UE8M0 is unsigned 8-bit exponent-only format:
    - value = 2^(ue8m0 - 127)
    - ue8m0 = ceil(log2(max_val)) + 127

    Uses lg2.approx.f32 for fast log2 approximation.
    Uses cvt.rpi (round towards positive infinity, i.e., ceiling).
    Returns value clamped to [0, 255].
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(max_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg, p_ovf;
                .reg .f32 log2_val;
                .reg .s32 exp_int, result;

                // Check for zero/negative
                setp.le.f32 p_zero, $1, 0f00000000;

                // Compute ceil(log2(max_val)) using cvt.rpi (round towards +inf)
                lg2.approx.f32 log2_val, $1;
                cvt.rpi.s32.f32 exp_int, log2_val;

                // Add bias and clamp to [0, 255]
                add.s32 result, exp_int, 127;
                setp.lt.s32 p_neg, result, 0;
                setp.gt.s32 p_ovf, result, 255;
                selp.s32 result, 0, result, p_neg;
                selp.s32 result, 255, result, p_ovf;
                selp.s32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ue8m0_to_output_scale(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """
    Convert UE8M0 to output_scale for MXFP4 quantization.

    UE8M0 value = 2^(ue8m0 - 127)
    Returns 1 / 2^(ue8m0 - 127) = 2^(127 - ue8m0)
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .s32 neg_exp;
                .reg .f32 neg_exp_f, result;

                // Check for zero
                setp.eq.u32 p_zero, $1, 0;

                // Compute 2^(127 - ue8m0) = 1 / 2^(ue8m0 - 127)
                sub.s32 neg_exp, 127, $1;
                cvt.rn.f32.s32 neg_exp_f, neg_exp;
                ex2.approx.f32 result, neg_exp_f;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fp8_quant_and_clamp(val: float | Float32, scale: float | Float32, *, loc=None, ip=None) -> Float32:
    """Multiply val by scale and clamp to [-448, 448] range for FP8 E4M3."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(val).ir_value(loc=loc, ip=ip),
                Float32(scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_neg, p_ovf;
                .reg .f32 result;

                // Compute val * scale_inv
                mul.f32 result, $1, $2;

                // Clamp to [-448, 448]
                setp.lt.f32 p_neg, result, 0fC3E00000;
                setp.gt.f32 p_ovf, result, 0f43E00000;
                selp.f32 result, 0fC3E00000, result, p_neg;
                selp.f32 $0, 0f43E00000, result, p_ovf;
            }
            """,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint8:
    """Convert float32 to E4M3 using native cvt.rn.satfinite.e4m3x2.f32.
    Returns raw FP8 bits as Uint8."""
    return Uint8(
        llvm.inline_asm(
            T.i8(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 $0, zero, $1;
            }
            """,
            "=h,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
