#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockSize = 16;
constexpr int kGroupSize = kWarpSize * kBlockSize;
constexpr int kColumnsPerWarp = 2;
constexpr bool kPackedGateUpWeight = true;
constexpr bool kPackedGateUpScale = true;

bool is_aligned(const void* ptr, uintptr_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

struct U64x4 {
  uint64_t x;
  uint64_t y;
  uint64_t z;
  uint64_t w;
};

__device__ __forceinline__ U64x4 ld_global_u64x4(const void* ptr) {
  U64x4 out;
  asm volatile(
      "ld.global.v4.u64 {%0,%1,%2,%3}, [%4];"
      : "=l"(out.x), "=l"(out.y), "=l"(out.z), "=l"(out.w)
      : "l"(ptr));
  return out;
}

__device__ __forceinline__ float tanh_approx_f32(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float e4m3_scalar_to_f32(uint8_t x) {
  float y;
  uint32_t packed = static_cast<uint32_t>(x);
  asm(
      "{\n\t"
      ".reg .b8 b0, b1;\n\t"
      ".reg .b16 fp8x2, h0;\n\t"
      ".reg .b32 h2;\n\t"
      "mov.b32 {b0, b1, _, _}, %1;\n\t"
      "mov.b16 fp8x2, {b0, b1};\n\t"
      "cvt.rn.f16x2.e4m3x2 h2, fp8x2;\n\t"
      "mov.b32 {h0, _}, h2;\n\t"
      "cvt.f32.f16 %0, h0;\n\t"
      "}"
      : "=f"(y)
      : "r"(packed));
  return y;
}

__device__ __forceinline__ float2 e4m3x2_to_f32x2(uint16_t x) {
  float2 y;
  uint32_t packed = static_cast<uint32_t>(x);
  asm(
      "{\n\t"
      ".reg .b8 b0, b1;\n\t"
      ".reg .b16 fp8x2, h0, h1;\n\t"
      ".reg .b32 h2;\n\t"
      "mov.b32 {b0, b1, _, _}, %2;\n\t"
      "mov.b16 fp8x2, {b0, b1};\n\t"
      "cvt.rn.f16x2.e4m3x2 h2, fp8x2;\n\t"
      "mov.b32 {h0, h1}, h2;\n\t"
      "cvt.f32.f16 %0, h0;\n\t"
      "cvt.f32.f16 %1, h1;\n\t"
      "}"
      : "=f"(y.x), "=f"(y.y)
      : "r"(packed));
  return y;
}

__device__ __forceinline__ float fp4x16_bf16_dot_scaled_accum(
    uint32_t w_lo,
    uint32_t w_hi,
    uint32_t x0,
    uint32_t x1,
    uint32_t x2,
    uint32_t x3,
    uint32_t x4,
    uint32_t x5,
    uint32_t x6,
    uint32_t x7,
    float scale,
    float acc) {
  float out;
  asm(
      "{\n\t"
      ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
      ".reg .b16 wh0, wh1, xh0, xh1;\n\t"
      ".reg .b32 wb2;\n\t"
      ".reg .f32 sum0, sum1, sum;\n\t"
      "mov.f32 sum0, 0f00000000;\n\t"
      "mov.f32 sum1, 0f00000000;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %1;\n\t"
      "mov.b32 {b4, b5, b6, b7}, %2;\n\t"

      "cvt.rn.satfinite.bf16x2.e2m1x2 wb2, b0;\n\t"
      "mov.b32 {wh0, wh1}, wb2;\n\t"
      "mov.b32 {xh0, xh1}, %3;\n\t"
      "fma.rn.f32.bf16 sum0, wh0, xh0, sum0;\n\t"
      "fma.rn.f32.bf16 sum1, wh1, xh1, sum1;\n\t"

      "cvt.rn.satfinite.bf16x2.e2m1x2 wb2, b1;\n\t"
      "mov.b32 {wh0, wh1}, wb2;\n\t"
      "mov.b32 {xh0, xh1}, %4;\n\t"
      "fma.rn.f32.bf16 sum0, wh0, xh0, sum0;\n\t"
      "fma.rn.f32.bf16 sum1, wh1, xh1, sum1;\n\t"

      "cvt.rn.satfinite.bf16x2.e2m1x2 wb2, b2;\n\t"
      "mov.b32 {wh0, wh1}, wb2;\n\t"
      "mov.b32 {xh0, xh1}, %5;\n\t"
      "fma.rn.f32.bf16 sum0, wh0, xh0, sum0;\n\t"
      "fma.rn.f32.bf16 sum1, wh1, xh1, sum1;\n\t"

      "cvt.rn.satfinite.bf16x2.e2m1x2 wb2, b3;\n\t"
      "mov.b32 {wh0, wh1}, wb2;\n\t"
      "mov.b32 {xh0, xh1}, %6;\n\t"
      "fma.rn.f32.bf16 sum0, wh0, xh0, sum0;\n\t"
      "fma.rn.f32.bf16 sum1, wh1, xh1, sum1;\n\t"

      "cvt.rn.satfinite.bf16x2.e2m1x2 wb2, b4;\n\t"
      "mov.b32 {wh0, wh1}, wb2;\n\t"
      "mov.b32 {xh0, xh1}, %7;\n\t"
      "fma.rn.f32.bf16 sum0, wh0, xh0, sum0;\n\t"
      "fma.rn.f32.bf16 sum1, wh1, xh1, sum1;\n\t"

      "cvt.rn.satfinite.bf16x2.e2m1x2 wb2, b5;\n\t"
      "mov.b32 {wh0, wh1}, wb2;\n\t"
      "mov.b32 {xh0, xh1}, %8;\n\t"
      "fma.rn.f32.bf16 sum0, wh0, xh0, sum0;\n\t"
      "fma.rn.f32.bf16 sum1, wh1, xh1, sum1;\n\t"

      "cvt.rn.satfinite.bf16x2.e2m1x2 wb2, b6;\n\t"
      "mov.b32 {wh0, wh1}, wb2;\n\t"
      "mov.b32 {xh0, xh1}, %9;\n\t"
      "fma.rn.f32.bf16 sum0, wh0, xh0, sum0;\n\t"
      "fma.rn.f32.bf16 sum1, wh1, xh1, sum1;\n\t"

      "cvt.rn.satfinite.bf16x2.e2m1x2 wb2, b7;\n\t"
      "mov.b32 {wh0, wh1}, wb2;\n\t"
      "mov.b32 {xh0, xh1}, %10;\n\t"
      "fma.rn.f32.bf16 sum0, wh0, xh0, sum0;\n\t"
      "fma.rn.f32.bf16 sum1, wh1, xh1, sum1;\n\t"

      "add.rn.f32 sum, sum0, sum1;\n\t"
      "fma.rn.f32 %0, sum, %11, %12;\n\t"
      "}"
      : "=f"(out)
      : "r"(w_lo), "r"(w_hi), "r"(x0), "r"(x1), "r"(x2), "r"(x3),
        "r"(x4), "r"(x5), "r"(x6), "r"(x7), "f"(scale), "f"(acc));
  return out;
}

__device__ __forceinline__ float warp_sum(float x) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(mask, x, offset);
  }
  return x;
}

template <int B, int H, int I, int E, int kTopK>
__global__ __launch_bounds__(256, 6) void moe_gate_up_nvfp4_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ w_scale,
    const float* __restrict__ alpha,
    const int32_t* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out) {
  static_assert(H % kGroupSize == 0, "H must be a multiple of 512");
  static_assert(I % kColumnsPerWarp == 0, "I must be even");
  static_assert(kTopK * kWarpSize <= 1024, "topk is too large for one CTA");
  constexpr int kGroups = H / kGroupSize;
  int out_i0 = blockIdx.x * kColumnsPerWarp;
  int out_i1 = out_i0 + 1;
  int token = blockIdx.y;
  int tid = threadIdx.x;
  int warp = tid / kWarpSize;
  int lane = tid & (kWarpSize - 1);

  if (warp >= kTopK || token >= B) {
    return;
  }

  int expert = expert_ids[token * kTopK + warp];
  if (expert < 0 || expert >= E) {
    return;
  }

  float gate_acc0 = 0.0f;
  float up_acc0 = 0.0f;
  float gate_acc1 = 0.0f;
  float up_acc1 = 0.0f;

  const __nv_bfloat16* x_token = x + static_cast<int64_t>(token) * H;

  int64_t packed_w_base0 = (static_cast<int64_t>(expert) * I + out_i0) * H;
  int64_t packed_w_base1 = packed_w_base0 + H;
  int64_t gate_w_base0 =
      (static_cast<int64_t>(expert) * (2 * I) + out_i0) * (H / 2);
  int64_t up_w_base0 =
      (static_cast<int64_t>(expert) * (2 * I) + I + out_i0) * (H / 2);
  int64_t gate_w_base1 = gate_w_base0 + (H / 2);
  int64_t up_w_base1 = up_w_base0 + (H / 2);
  int64_t packed_s_base0 =
      (static_cast<int64_t>(expert) * I + out_i0) * (H / (kBlockSize / 2));
  int64_t packed_s_base1 = packed_s_base0 + (H / (kBlockSize / 2));
  int64_t gate_s_base0 =
      (static_cast<int64_t>(expert) * (2 * I) + out_i0) * (H / kBlockSize);
  int64_t up_s_base0 =
      (static_cast<int64_t>(expert) * (2 * I) + I + out_i0) *
      (H / kBlockSize);
  int64_t gate_s_base1 = gate_s_base0 + (H / kBlockSize);
  int64_t up_s_base1 = up_s_base0 + (H / kBlockSize);

  for (int group = 0; group < kGroups; ++group) {
    int x_offset = group * kGroupSize + lane * kBlockSize;
    U64x4 x_vec = ld_global_u64x4(x_token + x_offset);
    uint32_t x0 = static_cast<uint32_t>(x_vec.x);
    uint32_t x1 = static_cast<uint32_t>(x_vec.x >> 32);
    uint32_t x2 = static_cast<uint32_t>(x_vec.y);
    uint32_t x3 = static_cast<uint32_t>(x_vec.y >> 32);
    uint32_t x4 = static_cast<uint32_t>(x_vec.z);
    uint32_t x5 = static_cast<uint32_t>(x_vec.z >> 32);
    uint32_t x6 = static_cast<uint32_t>(x_vec.w);
    uint32_t x7 = static_cast<uint32_t>(x_vec.w >> 32);

    int scale_offset = group * kWarpSize + lane;

    float gate_scale0;
    float up_scale0;
    float gate_scale1;
    float up_scale1;
    if constexpr (kPackedGateUpScale) {
      uint16_t gate_up_scale0 = *reinterpret_cast<const uint16_t*>(
          w_scale + packed_s_base0 + scale_offset * 2);
      uint16_t gate_up_scale1 = *reinterpret_cast<const uint16_t*>(
          w_scale + packed_s_base1 + scale_offset * 2);
      float2 scales0 = e4m3x2_to_f32x2(gate_up_scale0);
      float2 scales1 = e4m3x2_to_f32x2(gate_up_scale1);
      gate_scale0 = scales0.x;
      up_scale0 = scales0.y;
      gate_scale1 = scales1.x;
      up_scale1 = scales1.y;
    } else {
      gate_scale0 = e4m3_scalar_to_f32(w_scale[gate_s_base0 + scale_offset]);
      up_scale0 = e4m3_scalar_to_f32(w_scale[up_s_base0 + scale_offset]);
      gate_scale1 = e4m3_scalar_to_f32(w_scale[gate_s_base1 + scale_offset]);
      up_scale1 = e4m3_scalar_to_f32(w_scale[up_s_base1 + scale_offset]);
    }

    if constexpr (kPackedGateUpWeight) {
      int packed_byte_offset = group * kGroupSize + lane * kBlockSize;
      uint4 gate_up_w0 = *reinterpret_cast<const uint4*>(
          w + packed_w_base0 + packed_byte_offset);
      uint4 gate_up_w1 = *reinterpret_cast<const uint4*>(
          w + packed_w_base1 + packed_byte_offset);
      gate_acc0 = fp4x16_bf16_dot_scaled_accum(
          gate_up_w0.x, gate_up_w0.y, x0, x1, x2, x3, x4, x5, x6, x7,
          gate_scale0, gate_acc0);
      up_acc0 = fp4x16_bf16_dot_scaled_accum(
          gate_up_w0.z, gate_up_w0.w, x0, x1, x2, x3, x4, x5, x6, x7,
          up_scale0, up_acc0);
      gate_acc1 = fp4x16_bf16_dot_scaled_accum(
          gate_up_w1.x, gate_up_w1.y, x0, x1, x2, x3, x4, x5, x6, x7,
          gate_scale1, gate_acc1);
      up_acc1 = fp4x16_bf16_dot_scaled_accum(
          gate_up_w1.z, gate_up_w1.w, x0, x1, x2, x3, x4, x5, x6, x7,
          up_scale1, up_acc1);
    } else {
      int byte_offset = group * (kGroupSize / 2) + lane * (kBlockSize / 2);
      uint2 gate_w0 =
          *reinterpret_cast<const uint2*>(w + gate_w_base0 + byte_offset);
      gate_acc0 = fp4x16_bf16_dot_scaled_accum(
          gate_w0.x, gate_w0.y, x0, x1, x2, x3, x4, x5, x6, x7, gate_scale0,
          gate_acc0);

      uint2 up_w0 =
          *reinterpret_cast<const uint2*>(w + up_w_base0 + byte_offset);
      up_acc0 = fp4x16_bf16_dot_scaled_accum(
          up_w0.x, up_w0.y, x0, x1, x2, x3, x4, x5, x6, x7, up_scale0,
          up_acc0);

      uint2 gate_w1 =
          *reinterpret_cast<const uint2*>(w + gate_w_base1 + byte_offset);
      gate_acc1 = fp4x16_bf16_dot_scaled_accum(
          gate_w1.x, gate_w1.y, x0, x1, x2, x3, x4, x5, x6, x7, gate_scale1,
          gate_acc1);

      uint2 up_w1 =
          *reinterpret_cast<const uint2*>(w + up_w_base1 + byte_offset);
      up_acc1 = fp4x16_bf16_dot_scaled_accum(
          up_w1.x, up_w1.y, x0, x1, x2, x3, x4, x5, x6, x7, up_scale1,
          up_acc1);
    }

  }

  gate_acc0 = warp_sum(gate_acc0);
  up_acc0 = warp_sum(up_acc0);
  gate_acc1 = warp_sum(gate_acc1);
  up_acc1 = warp_sum(up_acc1);

  if (lane == 0) {
    float a = alpha[expert];
    float half_gate0 = gate_acc0 * (0.5f * a);
    float up_scaled0 = up_acc0 * a;
    float silu0 = fmaf(half_gate0, tanh_approx_f32(half_gate0), half_gate0);
    float half_gate1 = gate_acc1 * (0.5f * a);
    float up_scaled1 = up_acc1 * a;
    float silu1 = fmaf(half_gate1, tanh_approx_f32(half_gate1), half_gate1);
    int64_t out_base = (static_cast<int64_t>(token) * kTopK + warp) * I;
    out[out_base + out_i0] = __float2bfloat16_rn(silu0 * up_scaled0);
    out[out_base + out_i1] = __float2bfloat16_rn(silu1 * up_scaled1);
  }
}

template <int B, int H, int I, int E, int kTopK>
void launch_moe_gate_up_nvfp4_swiglu_kernel(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor w_scale,
    torch::Tensor alpha,
    torch::Tensor expert_ids,
    torch::Tensor out) {
  dim3 grid(I / kColumnsPerWarp, B);
  dim3 block(kTopK * kWarpSize);
  auto stream = at::cuda::getCurrentCUDAStream();

  moe_gate_up_nvfp4_swiglu_kernel<B, H, I, E, kTopK>
      <<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      w.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(w_scale.data_ptr()),
      alpha.data_ptr<float>(),
      expert_ids.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
}

void moe_gate_up_nvfp4_swiglu_out(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor w_scale,
    torch::Tensor alpha,
    torch::Tensor expert_ids,
    torch::Tensor out) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(w_scale.is_cuda(), "w_scale must be a CUDA tensor");
  TORCH_CHECK(alpha.is_cuda(), "alpha must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
  TORCH_CHECK(w_scale.is_contiguous(), "w_scale must be contiguous");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(is_aligned(x.data_ptr(), 32), "x must be 32-byte aligned");
  TORCH_CHECK(is_aligned(w.data_ptr(), 16), "w must be 16-byte aligned");
  TORCH_CHECK(is_aligned(w_scale.data_ptr(), 2), "w_scale must be 2-byte aligned");

  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(w.scalar_type() == at::kByte, "w must be uint8");
  TORCH_CHECK(
      w_scale.scalar_type() == at::kFloat8_e4m3fn,
      "w_scale must be float8_e4m3fn");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "expert_ids must be int32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

  TORCH_CHECK(x.dim() == 2, "x must have shape [B, H]");
  TORCH_CHECK(w.dim() == 3, "w must have shape [E, 2I, H/2]");
  TORCH_CHECK(w_scale.dim() == 3, "w_scale must have shape [E, 2I, H/16]");
  TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must have shape [B, topk]");
  TORCH_CHECK(out.dim() == 3, "out must have shape [B, topk, I]");

  int B = static_cast<int>(x.size(0));
  int H = static_cast<int>(x.size(1));
  int E = static_cast<int>(w.size(0));
  int two_I = static_cast<int>(w.size(1));
  int I = two_I / 2;
  int topk = static_cast<int>(expert_ids.size(1));

  TORCH_CHECK(two_I % 2 == 0, "w.size(1) must be even");
  TORCH_CHECK(H % kGroupSize == 0, "H must be a multiple of 512");
  TORCH_CHECK(w.size(2) == H / 2, "w shape mismatch");
  TORCH_CHECK(w_scale.size(0) == E, "w_scale E mismatch");
  TORCH_CHECK(w_scale.size(1) == two_I, "w_scale 2I mismatch");
  TORCH_CHECK(w_scale.size(2) == H / kBlockSize, "w_scale H/16 mismatch");
  TORCH_CHECK(alpha.numel() == E, "alpha shape mismatch");
  TORCH_CHECK(expert_ids.size(0) == B, "expert_ids B mismatch");
  TORCH_CHECK(out.size(0) == B, "out B mismatch");
  TORCH_CHECK(out.size(1) == topk, "out topk mismatch");
  TORCH_CHECK(out.size(2) == I, "out I mismatch");
  TORCH_CHECK(topk > 0 && topk * kWarpSize <= 1024, "unsupported topk");

#define LAUNCH_FOR_H(BV, HV, IV, EV, TV)                                      \
  launch_moe_gate_up_nvfp4_swiglu_kernel<BV, HV, IV, EV, TV>(                 \
      x, w, w_scale, alpha, expert_ids, out)

#define DISPATCH_H(BV, IV, EV, TV)                                            \
  do {                                                                        \
    switch (H) {                                                              \
      case 512:                                                               \
        LAUNCH_FOR_H(BV, 512, IV, EV, TV);                                    \
        break;                                                                \
      case 1024:                                                              \
        LAUNCH_FOR_H(BV, 1024, IV, EV, TV);                                   \
        break;                                                                \
      case 2048:                                                              \
        LAUNCH_FOR_H(BV, 2048, IV, EV, TV);                                   \
        break;                                                                \
      case 3072:                                                              \
        LAUNCH_FOR_H(BV, 3072, IV, EV, TV);                                   \
        break;                                                                \
      case 4096:                                                              \
        LAUNCH_FOR_H(BV, 4096, IV, EV, TV);                                   \
        break;                                                                \
      default:                                                                \
        TORCH_CHECK(                                                          \
            false,                                                            \
            "unsupported H=",                                                 \
            H,                                                                \
            "; add a template specialization for this hidden size");          \
    }                                                                         \
  } while (false)

  if (B == 32 && I == 1536 && E == 256 && topk == 8) {
    DISPATCH_H(32, 1536, 256, 8);
  } else if (B == 4 && I == 128 && E == 16 && topk == 4) {
    DISPATCH_H(4, 128, 16, 4);
  } else if (B == 2 && I == 64 && E == 8 && topk == 2) {
    DISPATCH_H(2, 64, 8, 2);
  } else {
    TORCH_CHECK(
        false,
        "unsupported shape B=",
        B,
        " I=",
        I,
        " E=",
        E,
        " topk=",
        topk,
        "; add a full-shape template specialization");
  }

#undef DISPATCH_H
#undef LAUNCH_FOR_H
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor moe_gate_up_nvfp4_swiglu(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor w_scale,
    torch::Tensor alpha,
    torch::Tensor expert_ids) {
  int64_t B = x.size(0);
  int64_t I = w.size(1) / 2;
  int64_t topk = expert_ids.size(1);
  auto out = torch::empty({B, topk, I}, x.options());
  moe_gate_up_nvfp4_swiglu_out(x, w, w_scale, alpha, expert_ids, out);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_gate_up_nvfp4_swiglu", &moe_gate_up_nvfp4_swiglu);
  m.def("moe_gate_up_nvfp4_swiglu_out", &moe_gate_up_nvfp4_swiglu_out);
}
