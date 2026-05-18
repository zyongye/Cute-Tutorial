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
constexpr int kMaxDownWarpsPerCta = 8;
constexpr int kTargetDownCtasPerSm = 8;

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

template <int I, int kTopK>
__global__ __launch_bounds__(
    kMaxDownWarpsPerCta * kWarpSize,
    kTargetDownCtasPerSm)
void moe_down_combine_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ w_scale,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    __nv_bfloat16* __restrict__ out,
    int B,
    int H,
    int E) {
  static_assert(I % kGroupSize == 0, "I must be a multiple of 512");
  static_assert(kTopK <= kMaxDownWarpsPerCta, "topk exceeds CTA warp budget");
  constexpr int kGroups = I / kGroupSize;

  __shared__ float partials[2 * kMaxDownWarpsPerCta];

  int out_h0 = blockIdx.x * kColumnsPerWarp;
  int out_h1 = out_h0 + 1;
  int token = blockIdx.y;
  int lane = threadIdx.x & (kWarpSize - 1);
  int route = threadIdx.x / kWarpSize;

  if (token >= B || out_h1 >= H) {
    return;
  }

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  int expert = expert_ids[token * kTopK + route];
  if (expert >= 0 && expert < E) {
    float route_weight = route_weights[token * kTopK + route];
    const __nv_bfloat16* x_route =
        x + (static_cast<int64_t>(token) * kTopK + route) * I;
    int64_t w_base0 = (static_cast<int64_t>(expert) * H + out_h0) * (I / 2);
    int64_t w_base1 = w_base0 + (I / 2);
    int64_t scale_base0 =
        (static_cast<int64_t>(expert) * H + out_h0) * (I / kBlockSize);
    int64_t scale_base1 = scale_base0 + (I / kBlockSize);

#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
      int elem_offset = group * kGroupSize + lane * kBlockSize;
      U64x4 x_vec = ld_global_u64x4(x_route + elem_offset);
      uint32_t x0 = static_cast<uint32_t>(x_vec.x);
      uint32_t x1 = static_cast<uint32_t>(x_vec.x >> 32);
      uint32_t x2 = static_cast<uint32_t>(x_vec.y);
      uint32_t x3 = static_cast<uint32_t>(x_vec.y >> 32);
      uint32_t x4 = static_cast<uint32_t>(x_vec.z);
      uint32_t x5 = static_cast<uint32_t>(x_vec.z >> 32);
      uint32_t x6 = static_cast<uint32_t>(x_vec.w);
      uint32_t x7 = static_cast<uint32_t>(x_vec.w >> 32);

      int scale_offset = group * kWarpSize + lane;
      float scale0 =
          e4m3_scalar_to_f32(w_scale[scale_base0 + scale_offset]) *
          route_weight;
      float scale1 =
          e4m3_scalar_to_f32(w_scale[scale_base1 + scale_offset]) *
          route_weight;

      int byte_offset = group * (kGroupSize / 2) + lane * (kBlockSize / 2);
      uint2 w0 = *reinterpret_cast<const uint2*>(w + w_base0 + byte_offset);
      uint2 w1 = *reinterpret_cast<const uint2*>(w + w_base1 + byte_offset);
      acc0 = fp4x16_bf16_dot_scaled_accum(
          w0.x, w0.y, x0, x1, x2, x3, x4, x5, x6, x7, scale0, acc0);
      acc1 = fp4x16_bf16_dot_scaled_accum(
          w1.x, w1.y, x0, x1, x2, x3, x4, x5, x6, x7, scale1, acc1);
    }
  }

  acc0 = warp_sum(acc0);
  acc1 = warp_sum(acc1);

  if (lane == 0) {
    partials[route] = acc0;
    partials[kMaxDownWarpsPerCta + route] = acc1;
  }
  __syncthreads();

  if (route == 0) {
    float combined0 = lane < kTopK ? partials[lane] : 0.0f;
    float combined1 = lane < kTopK ? partials[kMaxDownWarpsPerCta + lane] : 0.0f;
    combined0 = warp_sum(combined0);
    combined1 = warp_sum(combined1);

    if (lane == 0) {
      int64_t out_base = static_cast<int64_t>(token) * H;
      out[out_base + out_h0] = __float2bfloat16_rn(combined0);
      out[out_base + out_h1] = __float2bfloat16_rn(combined1);
    }
  }
}

template <int I, int kTopK>
void launch_moe_down_combine_nvfp4_kernel(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor w_scale,
    torch::Tensor expert_ids,
    torch::Tensor route_weights,
    torch::Tensor out,
    int B,
    int H,
    int E) {
  dim3 grid(H / kColumnsPerWarp, B);
  dim3 block(kTopK * kWarpSize);
  auto stream = at::cuda::getCurrentCUDAStream();

  moe_down_combine_nvfp4_kernel<I, kTopK><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      w.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(w_scale.data_ptr()),
      expert_ids.data_ptr<int32_t>(),
      route_weights.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      B,
      H,
      E);
}

void moe_down_combine_nvfp4_out(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor w_scale,
    torch::Tensor expert_ids,
    torch::Tensor route_weights,
    torch::Tensor out) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(w_scale.is_cuda(), "w_scale must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(route_weights.is_cuda(), "route_weights must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
  TORCH_CHECK(w_scale.is_contiguous(), "w_scale must be contiguous");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");
  TORCH_CHECK(route_weights.is_contiguous(), "route_weights must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(is_aligned(x.data_ptr(), 32), "x must be 32-byte aligned");
  TORCH_CHECK(is_aligned(w.data_ptr(), 8), "w must be 8-byte aligned");
  TORCH_CHECK(is_aligned(w_scale.data_ptr(), 1), "w_scale must be aligned");

  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(w.scalar_type() == at::kByte, "w must be uint8");
  TORCH_CHECK(
      w_scale.scalar_type() == at::kFloat8_e4m3fn,
      "w_scale must be float8_e4m3fn");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "expert_ids must be int32");
  TORCH_CHECK(route_weights.scalar_type() == at::kFloat, "route_weights must be float32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

  TORCH_CHECK(x.dim() == 3, "x must have shape [B, topk, I]");
  TORCH_CHECK(w.dim() == 3, "w must have shape [E, H, I/2]");
  TORCH_CHECK(w_scale.dim() == 3, "w_scale must have shape [E, H, I/16]");
  TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must have shape [B, topk]");
  TORCH_CHECK(route_weights.dim() == 2, "route_weights must have shape [B, topk]");
  TORCH_CHECK(out.dim() == 2, "out must have shape [B, H]");

  int B = static_cast<int>(x.size(0));
  int topk = static_cast<int>(x.size(1));
  int I = static_cast<int>(x.size(2));
  int E = static_cast<int>(w.size(0));
  int H = static_cast<int>(w.size(1));

  TORCH_CHECK(B > 0, "B must be positive");
  TORCH_CHECK(H > 0 && H % kColumnsPerWarp == 0, "H must be a positive multiple of 2");
  TORCH_CHECK(topk > 0 && topk <= kMaxDownWarpsPerCta, "topk must be in [1, 8]");
  TORCH_CHECK(I % kGroupSize == 0, "I must be a multiple of 512");
  TORCH_CHECK(w.size(2) == I / 2, "w I/2 mismatch");
  TORCH_CHECK(w_scale.size(0) == E, "w_scale E mismatch");
  TORCH_CHECK(w_scale.size(1) == H, "w_scale H mismatch");
  TORCH_CHECK(w_scale.size(2) == I / kBlockSize, "w_scale I/16 mismatch");
  TORCH_CHECK(expert_ids.size(0) == B, "expert_ids B mismatch");
  TORCH_CHECK(expert_ids.size(1) == topk, "expert_ids topk mismatch");
  TORCH_CHECK(route_weights.size(0) == B, "route_weights B mismatch");
  TORCH_CHECK(route_weights.size(1) == topk, "route_weights topk mismatch");
  TORCH_CHECK(out.size(0) == B, "out B mismatch");
  TORCH_CHECK(out.size(1) == H, "out H mismatch");

#define LAUNCH_FOR_I_TK(IV, TV)                                               \
  launch_moe_down_combine_nvfp4_kernel<IV, TV>(                               \
      x, w, w_scale, expert_ids, route_weights, out, B, H, E)

  if (I == 1536 && topk == 8) {
    LAUNCH_FOR_I_TK(1536, 8);
  } else if (I == 1024 && topk == 8) {
    LAUNCH_FOR_I_TK(1024, 8);
  } else if (I == 512 && topk == 8) {
    LAUNCH_FOR_I_TK(512, 8);
  } else if (I == 512 && topk == 4) {
    LAUNCH_FOR_I_TK(512, 4);
  } else if (I == 512 && topk == 2) {
    LAUNCH_FOR_I_TK(512, 2);
  } else {
    TORCH_CHECK(
        false,
        "unsupported shape I=",
        I,
        " topk=",
        topk,
        "; add an I/topk template specialization");
  }

#undef LAUNCH_FOR_I_TK
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor moe_down_combine_nvfp4(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor w_scale,
    torch::Tensor expert_ids,
    torch::Tensor route_weights) {
  int64_t B = x.size(0);
  int64_t H = w.size(1);
  auto out = torch::empty({B, H}, x.options());
  moe_down_combine_nvfp4_out(x, w, w_scale, expert_ids, route_weights, out);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_down_combine_nvfp4", &moe_down_combine_nvfp4);
  m.def("moe_down_combine_nvfp4_out", &moe_down_combine_nvfp4_out);
}
