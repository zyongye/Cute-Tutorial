import functools

import torch
import torch.nn.functional as F

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm


@dsl_user_op
def e4m3_scalar_to_f32(
    x: cutlass.Float8E4M3FN,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    """Convert one FP8 E4M3 scale to FP32 without requiring a 4-wide FP8 vector."""
    x_i8 = arith.bitcast(
        cutlass.Int8.mlir_type, x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    x_i16 = llvm.zext(cutlass.Int16.mlir_type, x_i8, loc=loc, ip=ip)
    f32 = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [x_i16],
        """{\n\t
            .reg .b32 h2;\n\t
            .reg .b16 h0;\n\t
            cvt.rn.f16x2.e4m3x2 h2, $1;\n\t
            mov.b32 {h0, _}, h2;\n\t
            cvt.f32.f16 $0, h0;\n\t
        }""",
        "=f,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(f32)


@dsl_user_op
def tanh_f32(
    x: cutlass.Float32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def silu_f32(
    x: cutlass.Float32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    x_half = cutlass.Float32(0.5) * x
    return x_half * tanh_f32(x_half, loc=loc, ip=ip) + x_half


@dsl_user_op
def swiglu_f32(
    gate: cutlass.Float32,
    up: cutlass.Float32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    return silu_f32(gate, loc=loc, ip=ip) * up


# -----------------------------------------------------------------------------
# NVFP4 fused kernel
# -----------------------------------------------------------------------------

class MoEGateUpNvfp4SwigluKernel:
    """
    CUTE kernel: MoE gate+up projection with bf16 activation and NVFP4 weights,
    fused with SwiGLU.

    Per token b and topk slot k with e = expert_ids[b, k]:
        gate_up   = alpha[e] * (x[b] @ dequant(w[e]))               # [2I]
        out[b, k] = silu(gate_up[0::2]) * gate_up[1::2]             # [I]  bf16

    H, I, E, topk are baked at compile time; only batch B is dynamic.
    """

    def __init__(
        self,
        H: int,
        I: int,
        E: int,
        topk: int,
        sm_version: int = 100,
    ):
        self.H = H
        self.I = I
        self.E = E
        self.topk = topk
        self.sm_version = sm_version

        self.warp_size = cute.arch.WARP_SIZE
        self.num_threads = self.topk * self.warp_size
        self.block_size = 16
        self.k_group_size = self.warp_size * self.block_size

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,          # [B, H]      bf16
        mW: cute.Tensor,          # [2I, H/2, E]  NVFP4 packed, K-major
        mSFW: cute.Tensor,        # [2I, H/16, E] FP8 E4M3 block scales
        mAlpha: cute.Tensor,      # [E]         fp32 per-expert global scale
        mExpertIds: cute.Tensor,  # [B, topk]   int32
        mOut: cute.Tensor,        # [B, topk, I]  bf16
        stream: cuda.CUstream,
    ):        
        B, _, I = mOut.shape
        H = mX.shape[1]
        k_groups = H // self.k_group_size

        grid = (I, B)
        block = (self.num_threads, 1, 1)
        tiler_mn = (1, H)
        tiler_mn_sf = (1, H // self.block_size)
        layout_v = cute.make_layout(
            shape=((self.warp_size, 1), (self.block_size, k_groups)),
            stride=((self.block_size, 0), (1, self.k_group_size)),
        )
        layout_sf = cute.make_layout(
            shape=((self.warp_size, 1), (1, k_groups)),
            stride=((1, 0), (1, self.warp_size)),
        )

        self.kernel(
            mX,
            mW,
            mSFW,
            mAlpha,
            mExpertIds,
            mOut,
            tiler_mn,
            tiler_mn_sf,
            layout_v,
            layout_sf,
        ).launch(
            grid=grid,
            block=block,
            stream=stream,
        )
    
    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mSFW: cute.Tensor,
        mAlpha: cute.Tensor,
        mExpertIds: cute.Tensor,
        mOut: cute.Tensor,
        tiler_mn: cute.Shape, 
        tiler_mn_sf: cute.Shape,
        layout_v: cute.Layout,
        layout_sf: cute.Layout,
    ):
        slot_id, token_id, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        warp_id = tidx // self.warp_size
        lane_id = tidx % self.warp_size

        print(f"[DSL INFO] mX={mX}")
        print(f"[DSL INFO] mW={mW}")

        warp_experts_id = mExpertIds[(token_id, warp_id)]

        mW_expert_fp4 = cute.recast_tensor(
            mW[(None, None, None, warp_experts_id)], cutlass.Float4E2M1FN
        )
        gW = cute.local_tile(mW_expert_fp4, tiler_mn, (slot_id, 0, None))
        gSF = cute.local_tile(mSFW[(None, None, None, warp_experts_id)], tiler_mn_sf, (slot_id, 0, None))

        print(f"[DSL INFO] gW={gW}")
        print(f"[DSL INFO] gSF={gSF}")

        copy_atom_w = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gW.element_type, num_bits_per_copy=64)
        thr_copy_W = cute.make_tiled_copy(copy_atom_w, layout_v, tiler_mn).get_slice(lane_id)

        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_rmem_tensor((16, ), dtype=gW.element_type)

        print(f"[DSL INFO] tWgW={tWgW}")
        print(f"[DSL INFO] tWrW={tWrW}")
        
        copy_atom_sf = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gSF.element_type, num_bits_per_copy=8)
        thr_copy_sf = cute.make_tiled_copy(copy_atom_sf, layout_sf, tiler_mn_sf).get_slice(lane_id)

        tWSFgWSF = thr_copy_sf.partition_S(gSF)
        tWSFrWSF = cute.make_rmem_tensor((1, ), dtype=mSFW.element_type)

        print(f"[DSL INFO] tWSFgWSF={tWSFgWSF}")
        print(f"[DSL INFO] tSFrSF={tWSFrWSF}")


        gX = cute.local_tile(mX, tiler_mn, (token_id, 0))
        copy_atom_x = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=256)
        thr_copy_X = cute.make_tiled_copy(copy_atom_x, layout_v, tiler_mn).get_slice(lane_id)
        tXgX = thr_copy_X.partition_S(gX)
        tXrX = cute.make_rmem_tensor((16,), dtype=gX.element_type)

        print(f"[DSL INFO] gX={gX}")
        print(f"[DSL INFO] tXgX={tXgX}")

        gate_acc = cutlass.Float32(0.0)
        up_acc = cutlass.Float32(0.0)
        alpha = mAlpha[warp_experts_id]

        # calculate gate first
        for i in range(tWgW.shape[0][1]):
            cute.copy(copy_atom_w, tWgW[(None, i), 0, 0, 0], tWrW)
            cute.copy(copy_atom_x, tXgX[(None, i), 0, 0], tXrX)
            cute.copy(copy_atom_sf, tWSFgWSF[(None, i), 0, 0, 0], tWSFrWSF)

            w = tWrW.load().to(cutlass.Float16).to(cutlass.BFloat16)
            x = tXrX.load()
            w_sf = e4m3_scalar_to_f32(tWSFrWSF[0]) * alpha
            sum = (w * x).to(cutlass.Float32).reduce(
                cute.ReductionOp.ADD, 0.0, reduction_profile=0
            ) * w_sf

            gate_acc = gate_acc + sum
        
        for i in range(tWgW.shape[0][1]):
            cute.copy(copy_atom_w, tWgW[(None, i), 0, 0, 1], tWrW)
            cute.copy(copy_atom_x, tXgX[(None, i), 0, 0], tXrX)
            cute.copy(copy_atom_sf, tWSFgWSF[(None, i), 0, 0, 1], tWSFrWSF)

            w = tWrW.load().to(cutlass.Float16).to(cutlass.BFloat16)
            x = tXrX.load()
            w_sf = e4m3_scalar_to_f32(tWSFrWSF[0]) * alpha
            sum = (w * x).to(cutlass.Float32).reduce(
                cute.ReductionOp.ADD, 0.0, reduction_profile=0
            ) * w_sf

            up_acc = up_acc + sum
        
        gate_acc = cute.arch.warp_reduction_sum(gate_acc)
        up_acc = cute.arch.warp_reduction_sum(up_acc)
        swiglu_acc = swiglu_f32(gate_acc, up_acc)

        if lane_id == 0:
            mOut[(token_id, warp_id, slot_id)] = swiglu_acc.to(cutlass.BFloat16)
        

@functools.cache
def _get_compiled_kernel(
    H: int,
    I: int,
    E: int,
    topk: int,
    sm_version: int = 100,
):
    """Compile MoEGateUpNvfp4SwigluKernel. Only batch size is dynamic."""
    k_group_size = cute.arch.WARP_SIZE * 16
    if H % k_group_size != 0:
        raise ValueError(f"H must be a multiple of {k_group_size}; got H={H}")

    kernel_obj = MoEGateUpNvfp4SwigluKernel(
        H=H, I=I, E=E, topk=topk, sm_version=sm_version,
    )

    sym_b = cute.sym_int()

    # bf16 activation; NVFP4 weights with per-16-element FP8 E4M3 block scales.
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_b, H), stride_order=(1, 0), assumed_align=256,
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (I, H // 2, 2, E), stride_order=(1, 0, 2, 3), assumed_align=16,
    )
    w_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (I, H // 16, 2, E), stride_order=(1, 0, 2, 3), assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (E,), assumed_align=4,
    )
    expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_b, topk), stride_order=(1, 0), assumed_align=4,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_b, topk, I), stride_order=(2, 1, 0), assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake, w_scale_fake,
        alpha_fake,
        expert_ids_fake,
        out_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_kernel


def _moe_gate_up_nvfp4_swiglu_fusion_cute(
    x: torch.Tensor,           # [B, H]        bf16
    w: torch.Tensor,           # [E, 2I, H/2]  uint8, NVFP4 packed (K-major)
    w_scale: torch.Tensor,     # [E, 2I, H/16] FP8 E4M3
    alpha: torch.Tensor,       # [E]           fp32 per-expert global scale
    expert_ids: torch.Tensor,  # [B, topk]     int32
    sm_version: int = 100,
) -> torch.Tensor:
    """Allocate the bf16 output and dispatch the compiled fused kernel."""
    B, H = x.shape
    E, two_I, _ = w.shape
    I = two_I // 2
    topk = expert_ids.shape[1]
    k_group_size = cute.arch.WARP_SIZE * 16
    if H % k_group_size != 0:
        raise ValueError(f"H must be a multiple of {k_group_size}; got H={H}")

    w = w.view(E, 2, I, H // 2)
    w_scale = w_scale.view(E, 2, I, H // 16)

    out = torch.empty((B, topk, I), dtype=torch.bfloat16, device=x.device)

    compiled_kernel = _get_compiled_kernel(
        H=H, I=I, E=E, topk=topk, sm_version=sm_version,
    )
    compiled_kernel(x, w.permute(2, 3, 1, 0), w_scale.permute(2, 3, 1, 0), alpha, expert_ids, out)

    return out


# -----------------------------------------------------------------------------
# bf16 references
# -----------------------------------------------------------------------------

def moe_gate_up_ref(
    x: torch.Tensor,           # [B, H] bf16
    w: torch.Tensor,           # [E, H, 2I] bf16
    alpha: torch.Tensor,       # [E] fp32 per-expert global scale
    expert_ids: torch.Tensor,  # [B, topk] int32
) -> torch.Tensor:
    """
    bf16 reference for MoE gate+up projection (pre-SwiGLU).

    Returns [B, topk, 2I] with gate/up interleaved along the last dim.
    """
    e = expert_ids.long()
    w_sel = w[e]                              # [B, topk, H, 2I]
    alpha_sel = alpha[e].unsqueeze(-1)        # [B, topk, 1]
    out = torch.einsum("bh,bkhi->bki", x.float(), w_sel.float())
    out = out * alpha_sel
    return out.to(torch.bfloat16)


def moe_gate_up_swiglu_ref(
    x: torch.Tensor,           # [B, H] bf16
    w: torch.Tensor,           # [E, H, 2I] bf16
    alpha: torch.Tensor,       # [E] fp32 per-expert global scale
    expert_ids: torch.Tensor,  # [B, topk] int32
) -> torch.Tensor:
    """
    bf16 reference: gate+up projection -> SwiGLU.

    Gate/up are interleaved along the 2I dim: [g0, u0, g1, u1, ...].
    Returns [B, topk, I] bf16: silu(gate) * up, where gate_up is scaled by alpha[e].
    """
    e = expert_ids.long()
    w_sel = w[e]                              # [B, topk, H, 2I]
    alpha_sel = alpha[e].unsqueeze(-1)        # [B, topk, 1]
    gate_up = torch.einsum("bh,bkhi->bki", x.float(), w_sel.float())  # [B, topk, 2I]
    gate_up = gate_up * alpha_sel
    out = F.silu(gate_up[..., 0::2]) * gate_up[..., 1::2]
    return out.to(torch.bfloat16)


def _nvfp4_e2m1_to_bf16(w: torch.Tensor) -> torch.Tensor:
    """Dequantize packed unsigned nibbles interpreted as FP4 E2M1FN."""
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=w.device,
    )
    lo = (w & 0x0F).long()
    hi = ((w >> 4) & 0x0F).long()
    lo_val = table[lo & 0x7] * torch.where((lo & 0x8) != 0, -1.0, 1.0)
    hi_val = table[hi & 0x7] * torch.where((hi & 0x8) != 0, -1.0, 1.0)

    out = torch.empty(
        *w.shape[:-1],
        w.shape[-1] * 2,
        dtype=torch.float32,
        device=w.device,
    )
    out[..., 0::2] = lo_val
    out[..., 1::2] = hi_val
    return out.to(torch.bfloat16)


def moe_gate_up_nvfp4_swiglu_ref(
    x: torch.Tensor,           # [B, H] bf16
    w: torch.Tensor,           # [E, 2I, H/2] uint8
    w_scale: torch.Tensor,     # [E, 2I, H/16] fp8 e4m3
    alpha: torch.Tensor,       # [E] fp32
    expert_ids: torch.Tensor,  # [B, topk] int32
) -> torch.Tensor:
    """Reference matching the kernel's FP4->bf16, bf16 product, fp32 reduction path."""
    B, H = x.shape
    E, two_I, H2 = w.shape
    I = two_I // 2
    topk = expert_ids.shape[1]
    block_size = 16
    num_blocks = H // block_size
    chunk_i = 64

    w = w.view(E, 2, I, H2)
    w_scale = w_scale.view(E, 2, I, num_blocks).float()
    x_blocks = x.view(B, num_blocks, block_size)

    out = torch.empty((B, topk, I), dtype=torch.bfloat16, device=x.device)
    for i0 in range(0, I, chunk_i):
        i1 = min(i0 + chunk_i, I)
        gate_up_chunks = []
        for part in range(2):
            part_out = torch.empty((B, topk, i1 - i0), dtype=torch.float32, device=x.device)
            for b in range(B):
                for k in range(topk):
                    e = int(expert_ids[b, k].item())
                    w_chunk = _nvfp4_e2m1_to_bf16(w[e, part, i0:i1]).view(
                        i1 - i0, num_blocks, block_size
                    )
                    partial = (x_blocks[b].unsqueeze(0) * w_chunk).float().sum(dim=-1)
                    scaled = partial * w_scale[e, part, i0:i1]
                    part_out[b, k] = scaled.sum(dim=-1) * alpha[e]
            gate_up_chunks.append(part_out)

        gate, up = gate_up_chunks
        silu = (0.5 * gate) * torch.tanh(0.5 * gate) + (0.5 * gate)
        out[..., i0:i1] = (silu * up).to(torch.bfloat16)
    return out


def _run_correctness_case(
    H: int,
    B: int,
    topk: int,
    I: int,
    E: int,
    device: str,
) -> None:
    # bf16 activation; NVFP4 packed weights with FP8 E4M3 per-block scales.
    x          = torch.randn(B, H,                         dtype=torch.bfloat16, device=device)
    w          = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w_scale    = torch.randn(E, 2 * I, H // 16,            device=device).to(torch.float8_e4m3fn)
    alpha      = torch.rand(E,                             dtype=torch.float32, device=device)
    expert_ids = torch.randint(0, E, (B, topk),            dtype=torch.int32, device=device)

    out = _moe_gate_up_nvfp4_swiglu_fusion_cute(
        x, w, w_scale, alpha, expert_ids,
    )
    ref = moe_gate_up_nvfp4_swiglu_ref(x, w, w_scale, alpha, expert_ids)
    torch.cuda.synchronize()

    diff = (out.float() - ref.float()).abs()
    print(
        f"H={H}: out={tuple(out.shape)} max_abs={diff.max().item():.6g} "
        f"mean_abs={diff.mean().item():.6g}"
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    del x, w, w_scale, alpha, expert_ids, out, ref, diff
    torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    device = "cuda"

    B, topk = 32, 8
    I = 1536
    E = 256
    hidden_sizes = (512, 1024, 2048, 3072, 4096)

    for H in hidden_sizes:
        _run_correctness_case(H, B, topk, I, E, device)
    print("PASS")


if __name__ == "__main__":
    main()
