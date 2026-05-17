import functools

import torch

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


@dsl_user_op
def fma_f32(
    a: cutlass.Float32,
    b: cutlass.Float32,
    c: cutlass.Float32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class MoEGateUpNvfp4SwigluKernel:
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
        mX: cute.Tensor,
        mW: cute.Tensor,
        mSFW: cute.Tensor,
        mAlpha: cute.Tensor,
        mExpertIds: cute.Tensor,
        mOut: cute.Tensor,
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
        ).launch(grid=grid, block=block, stream=stream)

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
        warp_expert_id = mExpertIds[(token_id, warp_id)]

        mW_expert_fp4 = cute.recast_tensor(
            mW[(None, None, None, warp_expert_id)], cutlass.Float4E2M1FN
        )
        gW = cute.local_tile(mW_expert_fp4, tiler_mn, (slot_id, 0, None))
        gSF = cute.local_tile(
            mSFW[(None, None, None, warp_expert_id)], tiler_mn_sf, (slot_id, 0, None)
        )

        copy_atom_w = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gW.element_type, num_bits_per_copy=64
        )
        thr_copy_w = cute.make_tiled_copy(copy_atom_w, layout_v, tiler_mn).get_slice(
            lane_id
        )
        tWgW = thr_copy_w.partition_S(gW)
        tWrW = cute.make_rmem_tensor((16,), dtype=gW.element_type)

        copy_atom_sf = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gSF.element_type, num_bits_per_copy=8
        )
        thr_copy_sf = cute.make_tiled_copy(
            copy_atom_sf, layout_sf, tiler_mn_sf
        ).get_slice(lane_id)
        tWSFgWSF = thr_copy_sf.partition_S(gSF)
        tWSFrWSF = cute.make_rmem_tensor((1,), dtype=mSFW.element_type)

        gX = cute.local_tile(mX, tiler_mn, (token_id, 0))
        copy_atom_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=256
        )
        thr_copy_x = cute.make_tiled_copy(copy_atom_x, layout_v, tiler_mn).get_slice(
            lane_id
        )
        tXgX = thr_copy_x.partition_S(gX)
        tXrX = cute.make_rmem_tensor((16,), dtype=gX.element_type)

        gate_acc = cutlass.Float32(0.0)
        up_acc = cutlass.Float32(0.0)
        alpha = mAlpha[warp_expert_id]

        for i in cutlass.range_constexpr(tWgW.shape[0][1]):
            cute.copy(copy_atom_x, tXgX[(None, i), 0, 0], tXrX)
            x = tXrX.load()

            cute.copy(copy_atom_w, tWgW[(None, i), 0, 0, 0], tWrW)
            cute.copy(copy_atom_sf, tWSFgWSF[(None, i), 0, 0, 0], tWSFrWSF)
            w = tWrW.load().to(cutlass.Float16).to(cutlass.BFloat16)
            w_sf = e4m3_scalar_to_f32(tWSFrWSF[0])
            partial = (w * x).to(cutlass.Float32).reduce(
                cute.ReductionOp.ADD, 0.0, reduction_profile=0
            )
            gate_acc = fma_f32(partial, w_sf, gate_acc)

            cute.copy(copy_atom_w, tWgW[(None, i), 0, 0, 1], tWrW)
            cute.copy(copy_atom_sf, tWSFgWSF[(None, i), 0, 0, 1], tWSFrWSF)
            w = tWrW.load().to(cutlass.Float16).to(cutlass.BFloat16)
            w_sf = e4m3_scalar_to_f32(tWSFrWSF[0])
            partial = (w * x).to(cutlass.Float32).reduce(
                cute.ReductionOp.ADD, 0.0, reduction_profile=0
            )
            up_acc = fma_f32(partial, w_sf, up_acc)

        gate_acc = cute.arch.warp_reduction_sum(gate_acc) * alpha
        up_acc = cute.arch.warp_reduction_sum(up_acc) * alpha

        if lane_id == 0:
            mOut[(token_id, warp_id, slot_id)] = swiglu_f32(gate_acc, up_acc).to(
                cutlass.BFloat16
            )


@functools.cache
def get_compiled_cutedsl_kernel(
    H: int,
    I: int,
    E: int,
    topk: int,
    sm_version: int = 100,
):
    k_group_size = cute.arch.WARP_SIZE * 16
    if H % k_group_size != 0:
        raise ValueError(f"H must be a multiple of {k_group_size}; got H={H}")

    kernel_obj = MoEGateUpNvfp4SwigluKernel(
        H=H, I=I, E=E, topk=topk, sm_version=sm_version
    )
    sym_b = cute.sym_int()

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_b, H), stride_order=(1, 0), assumed_align=256
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (I, H // 2, 2, E),
        stride_order=(1, 0, 2, 3),
        assumed_align=16,
    )
    w_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (I, H // 16, 2, E),
        stride_order=(1, 0, 2, 3),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (E,), assumed_align=4
    )
    expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_b, topk), stride_order=(1, 0), assumed_align=4
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, topk, I),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        w_scale_fake,
        alpha_fake,
        expert_ids_fake,
        out_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def moe_gate_up_nvfp4_swiglu_cutedsl_out(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
    sm_version: int = 100,
) -> torch.Tensor:
    B, H = x.shape
    E, two_I, H2 = w.shape
    I = two_I // 2
    topk = expert_ids.shape[1]
    k_group_size = cute.arch.WARP_SIZE * 16
    if H % k_group_size != 0:
        raise ValueError(f"H must be a multiple of {k_group_size}; got H={H}")
    if H2 != H // 2:
        raise ValueError(f"w.shape[2] must be H/2; got {H2} for H={H}")

    compiled_kernel = get_compiled_cutedsl_kernel(
        H=H, I=I, E=E, topk=topk, sm_version=sm_version
    )
    w_arg = w.view(E, 2, I, H // 2).permute(2, 3, 1, 0)
    w_scale_arg = w_scale.view(E, 2, I, H // 16).permute(2, 3, 1, 0)
    compiled_kernel(x, w_arg, w_scale_arg, alpha, expert_ids, out)
    return out


def moe_gate_up_nvfp4_swiglu_cutedsl(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    sm_version: int = 100,
) -> torch.Tensor:
    B, _ = x.shape
    _, two_I, _ = w.shape
    I = two_I // 2
    topk = expert_ids.shape[1]
    out = torch.empty((B, topk, I), dtype=torch.bfloat16, device=x.device)
    return moe_gate_up_nvfp4_swiglu_cutedsl_out(
        x, w, w_scale, alpha, expert_ids, out, sm_version=sm_version
    )


_moe_gate_up_nvfp4_swiglu_fusion_cute = moe_gate_up_nvfp4_swiglu_cutedsl
_moe_gate_up_nvfp4_swiglu_fusion_cute_out = moe_gate_up_nvfp4_swiglu_cutedsl_out
