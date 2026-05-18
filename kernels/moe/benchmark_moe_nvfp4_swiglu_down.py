import argparse
from dataclasses import dataclass
import sys

_SCRIPT_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import torch
import triton.testing

from moe_down_combine_nvfp4_cuda_interface import (
    load_cuda_extension as load_down_extension,
)
from moe_gate_up_nvfp4_swiglu_cuda_interface import (
    load_cuda_extension as load_gate_up_extension,
)


@dataclass
class PipelineCase:
    x: torch.Tensor
    gate_up_w: torch.Tensor
    gate_up_scale: torch.Tensor
    down_w: torch.Tensor
    down_scale: torch.Tensor
    alpha: torch.Tensor
    expert_ids: torch.Tensor
    route_weights: torch.Tensor
    intermediate: torch.Tensor
    out: torch.Tensor
    gate_up_ext: object
    down_ext: object


def peak_memory_bandwidth_gbps(device: str) -> float | None:
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz == 0 or memory_bus_width_bits == 0:
        return None
    return 2.0 * memory_clock_khz * 1000.0 * memory_bus_width_bits / 8.0 / 1.0e9


def make_case(
    H: int,
    B: int,
    topk: int,
    I: int,
    E: int,
    device: str,
) -> PipelineCase:
    x = torch.randn(B, H, dtype=torch.bfloat16, device=device)
    gate_up_w = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device
    )
    gate_up_scale = torch.randn(E, 2 * I, H // 16, device=device).to(
        torch.float8_e4m3fn
    )
    down_w = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device)
    down_scale = torch.randn(E, H, I // 16, device=device).to(torch.float8_e4m3fn)
    alpha = torch.rand(E, dtype=torch.float32, device=device)
    expert_ids = torch.randint(0, E, (B, topk), dtype=torch.int32, device=device)
    route_weights = torch.rand(B, topk, dtype=torch.float32, device=device)
    intermediate = torch.empty((B, topk, I), dtype=torch.bfloat16, device=device)
    out = torch.empty((B, H), dtype=torch.bfloat16, device=device)
    return PipelineCase(
        x=x,
        gate_up_w=gate_up_w,
        gate_up_scale=gate_up_scale,
        down_w=down_w,
        down_scale=down_scale,
        alpha=alpha,
        expert_ids=expert_ids,
        route_weights=route_weights,
        intermediate=intermediate,
        out=out,
        gate_up_ext=load_gate_up_extension(),
        down_ext=load_down_extension(),
    )


def _nvfp4_e2m1_to_bf16(w: torch.Tensor) -> torch.Tensor:
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


def gate_up_swiglu_swizzled_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    chunk_i: int = 64,
) -> torch.Tensor:
    B, H = x.shape
    E, two_I, H2 = w.shape
    I = two_I // 2
    topk = expert_ids.shape[1]
    block_size = 16
    num_blocks = H // block_size

    if H2 != H // 2:
        raise ValueError("gate/up w must have shape [E, 2I, H/2]")

    w_packed = w.view(E, I, H).view(E, I, num_blocks, block_size)
    scale_packed = w_scale.view(E, I, H // 8).view(E, I, num_blocks, 2).float()
    x_blocks = x.view(B, num_blocks, block_size)
    out = torch.empty((B, topk, I), dtype=torch.bfloat16, device=x.device)

    for i0 in range(0, I, chunk_i):
        i1 = min(i0 + chunk_i, I)
        cols = i1 - i0
        gate = torch.empty((B, topk, cols), dtype=torch.float32, device=x.device)
        up = torch.empty((B, topk, cols), dtype=torch.float32, device=x.device)

        for b in range(B):
            xb = x_blocks[b].unsqueeze(0).float()
            for route in range(topk):
                expert = int(expert_ids[b, route].item())
                if expert < 0 or expert >= E:
                    gate[b, route].zero_()
                    up[b, route].zero_()
                    continue

                packed = w_packed[expert, i0:i1]
                gate_w = _nvfp4_e2m1_to_bf16(packed[..., :8]).view(
                    cols, num_blocks, block_size
                )
                up_w = _nvfp4_e2m1_to_bf16(packed[..., 8:]).view(
                    cols, num_blocks, block_size
                )
                gate_partial = (xb * gate_w.float()).sum(dim=-1)
                up_partial = (xb * up_w.float()).sum(dim=-1)
                gate[b, route] = (
                    gate_partial * scale_packed[expert, i0:i1, :, 0]
                ).sum(dim=-1) * alpha[expert]
                up[b, route] = (
                    up_partial * scale_packed[expert, i0:i1, :, 1]
                ).sum(dim=-1) * alpha[expert]

        half_gate = 0.5 * gate
        silu = half_gate * torch.tanh(half_gate) + half_gate
        out[..., i0:i1] = (silu * up).to(torch.bfloat16)

    return out


def down_combine_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    chunk_h: int = 64,
) -> torch.Tensor:
    B, topk, I = x.shape
    E, H, I2 = w.shape
    block_size = 16
    num_blocks = I // block_size

    if I2 != I // 2:
        raise ValueError("down w must have shape [E, H, I/2]")

    x_blocks = x.view(B, topk, num_blocks, block_size)
    w_scale = w_scale.float()
    out = torch.empty((B, H), dtype=torch.float32, device=x.device)

    for h0 in range(0, H, chunk_h):
        h1 = min(h0 + chunk_h, H)
        out_chunk = torch.zeros((B, h1 - h0), dtype=torch.float32, device=x.device)
        for b in range(B):
            for route in range(topk):
                expert = int(expert_ids[b, route].item())
                if expert < 0 or expert >= E:
                    continue

                w_chunk = _nvfp4_e2m1_to_bf16(w[expert, h0:h1]).view(
                    h1 - h0, num_blocks, block_size
                )
                partial = (
                    x_blocks[b, route].unsqueeze(0).float() * w_chunk.float()
                ).sum(dim=-1)
                out_chunk[b] += (
                    partial * w_scale[expert, h0:h1]
                ).sum(dim=-1) * route_weights[b, route]
        out[:, h0:h1] = out_chunk

    return out.to(torch.bfloat16)


def slice_case(case: PipelineCase, B: int) -> PipelineCase:
    return PipelineCase(
        x=case.x[:B],
        gate_up_w=case.gate_up_w,
        gate_up_scale=case.gate_up_scale,
        down_w=case.down_w,
        down_scale=case.down_scale,
        alpha=case.alpha,
        expert_ids=case.expert_ids[:B],
        route_weights=case.route_weights[:B],
        intermediate=case.intermediate[:B],
        out=case.out[:B],
        gate_up_ext=case.gate_up_ext,
        down_ext=case.down_ext,
    )


def launch_gate_up(case: PipelineCase) -> None:
    case.gate_up_ext.moe_gate_up_nvfp4_swiglu_out(
        case.x,
        case.gate_up_w,
        case.gate_up_scale,
        case.alpha,
        case.expert_ids,
        case.intermediate,
    )


def launch_down(case: PipelineCase) -> None:
    case.down_ext.moe_down_combine_nvfp4_out(
        case.intermediate,
        case.down_w,
        case.down_scale,
        case.expert_ids,
        case.route_weights,
        case.out,
    )


def launch_full(case: PipelineCase) -> None:
    launch_gate_up(case)
    launch_down(case)


def gate_up_min_hbm_bytes(B: int, H: int, I: int, topk: int) -> float:
    output_elems = B * topk * I
    weight_scale_meta_out_bytes = output_elems * (H + H / 8.0 + 4 + 4 + 2)
    activation_once_bytes = B * H * 2
    return weight_scale_meta_out_bytes + activation_once_bytes


def down_min_hbm_bytes(B: int, H: int, I: int, topk: int) -> float:
    output_elems = B * H
    unique_activation_bytes = B * topk * I * 2
    weight_scale_bytes = output_elems * topk * (I / 2.0 + I / 16.0)
    meta_output_bytes = B * topk * (4 + 4) + output_elems * 2
    return unique_activation_bytes + weight_scale_bytes + meta_output_bytes


def fmt_peak(tbps: float, peak_memory_gbps: float | None) -> str:
    if peak_memory_gbps is None:
        return f"{tbps:.2f}"
    return f"{tbps:.2f} ({tbps * 1000.0 / peak_memory_gbps * 100.0:.1f}%)"


def benchmark_case(
    case: PipelineCase,
    B: int,
    H: int,
    I: int,
    topk: int,
    rep: int,
    peak_memory_gbps: float | None,
) -> dict[str, float]:
    launch_gate_up(case)
    torch.cuda.synchronize()

    gate_up_ms = triton.testing.do_bench_cudagraph(
        lambda: launch_gate_up(case), rep=rep
    )
    down_ms = triton.testing.do_bench_cudagraph(lambda: launch_down(case), rep=rep)
    full_ms = triton.testing.do_bench_cudagraph(lambda: launch_full(case), rep=rep)

    gate_up_flops = 4.0 * B * topk * I * H
    down_flops = 2.0 * B * topk * I * H
    full_flops = gate_up_flops + down_flops
    gate_up_hbm = gate_up_min_hbm_bytes(B, H, I, topk)
    down_hbm = down_min_hbm_bytes(B, H, I, topk)
    full_hbm = gate_up_hbm + down_hbm

    return {
        "B": float(B),
        "gate_up_ms": gate_up_ms,
        "down_ms": down_ms,
        "sum_ms": gate_up_ms + down_ms,
        "full_ms": full_ms,
        "gate_up_tflops": gate_up_flops / (gate_up_ms * 1.0e-3) / 1.0e12,
        "down_tflops": down_flops / (down_ms * 1.0e-3) / 1.0e12,
        "full_tflops": full_flops / (full_ms * 1.0e-3) / 1.0e12,
        "gate_up_hbm_tbps": gate_up_hbm / (gate_up_ms * 1.0e-3) / 1.0e12,
        "down_hbm_tbps": down_hbm / (down_ms * 1.0e-3) / 1.0e12,
        "full_hbm_tbps": full_hbm / (full_ms * 1.0e-3) / 1.0e12,
        "peak_memory_gbps": 0.0 if peak_memory_gbps is None else peak_memory_gbps,
    }


def run_correctness(args: argparse.Namespace) -> None:
    case = make_case(args.h, args.b, args.topk, args.i, args.e, args.device)
    launch_full(case)
    torch.cuda.synchronize()

    ref_intermediate = gate_up_swiglu_swizzled_ref(
        case.x,
        case.gate_up_w,
        case.gate_up_scale,
        case.alpha,
        case.expert_ids,
    )
    ref_out_from_cuda_intermediate = down_combine_ref(
        case.intermediate,
        case.down_w,
        case.down_scale,
        case.expert_ids,
        case.route_weights,
    )
    ref_out_from_torch_intermediate = down_combine_ref(
        ref_intermediate,
        case.down_w,
        case.down_scale,
        case.expert_ids,
        case.route_weights,
    )
    torch.cuda.synchronize()

    inter_diff = (case.intermediate.float() - ref_intermediate.float()).abs()
    out_diff = (case.out.float() - ref_out_from_cuda_intermediate.float()).abs()
    e2e_diff = (case.out.float() - ref_out_from_torch_intermediate.float()).abs()
    inter_close = torch.isclose(
        case.intermediate.float(),
        ref_intermediate.float(),
        rtol=args.rtol,
        atol=args.atol,
    )
    out_close = torch.isclose(
        case.out.float(),
        ref_out_from_cuda_intermediate.float(),
        rtol=args.rtol,
        atol=args.atol,
    )
    e2e_close = torch.isclose(
        case.out.float(),
        ref_out_from_torch_intermediate.float(),
        rtol=args.rtol,
        atol=args.atol,
    )
    print(
        f"Gate/up: out={tuple(case.intermediate.shape)} "
        f"max_abs={inter_diff.max().item():.6g} "
        f"mean_abs={inter_diff.mean().item():.6g} "
        f"bad={(~inter_close).sum().item()}/{inter_diff.numel()}"
    )
    print(
        f"Down+combine: out={tuple(case.out.shape)} "
        f"max_abs={out_diff.max().item():.6g} "
        f"mean_abs={out_diff.mean().item():.6g} "
        f"bad={(~out_close).sum().item()}/{out_diff.numel()}"
    )
    print(
        f"Pure torch e2e diagnostic: out={tuple(case.out.shape)} "
        f"max_abs={e2e_diff.max().item():.6g} "
        f"mean_abs={e2e_diff.mean().item():.6g} "
        f"bad={(~e2e_close).sum().item()}/{e2e_diff.numel()}"
    )
    torch.testing.assert_close(
        case.intermediate, ref_intermediate, rtol=args.rtol, atol=args.atol
    )
    torch.testing.assert_close(
        case.out,
        ref_out_from_cuda_intermediate,
        rtol=args.rtol,
        atol=args.atol,
    )
    print("PASS")


def print_header(args: argparse.Namespace, peak_memory_gbps: float | None) -> None:
    peak_msg = (
        "unknown"
        if peak_memory_gbps is None
        else f"{peak_memory_gbps / 1000.0:.2f} TB/s"
    )
    print(
        f"Pipeline benchmark: H={args.h} I={args.i} E={args.e} "
        f"topk={args.topk} rep={args.rep} peak_mem={peak_msg}"
    )
    print(
        "B,gate_up_ms,down_ms,sum_ms,full_ms,full_TFLOP/s,"
        "gate_up_min_hbm_TB/s,down_min_hbm_TB/s,full_min_hbm_TB/s"
    )


def print_result(row: dict[str, float], peak_memory_gbps: float | None) -> None:
    print(
        f"{int(row['B'])},"
        f"{row['gate_up_ms']:.6f},"
        f"{row['down_ms']:.6f},"
        f"{row['sum_ms']:.6f},"
        f"{row['full_ms']:.6f},"
        f"{row['full_tflops']:.2f},"
        f"{fmt_peak(row['gate_up_hbm_tbps'], peak_memory_gbps)},"
        f"{fmt_peak(row['down_hbm_tbps'], peak_memory_gbps)},"
        f"{fmt_peak(row['full_hbm_tbps'], peak_memory_gbps)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "correctness"), default="benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--min-b", type=int, default=1)
    parser.add_argument("--max-b", type=int, default=32)
    parser.add_argument("--sweep-bs", action="store_true")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--i", type=int, default=1536)
    parser.add_argument("--e", type=int, default=256)
    parser.add_argument("--h", type=int, default=3072)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    return parser.parse_args(_SCRIPT_ARGV)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    if args.mode == "correctness":
        run_correctness(args)
        return

    if args.sweep_bs:
        if args.min_b < 1 or args.max_b < args.min_b:
            raise ValueError("--sweep-bs requires 1 <= min-b <= max-b")
        base_case = make_case(args.h, args.max_b, args.topk, args.i, args.e, args.device)
        batch_sizes = range(args.min_b, args.max_b + 1)
    else:
        base_case = make_case(args.h, args.b, args.topk, args.i, args.e, args.device)
        batch_sizes = (args.b,)

    peak_memory_gbps = peak_memory_bandwidth_gbps(args.device)
    print_header(args, peak_memory_gbps)
    for B in batch_sizes:
        case = slice_case(base_case, B)
        row = benchmark_case(case, B, args.h, args.i, args.topk, args.rep, peak_memory_gbps)
        print_result(row, peak_memory_gbps)


if __name__ == "__main__":
    main()
