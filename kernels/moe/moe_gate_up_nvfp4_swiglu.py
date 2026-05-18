import argparse
import glob
from pathlib import Path
import shutil
import subprocess
import sys

_SCRIPT_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import torch
import torch.nn.functional as F
import triton.testing

from moe_gate_up_nvfp4_swiglu_cuda_interface import (
    check_hidden_size,
    load_cuda_extension,
    moe_gate_up_nvfp4_swiglu_cuda,
    moe_gate_up_nvfp4_swiglu_cuda_out,
)


THIS_DIR = Path(__file__).resolve().parent


# Backward-compatible names for older local commands.
_moe_gate_up_nvfp4_swiglu_fusion_cuda = moe_gate_up_nvfp4_swiglu_cuda
_moe_gate_up_nvfp4_swiglu_fusion_cuda_out = moe_gate_up_nvfp4_swiglu_cuda_out
_moe_gate_up_nvfp4_swiglu_fusion_cute = moe_gate_up_nvfp4_swiglu_cuda
_moe_gate_up_nvfp4_swiglu_fusion_cute_out = moe_gate_up_nvfp4_swiglu_cuda_out
_check_hidden_size = check_hidden_size
_load_cuda_extension = load_cuda_extension


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


def moe_gate_up_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    e = expert_ids.long()
    w_sel = w[e]
    alpha_sel = alpha[e].unsqueeze(-1)
    out = torch.einsum("bh,bkhi->bki", x.float(), w_sel.float())
    return (out * alpha_sel).to(torch.bfloat16)


def moe_gate_up_swiglu_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    e = expert_ids.long()
    w_sel = w[e]
    alpha_sel = alpha[e].unsqueeze(-1)
    gate_up = torch.einsum("bh,bkhi->bki", x.float(), w_sel.float())
    gate_up = gate_up * alpha_sel
    out = F.silu(gate_up[..., 0::2]) * gate_up[..., 1::2]
    return out.to(torch.bfloat16)


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


def moe_gate_up_nvfp4_swiglu_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
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
            part_out = torch.empty(
                (B, topk, i1 - i0), dtype=torch.float32, device=x.device
            )
            for b in range(B):
                for k in range(topk):
                    e = int(expert_ids[b, k].item())
                    w_chunk = _nvfp4_e2m1_to_bf16(w[e, part, i0:i1]).view(
                        i1 - i0, num_blocks, block_size
                    )
                    partial = (
                        x_blocks[b].unsqueeze(0).float() * w_chunk.float()
                    ).sum(dim=-1)
                    scaled = partial * w_scale[e, part, i0:i1]
                    part_out[b, k] = scaled.sum(dim=-1) * alpha[e]
            gate_up_chunks.append(part_out)

        gate, up = gate_up_chunks
        silu = (0.5 * gate) * torch.tanh(0.5 * gate) + (0.5 * gate)
        out[..., i0:i1] = (silu * up).to(torch.bfloat16)
    return out


def make_launch_case(
    H: int,
    B: int,
    topk: int,
    I: int,
    E: int,
    device: str,
):
    x = torch.randn(B, H, dtype=torch.bfloat16, device=device)
    w = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w_scale = torch.randn(E, 2 * I, H // 16, device=device).to(torch.float8_e4m3fn)
    alpha = torch.rand(E, dtype=torch.float32, device=device)
    expert_ids = torch.randint(0, E, (B, topk), dtype=torch.int32, device=device)
    out = torch.empty((B, topk, I), dtype=torch.bfloat16, device=device)

    check_hidden_size(H)
    ext = load_cuda_extension()

    def launch():
        ext.moe_gate_up_nvfp4_swiglu_out(x, w, w_scale, alpha, expert_ids, out)

    keepalive = (x, w, w_scale, alpha, expert_ids, out, ext)
    return launch, out, keepalive


_prepare_launch_case = make_launch_case


def run_correctness_case(
    H: int,
    B: int,
    topk: int,
    I: int,
    E: int,
    device: str,
) -> None:
    x = torch.randn(B, H, dtype=torch.bfloat16, device=device)
    w = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w_scale = torch.randn(E, 2 * I, H // 16, device=device).to(torch.float8_e4m3fn)
    alpha = torch.rand(E, dtype=torch.float32, device=device)
    expert_ids = torch.randint(0, E, (B, topk), dtype=torch.int32, device=device)

    out = moe_gate_up_nvfp4_swiglu_cuda(x, w, w_scale, alpha, expert_ids)
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


_run_correctness_case = run_correctness_case


def run_benchmark_case(
    H: int,
    B: int,
    topk: int,
    I: int,
    E: int,
    device: str,
    rep: int,
    peak_memory_gbps: float | None,
) -> None:
    launch, out, keepalive = make_launch_case(H, B, topk, I, E, device)
    launch()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench_cudagraph(launch, rep=rep)

    dot_flops = 4.0 * B * topk * I * H
    dot_tflops = dot_flops / (ms * 1.0e-3) / 1.0e12
    output_elems = B * topk * I

    weight_scale_meta_out_bytes = output_elems * (H + H / 8.0 + 4 + 4 + 2)
    activation_once_bytes = B * H * 2
    estimated_hbm_bytes = weight_scale_meta_out_bytes + activation_once_bytes
    estimated_hbm_gbps = estimated_hbm_bytes / (ms * 1.0e-3) / 1.0e9

    logical_bytes = output_elems * (H + 4.0 * H + H / 8.0 + 4 + 4 + 2)
    logical_gbps = logical_bytes / (ms * 1.0e-3) / 1.0e9

    if peak_memory_gbps is None:
        memory_msg = f"est_hbm={estimated_hbm_gbps / 1000.0:.2f} TB/s"
    else:
        memory_msg = (
            f"est_hbm={estimated_hbm_gbps / 1000.0:.2f} TB/s "
            f"({estimated_hbm_gbps / peak_memory_gbps * 100.0:.1f}% peak)"
        )
    print(
        f"H={H}: {ms:.4f} ms, projection={dot_tflops:.2f} TFLOP/s, "
        f"{memory_msg}, logical={logical_gbps / 1000.0:.2f} TB/s, "
        f"out={tuple(out.shape)}"
    )

    del launch, out, keepalive
    torch.cuda.empty_cache()


_run_benchmark_case = run_benchmark_case


def run_ncu_target_case(
    H: int,
    B: int,
    topk: int,
    I: int,
    E: int,
    device: str,
    warmup: int,
    profile_iters: int,
) -> None:
    launch, out, keepalive = make_launch_case(H, B, topk, I, E, device)

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    print(
        f"NCU target: H={H} B={B} topk={topk} I={I} E={E} "
        f"warmup={warmup} profile_iters={profile_iters} out={tuple(out.shape)}"
    )

    cudart = torch.cuda.cudart()
    err = cudart.cudaProfilerStart()
    if err != 0:
        raise RuntimeError(f"cudaProfilerStart failed with {err}")
    for _ in range(profile_iters):
        launch()
    torch.cuda.synchronize()
    err = cudart.cudaProfilerStop()
    if err != 0:
        raise RuntimeError(f"cudaProfilerStop failed with {err}")
    torch.cuda.synchronize()
    print("NCU target complete")

    del launch, out, keepalive
    torch.cuda.empty_cache()


_run_ncu_target_case = run_ncu_target_case


def resolve_ncu(ncu_bin: str) -> str | None:
    ncu = shutil.which(ncu_bin)
    if ncu is not None:
        return ncu
    for pattern in (
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda-*/bin/ncu",
        "/opt/nvidia/nsight-compute/*/ncu",
        "/usr/local/NVIDIA-Nsight-Compute*/ncu",
    ):
        for candidate in sorted(glob.glob(pattern), reverse=True):
            ncu = shutil.which(candidate)
            if ncu is not None:
                return ncu
    return None


_resolve_ncu = resolve_ncu


def run_ncu(args: argparse.Namespace) -> None:
    ncu = resolve_ncu(args.ncu_bin)
    if ncu is None:
        raise RuntimeError("ncu was not found")

    target_cmd = [
        sys.executable,
        __file__,
        "--mode",
        "ncu-target",
        "--h",
        str(args.h),
        "--b",
        str(args.b),
        "--topk",
        str(args.topk),
        "--i",
        str(args.i),
        "--e",
        str(args.e),
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--profile-iters",
        str(args.profile_iters),
    ]
    cmd = [
        ncu,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--launch-count",
        str(args.ncu_launch_count),
        "--section",
        "SpeedOfLight",
        "--section",
        "MemoryWorkloadAnalysis",
        "--section",
        "SchedulerStats",
        "--section",
        "WarpStateStats",
        "--section",
        "LaunchStats",
        "--section",
        "Occupancy",
        "--force-overwrite",
        "--export",
        args.ncu_output,
        *target_cmd,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


_run_ncu = run_ncu


def parse_hidden_sizes(hidden_sizes: str) -> tuple[int, ...]:
    return tuple(int(item) for item in hidden_sizes.split(",") if item)


_parse_hidden_sizes = parse_hidden_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("benchmark", "correctness", "ncu", "ncu-target"),
        default="benchmark",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--i", type=int, default=1536)
    parser.add_argument("--e", type=int, default=256)
    parser.add_argument("--h", type=int, default=3072)
    parser.add_argument("--hidden-sizes", default="512,1024,2048,3072,4096")
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--ncu-launch-count", type=int, default=1)
    parser.add_argument("--ncu-output", default="moe_gate_up_h3072")
    parser.add_argument("--ncu-bin", default="ncu")
    return parser.parse_args(_SCRIPT_ARGV)


_parse_args = parse_args


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    B = args.b
    topk = args.topk
    I = args.i
    E = args.e
    device = args.device

    if args.mode == "ncu":
        run_ncu(args)
        return

    if args.mode == "ncu-target":
        run_ncu_target_case(
            args.h, B, topk, I, E, device, args.warmup, args.profile_iters
        )
        return

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    if args.mode == "correctness":
        for H in hidden_sizes:
            run_correctness_case(H, B, topk, I, E, device)
        print("PASS")
        return

    peak_memory_gbps = peak_memory_bandwidth_gbps(device)
    peak_msg = (
        "unknown"
        if peak_memory_gbps is None
        else f"{peak_memory_gbps / 1000.0:.2f} TB/s"
    )
    print(
        f"Benchmark: B={B} topk={topk} I={I} E={E} rep={args.rep} "
        f"peak_mem={peak_msg}"
    )
    for H in hidden_sizes:
        run_benchmark_case(H, B, topk, I, E, device, args.rep, peak_memory_gbps)


if __name__ == "__main__":
    main()
