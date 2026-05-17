import argparse
import glob
from pathlib import Path
import shutil
import subprocess
import sys

_SCRIPT_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import torch
import triton.testing

from moe_down_combine_nvfp4_cuda_interface import (
    check_intermediate_size,
    load_cuda_extension,
    moe_down_combine_nvfp4_cuda,
    moe_down_combine_nvfp4_cuda_out,
)


THIS_DIR = Path(__file__).resolve().parent


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


def moe_down_combine_nvfp4_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    chunk_h: int = 64,
) -> torch.Tensor:
    B, topk, I = x.shape
    E, H, I2 = w.shape
    if I2 != I // 2:
        raise ValueError("w must have shape [E, H, I/2]")

    block_size = 16
    num_blocks = I // block_size
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
                scaled = partial * w_scale[expert, h0:h1]
                out_chunk[b] += scaled.sum(dim=-1) * route_weights[b, route]
        out[:, h0:h1] = out_chunk

    return out.to(torch.bfloat16)


def make_launch_case(
    H: int,
    B: int,
    topk: int,
    I: int,
    E: int,
    device: str,
):
    x = torch.randn(B, topk, I, dtype=torch.bfloat16, device=device)
    w = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device)
    w_scale = torch.randn(E, H, I // 16, device=device).to(torch.float8_e4m3fn)
    expert_ids = torch.randint(0, E, (B, topk), dtype=torch.int32, device=device)
    route_weights = torch.rand(B, topk, dtype=torch.float32, device=device)
    out = torch.empty((B, H), dtype=torch.bfloat16, device=device)

    check_intermediate_size(I)
    ext = load_cuda_extension()

    def launch():
        ext.moe_down_combine_nvfp4_out(
            x, w, w_scale, expert_ids, route_weights, out
        )

    keepalive = (x, w, w_scale, expert_ids, route_weights, out, ext)
    return launch, out, keepalive


def run_correctness_case(
    H: int,
    B: int,
    topk: int,
    I: int,
    E: int,
    device: str,
) -> None:
    x = torch.randn(B, topk, I, dtype=torch.bfloat16, device=device)
    w = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device)
    w_scale = torch.randn(E, H, I // 16, device=device).to(torch.float8_e4m3fn)
    expert_ids = torch.randint(0, E, (B, topk), dtype=torch.int32, device=device)
    route_weights = torch.rand(B, topk, dtype=torch.float32, device=device)

    out = moe_down_combine_nvfp4_cuda(x, w, w_scale, expert_ids, route_weights)
    ref = moe_down_combine_nvfp4_ref(x, w, w_scale, expert_ids, route_weights)
    torch.cuda.synchronize()

    diff = (out.float() - ref.float()).abs()
    print(
        f"H={H} I={I}: out={tuple(out.shape)} max_abs={diff.max().item():.6g} "
        f"mean_abs={diff.mean().item():.6g}"
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    del x, w, w_scale, expert_ids, route_weights, out, ref, diff
    torch.cuda.empty_cache()


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

    dot_flops = 2.0 * B * topk * H * I
    dot_tflops = dot_flops / (ms * 1.0e-3) / 1.0e12
    output_elems = B * H

    unique_activation_bytes = B * topk * I * 2
    weight_scale_bytes = output_elems * topk * (I / 2.0 + I / 16.0)
    meta_output_bytes = B * topk * (4 + 4) + output_elems * 2
    min_hbm_bytes = unique_activation_bytes + weight_scale_bytes + meta_output_bytes
    min_hbm_gbps = min_hbm_bytes / (ms * 1.0e-3) / 1.0e9
    logical_bytes = output_elems * (topk * (I + I / 2.0 + I / 16.0 + 4) + 2)
    logical_gbps = logical_bytes / (ms * 1.0e-3) / 1.0e9

    if peak_memory_gbps is None:
        memory_msg = f"min_hbm={min_hbm_gbps / 1000.0:.2f} TB/s"
    else:
        memory_msg = (
            f"min_hbm={min_hbm_gbps / 1000.0:.2f} TB/s "
            f"({min_hbm_gbps / peak_memory_gbps * 100.0:.1f}% peak)"
        )
    print(
        f"H={H} I={I}: {ms:.4f} ms, projection={dot_tflops:.2f} TFLOP/s, "
        f"{memory_msg}, logical={logical_gbps / 1000.0:.2f} TB/s, "
        f"out={tuple(out.shape)}"
    )

    del launch, out, keepalive
    torch.cuda.empty_cache()


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
        f"NCU target: H={H} I={I} B={B} topk={topk} E={E} "
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
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--ncu-launch-count", type=int, default=1)
    parser.add_argument("--ncu-output", default="moe_down_combine_h3072")
    parser.add_argument("--ncu-bin", default="ncu")
    return parser.parse_args(_SCRIPT_ARGV)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    if args.mode == "ncu":
        run_ncu(args)
        return

    if args.mode == "ncu-target":
        run_ncu_target_case(
            args.h,
            args.b,
            args.topk,
            args.i,
            args.e,
            args.device,
            args.warmup,
            args.profile_iters,
        )
        return

    if args.mode == "correctness":
        run_correctness_case(args.h, args.b, args.topk, args.i, args.e, args.device)
        print("PASS")
        return

    peak_memory_gbps = peak_memory_bandwidth_gbps(args.device)
    peak_msg = (
        "unknown"
        if peak_memory_gbps is None
        else f"{peak_memory_gbps / 1000.0:.2f} TB/s"
    )
    print(
        f"Benchmark: B={args.b} topk={args.topk} H={args.h} I={args.i} "
        f"E={args.e} rep={args.rep} peak_mem={peak_msg}"
    )
    run_benchmark_case(
        args.h,
        args.b,
        args.topk,
        args.i,
        args.e,
        args.device,
        args.rep,
        peak_memory_gbps,
    )


if __name__ == "__main__":
    main()
