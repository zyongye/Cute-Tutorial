import functools
import importlib
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import torch


THIS_DIR = Path(__file__).resolve().parent
CUDA_SOURCE = THIS_DIR / "moe_down_combine_nvfp4_cuda.cu"
SETUP_SOURCE = THIS_DIR / "setup_moe_down_combine_nvfp4.py"
EXTENSION_NAME = "moe_down_combine_nvfp4_cuda"


def configure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for candidate in (
        "/usr/local/cuda-13.2",
        "/usr/local/cuda-13.1",
        "/usr/local/cuda",
    ):
        cuda_home = Path(candidate)
        if (cuda_home / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(cuda_home)
            os.environ.setdefault("CUDA_PATH", str(cuda_home))
            os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"
            lib_paths = [cuda_home / "nvvm" / "lib64", cuda_home / "lib64"]
            os.environ["LD_LIBRARY_PATH"] = ":".join(
                [str(path) for path in lib_paths]
                + [os.environ.get("LD_LIBRARY_PATH", "")]
            )
            return


@functools.cache
def load_cuda_extension():
    configure_cuda_home()
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    extension_path = THIS_DIR / f"{EXTENSION_NAME}{suffix}"
    sources = (CUDA_SOURCE, SETUP_SOURCE)
    needs_build = not extension_path.exists() or any(
        path.stat().st_mtime > extension_path.stat().st_mtime for path in sources
    )

    if needs_build:
        env = os.environ.copy()
        env.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
        tmpdir = THIS_DIR / "build" / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(tmpdir)
        env["TMP"] = str(tmpdir)
        env["TEMP"] = str(tmpdir)
        subprocess.run(
            [sys.executable, str(SETUP_SOURCE), "build_ext", "--inplace"],
            cwd=THIS_DIR,
            env=env,
            check=True,
        )

    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    importlib.invalidate_caches()
    return importlib.import_module(EXTENSION_NAME)


def check_intermediate_size(I: int) -> None:
    if I % 512 != 0:
        raise ValueError(f"I must be a multiple of 512; got I={I}")


def moe_down_combine_nvfp4_cuda_out(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    B, topk, I = x.shape
    E, H, I2 = w.shape
    check_intermediate_size(I)

    if I2 != I // 2:
        raise ValueError(f"w.shape[2] must be I/2; got {I2} for I={I}")
    if w_scale.shape != (E, H, I // 16):
        raise ValueError(
            f"w_scale must have shape {(E, H, I // 16)}; "
            f"got {tuple(w_scale.shape)}"
        )
    if expert_ids.shape != (B, topk):
        raise ValueError(
            f"expert_ids must have shape {(B, topk)}; got {tuple(expert_ids.shape)}"
        )
    if route_weights.shape != (B, topk):
        raise ValueError(
            f"route_weights must have shape {(B, topk)}; "
            f"got {tuple(route_weights.shape)}"
        )
    if out.shape != (B, H):
        raise ValueError(f"out must have shape {(B, H)}; got {tuple(out.shape)}")

    ext = load_cuda_extension()
    ext.moe_down_combine_nvfp4_out(x, w, w_scale, expert_ids, route_weights, out)
    return out


def moe_down_combine_nvfp4_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    B, _, _ = x.shape
    _, H, _ = w.shape
    out = torch.empty((B, H), dtype=torch.bfloat16, device=x.device)
    return moe_down_combine_nvfp4_cuda_out(
        x, w, w_scale, expert_ids, route_weights, out
    )
