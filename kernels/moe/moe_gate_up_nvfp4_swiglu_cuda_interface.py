import functools
import importlib
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import torch


THIS_DIR = Path(__file__).resolve().parent
CUDA_SOURCE = THIS_DIR / "moe_gate_up_nvfp4_swiglu_cuda.cu"
SETUP_SOURCE = THIS_DIR / "setup_moe_gate_up_nvfp4_swiglu.py"
EXTENSION_NAME = "moe_gate_up_nvfp4_swiglu_cuda"


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


def check_hidden_size(H: int) -> None:
    if H % 512 != 0:
        raise ValueError(f"H must be a multiple of 512; got H={H}")


def moe_gate_up_nvfp4_swiglu_cuda_out(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    B, H = x.shape
    E, two_I, H2 = w.shape
    I = two_I // 2
    topk = expert_ids.shape[1]
    check_hidden_size(H)

    if H2 != H // 2:
        raise ValueError(f"w.shape[2] must be H/2; got {H2} for H={H}")
    if w_scale.shape != (E, two_I, H // 16):
        raise ValueError(
            f"w_scale must have shape {(E, two_I, H // 16)}; "
            f"got {tuple(w_scale.shape)}"
        )
    if out.shape != (B, topk, I):
        raise ValueError(f"out must have shape {(B, topk, I)}; got {tuple(out.shape)}")

    ext = load_cuda_extension()
    ext.moe_gate_up_nvfp4_swiglu_out(x, w, w_scale, alpha, expert_ids, out)
    return out


def moe_gate_up_nvfp4_swiglu_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    B, _ = x.shape
    _, two_I, _ = w.shape
    I = two_I // 2
    topk = expert_ids.shape[1]
    out = torch.empty((B, topk, I), dtype=torch.bfloat16, device=x.device)
    return moe_gate_up_nvfp4_swiglu_cuda_out(x, w, w_scale, alpha, expert_ids, out)
