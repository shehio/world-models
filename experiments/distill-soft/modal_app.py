"""Modal runner for the distill-soft training job.

Same script, same flags, different machine: this wraps
`scripts/train.py` so a training run can go to a Modal GPU instead of an
EKS pod or a bare EC2 box, without a second copy of the training code.

The GPU is L40S by default because that is what the headline runs used
(one L40S, ~16 GPU-hours). Pass --gpu A100 if L40S is unavailable in the
region or the account has no L40S quota; anything Modal accepts as a gpu
spec works.

The wandb API key comes from a Modal secret named "wandb" (create it
once with `modal secret create wandb WANDB_API_KEY=...`). Without it the
run still trains: wandb_utils degrades to no-op tracking.

Data and checkpoints live on a Modal volume so a run can be resumed and
the checkpoints outlive the container.

Usage (from experiments/distill-soft/):
    modal run modal_app.py --data /data/multipv.npz --epochs 20
    modal run modal_app.py --gpu A100 --smoke          # wiring check only

Nothing here runs at import time, so `python -m py_compile modal_app.py`
is a valid check without a Modal token.
"""
from __future__ import annotations

import shlex
import subprocess
import sys

import modal

APP_NAME = "world-models-distill-soft"
VOLUME_NAME = "world-models-distill-soft"
REMOTE_ROOT = "/root/distill-soft"
VOLUME_MOUNT = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5",
        "numpy>=1.26",
        "python-chess>=1.999",
        "tqdm>=4.66",
        "wandb>=0.19",
    )
    # The two local packages the training script imports. Shipping the
    # source (rather than pip-installing from git) keeps the container in
    # sync with the working tree, which is the point of a cloud runner
    # you drive from your laptop.
    .add_local_dir("src", f"{REMOTE_ROOT}/src")
    .add_local_dir("scripts", f"{REMOTE_ROOT}/scripts")
    .add_local_dir("../../wm_chess/src", f"{REMOTE_ROOT}/wm_chess_src")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    gpu="L40S",
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(argv: list[str]) -> int:
    """Run scripts/train.py with `argv` inside the container."""
    return _run(argv)


@app.function(
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_a100(argv: list[str]) -> int:
    """Identical to `train`, on A100. Two functions rather than one
    parameterized on gpu= because Modal resolves the gpu spec when the
    function is registered, not when it is called."""
    return _run(argv)


def _run(argv: list[str]) -> int:
    cmd = [sys.executable, f"{REMOTE_ROOT}/scripts/train.py", *argv]
    env_pythonpath = f"{REMOTE_ROOT}/src:{REMOTE_ROOT}/wm_chess_src"
    print(f"[modal] {shlex.join(cmd)}  (PYTHONPATH={env_pythonpath})", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=REMOTE_ROOT,
        env={"PYTHONPATH": env_pythonpath, "PATH": "/usr/local/bin:/usr/bin:/bin",
             **_wandb_env()},
    )
    # Flush checkpoints written under /data back to the volume so they
    # survive the container.
    volume.commit()
    return proc.returncode


def _wandb_env() -> dict:
    import os
    return {k: v for k, v in os.environ.items()
            if k.startswith("WANDB_") or k == "HOME"}


@app.local_entrypoint()
def main(
    data: str = f"{VOLUME_MOUNT}/multipv.npz",
    ckpt_dir: str = f"{VOLUME_MOUNT}/checkpoints/run01",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    n_blocks: int = 20,
    n_filters: int = 256,
    value_weight: float = 1.0,
    hard_targets: bool = False,
    amp: bool = False,
    smoke: bool = False,
    no_wandb: bool = False,
    gpu: str = "L40S",
) -> None:
    """Mirrors the scripts/train.py CLI, one flag per hyperparameter."""
    argv = [
        "--ckpt-dir", ckpt_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--n-blocks", str(n_blocks),
        "--n-filters", str(n_filters),
        "--value-weight", str(value_weight),
        "--device", "cuda",
    ]
    if smoke:
        argv.append("--smoke")
    else:
        argv += ["--data", data]
    if hard_targets:
        argv.append("--hard-targets")
    if amp:
        argv.append("--amp")
    if no_wandb:
        argv.append("--no-wandb")

    fn = train_a100 if gpu.upper() == "A100" else train
    rc = fn.remote(argv)
    print(f"[modal] train.py exited {rc}")
