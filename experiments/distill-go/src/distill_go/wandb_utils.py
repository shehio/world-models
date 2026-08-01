"""Optional Weights & Biases hooks for the distill-go scripts.

Copied from wm_chess/src/wm_chess/wandb_utils.py (distill-go is a standalone
uv project, so no reason to depend on the chess workspace member).

Design goal: tracking is strictly additive. Every script that logs to
wandb also keeps its existing JSON-history/stdout behavior, and must run
unchanged when wandb is unavailable. So `init_wandb` returns either a
live run or None, and callers guard every log call with `if run is not
None`. Three ways to turn tracking off:

  1. pass --no-wandb to the script,
  2. export WANDB_MODE=disabled,
  3. just don't install wandb (init prints a notice and returns None).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

WANDB_ENTITY = "shehio"
WANDB_PROJECT = "world-models"


def init_wandb(args: Any, *, tags: list[str], name: str | None = None,
               config_extra: dict | None = None):
    """Start a wandb run, or return None when tracking is off/unavailable.

    `args` is the script's parsed argparse namespace; every hyperparameter
    in it lands in the run config (plus anything in `config_extra`, e.g.
    derived values like parameter counts).
    """
    if getattr(args, "no_wandb", False):
        return None
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] package not installed; continuing without tracking",
              flush=True)
        return None

    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items() if k != "no_wandb"}
    if config_extra:
        config.update(config_extra)
    # WANDB_TAGS (comma-separated) appends to the script's own tags, so a
    # launcher can mark runs (e.g. WANDB_TAGS=smoke-ab) without code changes.
    env_tags = [t for t in os.environ.get("WANDB_TAGS", "").split(",") if t]
    all_tags = list(tags) + [t for t in env_tags if t not in tags]

    try:
        return wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                          name=name, config=config, tags=all_tags)
    except Exception as e:  # no API key, no network, etc.
        print(f"[wandb] init failed ({e}); continuing without tracking",
              flush=True)
        return None


def set_summary(run, metrics: dict) -> None:
    """Write final results (eval scores, Elo estimates) as summary metrics."""
    if run is None:
        return
    for k, v in metrics.items():
        run.summary[k] = v


def finish(run) -> None:
    if run is None:
        return
    run.finish()
