"""Checkpoint-format normalization for eval.py.

eval.py loads checkpoints written by two distinct training paths:

  - `experiments/distill-soft/scripts/train.py` saves a **bare**
    `torch.save(net.state_dict(), path)`. Loading returns a plain
    state_dict.
  - `experiments/selfplay/scripts/selfplay_loop_mp.py` saves a
    **wrapped** `{"net": state_dict, "opt": ..., "iter": ..., "hour": ...}`
    so resume can pick up optimizer + iter index.

`unwrap_state_dict` accepts either and returns the bare state_dict
ready for `net.load_state_dict(...)`. Sits in src/ so it's importable
from both eval.py and the test suite.

(Mirror of `selfplay.ckpt.load_net_state_dict` — duplicated rather
than imported because distill-soft doesn't take a hard dependency on
the selfplay package.)
"""
from __future__ import annotations

from typing import Any, Mapping


def unwrap_state_dict(obj: Any) -> Mapping[str, Any]:
    """Return the bare state_dict from either the legacy / supervised
    format (a raw state_dict) or the wrapped selfplay format
    ({'net': state_dict, ...}). Raises ValueError on anything else."""
    if isinstance(obj, dict) and "net" in obj and isinstance(obj["net"], dict):
        return obj["net"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(
        f"unsupported checkpoint format: {type(obj).__name__}; expected "
        "a state_dict or a wrapped {'net': state_dict, ...}")
