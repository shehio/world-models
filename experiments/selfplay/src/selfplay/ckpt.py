"""Checkpoint loading utilities for the self-play loop.

The self-play loop is initialized from one of two checkpoint formats:

  Legacy / distilled-supervised: a raw torch.state_dict() (just the
      tensors keyed by layer name). Written by
      experiments/distill-soft/scripts/train.py and the older v1-v4
      selfplay runs.

  Self-play wrapped: {"net": state_dict, "opt": ..., "iter": ...,
      "hour": ...}. Written by recent selfplay_loop_mp.py runs that
      bundle the optimizer state + iter index alongside the weights so
      runs can resume cleanly.

`load_net_state_dict` returns the bare state_dict from either format,
ready to pass to `network.load_state_dict(...)`. Sits in src/ so it's
importable from both the script and the test suite.
"""
from __future__ import annotations

from typing import Any, Mapping


def load_net_state_dict(obj: Any) -> Mapping[str, Any]:
    """Extract the network state_dict from a checkpoint object.

    Accepts:
      - A raw state_dict (the legacy / distilled-supervised format).
      - A wrapped dict with key "net" holding the state_dict (the
        newer selfplay format).

    Returns the bare state_dict. Raises ValueError on anything else.
    """
    if isinstance(obj, dict) and "net" in obj and isinstance(obj["net"], dict):
        return obj["net"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(
        f"unsupported checkpoint format: {type(obj).__name__}; expected "
        "a state_dict (dict[str, Tensor]) or a wrapped {'net': state_dict, ...}")
