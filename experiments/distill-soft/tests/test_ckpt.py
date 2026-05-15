"""Tests for unwrap_state_dict — the eval-side checkpoint normalizer.

eval.py needs to load checkpoints written by both train.py (bare
state_dict) and the self-play loop (wrapped {"net": ..., "opt": ...}).
This regression class bit us once already — selfplay-final eval crashed
with `Unexpected key(s) in state_dict: "net", "opt", "iter", "hour"`
because the inline conditional in eval.py was added but not covered by
tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_soft.ckpt import unwrap_state_dict


def _toy_state_dict():
    return {"layer.weight": torch.ones(2, 2), "layer.bias": torch.zeros(2)}


def test_load_raw_state_dict():
    """The supervised-distill format: a bare state_dict (no wrapper)."""
    sd = _toy_state_dict()
    out = unwrap_state_dict(sd)
    assert out is sd
    assert "layer.weight" in out


def test_load_wrapped_selfplay_ckpt():
    """The self-play format: {'net': state_dict, 'opt': ..., 'iter': N}."""
    sd = _toy_state_dict()
    wrapped = {"net": sd, "opt": {"state": {}}, "iter": 42, "hour": 7}
    out = unwrap_state_dict(wrapped)
    assert out is sd
    assert "layer.weight" in out


def test_wrapped_form_takes_precedence_over_outer_keys():
    """An outer dict that has both a 'net' subdict and weight-like
    sibling keys must resolve to the inner state_dict — otherwise the
    nonsense sibling values get fed to load_state_dict."""
    inner = _toy_state_dict()
    outer = {"net": inner, "layer.weight": "trash-not-a-tensor"}
    out = unwrap_state_dict(outer)
    assert out is inner
    assert torch.is_tensor(out["layer.weight"])


def test_load_distill_soft_format_via_torch_save_load(tmp_path):
    """End-to-end: write a state_dict the way distill-soft/scripts/train.py
    does, read it back through torch.load, feed it into the unwrapper."""
    sd = _toy_state_dict()
    path = tmp_path / "distilled_epoch019.pt"
    torch.save(sd, path)
    loaded = torch.load(path)
    out = unwrap_state_dict(loaded)
    assert set(out.keys()) == set(sd.keys())
    for k in sd:
        assert torch.equal(out[k], sd[k])


def test_load_selfplay_format_via_torch_save_load(tmp_path):
    """End-to-end for the wrapped format the self-play loop writes —
    this is the exact regression that crashed eval-selfplay-final."""
    sd = _toy_state_dict()
    wrapped = {"net": sd, "opt": {}, "iter": 5, "hour": 1}
    path = tmp_path / "final.pt"
    torch.save(wrapped, path)
    loaded = torch.load(path)
    out = unwrap_state_dict(loaded)
    assert set(out.keys()) == set(sd.keys())
    for k in sd:
        assert torch.equal(out[k], sd[k])


def test_rejects_garbage_input():
    """Non-dict input should fail loudly rather than feed nonsense
    downstream to load_state_dict."""
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        unwrap_state_dict("a-string-not-a-dict")
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        unwrap_state_dict(42)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        unwrap_state_dict(None)


def test_empty_state_dict_is_accepted():
    """Edge case: an empty state_dict is still a dict and should pass
    through. Calling code is responsible for what it does with it."""
    out = unwrap_state_dict({})
    assert out == {}


def test_wrapped_with_non_dict_net_falls_through_to_outer():
    """If 'net' is present but isn't a dict (e.g. someone stuffed a
    tensor in there), the outer dict is treated as the state_dict —
    consistent with the 'is_dict and net is dict' guard."""
    sd_like = {"layer.weight": torch.ones(2, 2), "net": torch.zeros(3)}
    out = unwrap_state_dict(sd_like)
    assert out is sd_like
