"""Tests for the cross-format checkpoint loader.

The self-play loop is initialized from either a raw state_dict (the
format that experiments/distill-soft/scripts/train.py writes) or a
wrapped {"net": state_dict, ...} dict (what the self-play loop itself
writes). load_net_state_dict normalizes both to a bare state_dict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfplay.ckpt import load_net_state_dict


def _toy_state_dict():
    return {"layer.weight": torch.ones(2, 2), "layer.bias": torch.zeros(2)}


def test_load_raw_state_dict():
    """The distilled-supervised format: a bare state_dict (no wrapper)."""
    sd = _toy_state_dict()
    out = load_net_state_dict(sd)
    assert out is sd  # no copy — just passed through
    assert "layer.weight" in out


def test_load_wrapped_selfplay_ckpt():
    """The self-play format: {'net': state_dict, 'opt': ..., 'iter': N}."""
    sd = _toy_state_dict()
    wrapped = {"net": sd, "opt": {"state": {}}, "iter": 42, "hour": 7}
    out = load_net_state_dict(wrapped)
    assert out is sd
    assert "layer.weight" in out


def test_wrapped_form_takes_precedence_over_outer_keys():
    """An outer dict that happens to contain weight-like keys AND a
    'net' subdict must still resolve to the 'net' subdict (the wrapper
    form), not the outer dict."""
    inner = _toy_state_dict()
    outer = {"net": inner, "layer.weight": "trash-not-a-tensor"}
    out = load_net_state_dict(outer)
    assert out is inner
    assert torch.is_tensor(out["layer.weight"])


def test_load_distill_soft_format_via_torch_save_load(tmp_path):
    """End-to-end: write a state_dict the way distill-soft/scripts/train.py
    does, read it back through torch.load, feed it into the loader."""
    sd = _toy_state_dict()
    path = tmp_path / "distilled_epoch019.pt"
    torch.save(sd, path)
    loaded = torch.load(path)
    out = load_net_state_dict(loaded)
    assert set(out.keys()) == set(sd.keys())
    for k in sd:
        assert torch.equal(out[k], sd[k])


def test_load_selfplay_format_via_torch_save_load(tmp_path):
    """End-to-end for the wrapped format that the self-play loop saves."""
    sd = _toy_state_dict()
    wrapped = {"net": sd, "opt": {}, "iter": 5, "hour": 1}
    path = tmp_path / "iter_005.pt"
    torch.save(wrapped, path)
    loaded = torch.load(path)
    out = load_net_state_dict(loaded)
    assert set(out.keys()) == set(sd.keys())


def test_rejects_garbage_input():
    """Anything that isn't a dict (or a dict without a 'net' subdict
    and without state_dict-shaped contents) should fail loudly rather
    than feed nonsense to load_state_dict."""
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_net_state_dict("a-string-not-a-dict")
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_net_state_dict(42)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_net_state_dict(None)


def test_empty_state_dict_is_accepted():
    """Edge case: an empty state_dict is still a dict and should pass
    through. Calling code is responsible for what it does with it."""
    out = load_net_state_dict({})
    assert out == {}
