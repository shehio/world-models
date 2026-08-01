"""Turn the driver's `history` dict into one flat wandb record per iter.

`train_loop` already accumulates everything worth tracking in
`history` (one row per iter under "games", "losses" and "evals"), and
hands it to `on_iter_end(it, network, history)`. Both runners
(run_local.py, run_cloud.py) use this to mirror that iter's rows to
wandb without changing the driver or the history JSON.

Rows are keyed by iter, and "losses"/"evals" only get a row on the
iters where training / eval actually ran, so we take the tail row only
when its iter matches.
"""
from __future__ import annotations


def _tail_for_iter(rows: list[dict], it: int) -> dict:
    if rows and rows[-1].get("iter") == it:
        return {k: v for k, v in rows[-1].items() if k != "iter"}
    return {}


def iter_record(history: dict, it: int) -> dict:
    """Flat {metric: value} for iteration `it`, ready for wandb.log().

    Eval metrics are prefixed `eval_` so they can't collide with the
    self-play/loss keys of the same name (e.g. "losses").
    """
    record: dict = {"iter": it}
    record.update(_tail_for_iter(history.get("games", []), it))
    record.update(_tail_for_iter(history.get("losses", []), it))
    record.update({f"eval_{k}": v
                   for k, v in _tail_for_iter(history.get("evals", []), it).items()})
    return record
