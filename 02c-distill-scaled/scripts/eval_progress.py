"""Aggregate per-worker eval progress files into one status line.

Each eval worker writes .eval_progress_w<NN>.json after every game. This
script reads them all and prints a single aggregated snapshot — same
pattern as scripts/progress.py for data generation.

Usage:
    python scripts/eval_progress.py <ckpt_dir>          # one-shot snapshot
    python scripts/eval_progress.py <ckpt_dir> --watch  # loop until all workers done
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime


def snapshot(progress_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(progress_dir, ".eval_progress_w*.json")))
    workers = []
    for p in paths:
        try:
            with open(p) as f:
                workers.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return {"ts": time.time(), "workers": workers}


def format_line(snap: dict) -> str:
    ws = snap["workers"]
    if not ws:
        return f"[{datetime.now():%H:%M:%S}] (no eval worker progress files yet)"
    total_done = sum(w["games_done"] for w in ws)
    total_target = sum(w["games_total"] for w in ws)
    n_workers = len(ws)
    n_done = sum(1 for w in ws if w.get("phase") == "done")

    rates = [w["games_done"] / w["elapsed_s"] for w in ws
             if w.get("elapsed_s", 0) > 5 and w["games_done"] > 0]
    avg_rate_per_worker = (sum(rates) / len(rates) * 60) if rates else 0.0
    total_rate = avg_rate_per_worker * n_workers

    eta_str = "—"
    if total_rate > 0:
        eta = (total_target - total_done) / total_rate
        eta_str = f"{eta:.0f}min" if eta < 120 else f"{eta/60:.1f}h"

    pct = (total_done / total_target * 100) if total_target else 0
    return (
        f"[{datetime.now():%H:%M:%S}] "
        f"eval {total_done}/{total_target} games ({pct:.1f}%)  "
        f"rate={total_rate:.1f} games/min  "
        f"workers_done={n_done}/{n_workers}  "
        f"eta={eta_str}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("progress_dir")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()

    if not args.watch:
        print(format_line(snapshot(args.progress_dir)), flush=True)
        return

    while True:
        snap = snapshot(args.progress_dir)
        print(format_line(snap), flush=True)
        ws = snap["workers"]
        if ws and all(w.get("phase") == "done" for w in ws):
            print("[eval_progress] all workers done — exiting", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
