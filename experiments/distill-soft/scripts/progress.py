"""Aggregate per-worker progress files into a single status line.

Background: data generation runs as N spawn'd worker processes in an mp.Pool.
With `spawn` start method, worker stdout doesn't reach the parent, so the
parent log file is silent for hours while workers grind. Each worker writes
a JSON file every game to a shared progress dir; this script reads them and
prints one aggregated line.

Usage:
    python scripts/progress.py <progress_dir>          # one-shot snapshot
    python scripts/progress.py <progress_dir> --watch  # loop every 60s

Designed to be called from a background bash loop in `run_overnight.sh`,
so the combined log gets a progress line every minute.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime


def snapshot(progress_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(progress_dir, ".progress_w*.json")))
    workers = []
    for p in paths:
        try:
            with open(p) as f:
                workers.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            # Mid-write or stale; skip this tick.
            continue
    return {"ts": time.time(), "workers": workers}


def format_line(snap: dict) -> str:
    ws = snap["workers"]
    if not ws:
        return f"[{datetime.now():%H:%M:%S}] (no worker progress files yet)"

    total_done = sum(w["games_done"] for w in ws)
    total_target = sum(w["games_total"] for w in ws)
    total_positions = sum(w.get("positions", 0) for w in ws)
    n_workers = len(ws)

    # Per-worker rate: each worker's games_done / elapsed_s, then average.
    rates = []
    for w in ws:
        e = w.get("elapsed_s", 0)
        if e > 5:
            rates.append(w["games_done"] / e)
    avg_rate_per_worker_per_min = (sum(rates) / len(rates) * 60) if rates else 0.0
    total_rate_per_min = avg_rate_per_worker_per_min * n_workers

    eta_str = "—"
    if total_rate_per_min > 0:
        eta_min = (total_target - total_done) / total_rate_per_min
        eta_str = f"{eta_min:.0f}min" if eta_min < 120 else f"{eta_min/60:.1f}h"

    pct = (total_done / total_target * 100) if total_target else 0
    n_done = sum(1 for w in ws if w.get("phase") == "done")

    return (
        f"[{datetime.now():%H:%M:%S}] "
        f"{total_done}/{total_target} games ({pct:.1f}%)  "
        f"positions={total_positions:,}  "
        f"rate={total_rate_per_min:.1f} games/min  "
        f"workers_done={n_done}/{n_workers}  "
        f"eta={eta_str}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("progress_dir", help="directory containing .progress_wNN.json files")
    p.add_argument("--watch", action="store_true", help="loop forever, printing every --interval seconds")
    p.add_argument("--interval", type=int, default=60)
    args = p.parse_args()

    if not args.watch:
        print(format_line(snapshot(args.progress_dir)), flush=True)
        return

    while True:
        snap = snapshot(args.progress_dir)
        print(format_line(snap), flush=True)
        # Exit cleanly once every worker reports done.
        ws = snap["workers"]
        if ws and all(w.get("phase") == "done" for w in ws):
            print("[progress] all workers done — exiting watcher", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
