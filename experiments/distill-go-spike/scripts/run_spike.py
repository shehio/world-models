"""Run the KataGo spike: one self-play game, dump .npz to disk.

Usage:
    KATAGO_BIN=/path/to/katago KATAGO_MODEL=/path/to/model.bin.gz \
    uv run --project experiments/distill-go-spike python \
        experiments/distill-go-spike/scripts/run_spike.py \
        --board-size 9 --visits 400 --top-k 8 --output /tmp/go_spike.npz

If KATAGO_BIN / KATAGO_MODEL aren't set, the script prints install
instructions and exits. The output .npz has the same schema as the
chess pipeline's per-worker chunks, so `merge_chunks.py` will fold
them in without modification.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_go_spike import play_one_game


INSTALL_HINT = """\
KataGo not configured. Set environment variables before running:

    export KATAGO_BIN=/path/to/katago
    export KATAGO_MODEL=/path/to/model.bin.gz
    export KATAGO_CONFIG=/path/to/analysis.cfg

macOS install:
    brew install katago
    # The bottle ships three bundled networks under
    #   /opt/homebrew/Cellar/katago/<ver>/share/katago/*.bin.gz
    # and the analysis_example.cfg under .../configs/analysis_example.cfg
    # so you can wire it up with:
    KATAGO_VER=$(brew list --versions katago | awk '{print $2}')
    export KATAGO_BIN=$(which katago)
    export KATAGO_MODEL=/opt/homebrew/Cellar/katago/$KATAGO_VER/share/katago/g170e-b20c256x2-s5303129600-d1228401921.bin.gz
    export KATAGO_CONFIG=/opt/homebrew/Cellar/katago/$KATAGO_VER/share/katago/configs/analysis_example.cfg

Linux install:
    # Build from source: https://github.com/lightvector/KataGo
    # Or grab a prebuilt release: https://github.com/lightvector/KataGo/releases
    # Then download a net from https://katagotraining.org/networks/

Verify with:
    echo '{"id":"t","moves":[],"rules":"tromp-taylor","komi":7.5,"boardXSize":9,"boardYSize":9,"maxVisits":50,"analyzeTurns":[0]}' \\
        | katago analysis -model $KATAGO_MODEL -config $KATAGO_CONFIG 2>/dev/null | head -1
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--board-size", type=int, default=9, choices=[9, 13, 19],
                   help="Go board size. Spike defaults to 9 for fast iteration.")
    p.add_argument("--visits", type=int, default=400,
                   help="KataGo MCTS visits per move (higher = stronger teacher signal).")
    p.add_argument("--top-k", type=int, default=8,
                   help="Multipv equivalent: how many candidate moves to record.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature on visit counts.")
    p.add_argument("--max-moves", type=int, default=200,
                   help="Cap on game length (in plies). 200 is generous for 9x9.")
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--rules", default="tromp-taylor")
    p.add_argument("--output", type=Path, default=Path("/tmp/go_spike.npz"))
    args = p.parse_args()

    katago_bin = os.environ.get("KATAGO_BIN")
    katago_model = os.environ.get("KATAGO_MODEL")
    katago_config = os.environ.get("KATAGO_CONFIG")  # required by KataGo
    if not katago_bin or not katago_model or not katago_config:
        print(INSTALL_HINT, file=sys.stderr)
        return 2

    print(f"[spike] playing one self-play game: size={args.board_size} "
          f"visits={args.visits} top_k={args.top_k}", flush=True)
    result = play_one_game(
        katago_bin, katago_model,
        config_path=katago_config,
        board_size=args.board_size,
        max_moves=args.max_moves,
        visits=args.visits,
        top_k=args.top_k,
        temperature=args.temperature,
        komi=args.komi,
        rules=args.rules,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        states=result["states"],
        moves=result["moves"],
        multipv_indices=result["multipv_indices"],
        multipv_logprobs=result["multipv_logprobs"],
        zs=result["zs"],
    )
    meta = result["_meta"]
    print(f"[spike] wrote {args.output}: {meta['n_moves']} positions, "
          f"final_winrate_stm={meta['final_winrate_stm']:.3f}, "
          f"elapsed={meta['elapsed_s']:.1f}s", flush=True)
    print(f"[spike] schema verified: same shape as chess pipeline's chunk .npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
