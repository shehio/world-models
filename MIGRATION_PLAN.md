# Migration plan — unify 02/02b/02c + build a shared game library

> Status: **proposal**, not yet implemented. Sign-off needed before any
> file moves. Designed as one PR per phase so each is reviewable in
> isolation and reversible with `git revert`.

## Why now

The repo has three sibling chess projects (02 self-play, 02b distillation
with hard targets, 02c distillation with multipv soft targets) that
share five files almost verbatim:

```
                  02         02b        02c
board.py        225 ln    146 ln    146 ln    (02b ≡ 02c; 02 has extras)
network.py       73 ln     73 ln     73 ln    (02 drift vs 02b/02c)
mcts.py         445 ln    397 ln    397 ln    (02 has batched MCTS + virtual loss)
arena.py        171 ln    141 ln    141 ln    (02 has extras)
config.py        37 ln     37 ln     37 ln    (all three identical)
─────────────────────────────────────────────
                951 ln    794 ln    794 ln    = ~2.5k lines of duplication
```

**The drift is already biting us.** Examples I've already hit this
session:

- 02's selfplay loop checkpoints `{net, opt, iter, hour}` (full optimizer
  state). 02c's training only checkpoints `state_dict()` — Adam moments
  are lost on resume. The fix in 02 (commit `acd5dba`) never propagated.
- 02c built `legal_move_mask()` in `board.py`; 02 had its own version.
- The chunked-save crash safety I just added to 02c lives only in 02c.
  02 and 02b will lose work the same way 02c did before.

Three copies = three places to fix every bug, three places to land every
improvement, three places where the answer to "which one is canonical?"
is "whichever you grep first."

## Goals

1. **One canonical core**: `chess_core` package owning `board`, `network`,
   `mcts`, `arena`, `config`. All three experiments import from it.
2. **Experiments remain distinct stories.** Each keeps its `results.md`,
   its `scripts/`, its training loop. The narrative *of three different
   training procedures on one architecture* is the point of this repo.
3. **One game library, not three `data/` folders.** All experiments
   read from `library/games/` so a dataset generated once can train
   multiple experiments.
4. **Overnight-runnable data generation.** A multi-config orchestrator
   that drives N data-gen configs sequentially on AWS spot, writes to
   the indexed library, regenerates the catalog, destroys when done.

## Target layout

```
world-models/
├── chess_core/                              ← NEW: shared chess library
│   ├── pyproject.toml                       (uv package; deps: numpy, torch, python-chess)
│   ├── src/chess_core/
│   │   ├── __init__.py
│   │   ├── board.py                         (canonical 19-plane / 4672-move encoder)
│   │   ├── network.py                       (canonical ResNet, parameterized by Config)
│   │   ├── mcts.py                          (canonical batched PUCT MCTS w/ virtual loss)
│   │   ├── arena.py                         (canonical vs-engine match player)
│   │   └── config.py                        (canonical frozen dataclass)
│   └── tests/
│       ├── test_board.py
│       ├── test_network.py
│       ├── test_mcts.py
│       └── test_arena.py
│
├── experiments/                             ← three thin experiments
│   ├── 02-selfplay/                         (was 02-alphazero-chess/)
│   │   ├── pyproject.toml                   (depends on chess-core, editable)
│   │   ├── src/selfplay/
│   │   │   ├── selfplay_loop_mp.py
│   │   │   ├── replay_buffer.py
│   │   │   ├── train_step.py
│   │   │   └── ...
│   │   ├── scripts/                         (run_overnight.sh, eval.py, …)
│   │   ├── results.md                       (v1 → v5 story, unchanged)
│   │   └── tests/
│   │
│   ├── 02b-distill-hard/                    (was 02b-alphazero-stockfish-distill/)
│   │   ├── pyproject.toml
│   │   ├── src/distill_hard/
│   │   │   ├── stockfish_data.py            (hard-target only; thin wrapper)
│   │   │   └── train_supervised.py          (one-hot CE)
│   │   ├── scripts/
│   │   ├── results.md                       (1185 Elo writeup)
│   │   └── tests/
│   │
│   └── 02c-distill-soft/                    (was 02c-distill-scaled/)
│       ├── pyproject.toml
│       ├── src/distill_soft/
│       │   ├── stockfish_data.py            (multipv + library + chunked save)
│       │   └── train_supervised.py          (soft-CE + hard-CE flag)
│       ├── scripts/
│       ├── results.md                       (1086 Elo writeup)
│       └── tests/
│
├── 01-ha-world-models/                      (unchanged — unrelated, CarRacing)
├── library/                                 ← NEW: shared game library (cross-experiment)
│   ├── games/
│   │   └── sf-<v>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/
│   │       ├── data.npz
│   │       ├── games.pgn
│   │       ├── metadata.json
│   │       └── chunks/
│   └── CATALOG.md                           (auto-generated index of available datasets)
│
├── checkpoints/                             ← NEW: matching ckpt index (symmetric to library/)
│   └── sf-<v>/d<D>-mpv<K>-T<T>/g<N>-seed<S>/net-<R>x<F>/<run-id>/
│
├── infra/                                   (unchanged)
├── DISTILLATION_VS_ALPHAZERO.md             (unchanged content; paths in code blocks fixed)
├── MIGRATION_PLAN.md                        (this file)
├── README.md                                (top-level; nav updated)
└── dashboard.html
```

**Naming choice.** `chess_core` (snake_case Python package) is the
import name; the directory is `chess_core/` (matches). I considered
just `chess/` but that collides with the popular `python-chess`
package (which is imported as `chess`). Better to avoid the shadowing.

## Phased migration

Six phases. Each is one PR, mergeable independently. Order matters only
within each phase block.

### Phase 0 — divergence audit (no code changes)

**Goal:** know exactly what differs between the three copies before
unifying, so we don't silently lose a fix.

1. `git diff --no-index 02-alphazero-chess/src/azchess/board.py 02c-distill-scaled/src/azdistill_scaled/board.py > audit/board.diff`
2. Same for `network.py`, `mcts.py`, `arena.py`. (`config.py` is byte-identical.)
3. For each non-identical pair, classify each hunk as:
   - **canonical** = belongs in `chess_core` (a real improvement)
   - **experiment-specific** = needs to stay in the experiment (e.g., 02's
     batched MCTS used at self-play time but never by 02b/02c eval)
   - **drift** = no good reason for the difference (pick one, document)
4. Output: `audit/DIVERGENCE.md` — one section per file with the
   table of decisions. **This is the deliverable of Phase 0.**

Without this, Phase 1 is guessing.

### Phase 1 — create `chess_core/` package

**Goal:** the new shared package exists, has all the canonical code,
and its own tests pass. No experiment imports it yet.

1. Create `chess_core/pyproject.toml`:
   ```toml
   [project]
   name = "chess-core"
   version = "0.1.0"
   requires-python = ">=3.10"
   dependencies = ["numpy>=1.24", "torch>=2.0", "python-chess>=1.999"]

   [build-system]
   requires = ["setuptools>=61"]
   build-backend = "setuptools.build_meta"
   ```
2. Create `chess_core/src/chess_core/` with the **canonical** version
   of each file (per Phase 0's decisions). Default canonical: 02's
   versions (they're supersets — batched MCTS, virtual-loss, helpers
   in board.py).
3. Move tests for shared code from `02-alphazero-chess/tests/` to
   `chess_core/tests/`. Specifically: `test_board.py`, `test_mcts.py`,
   `test_mcts_batched.py`, `test_arena.py`, `test_pcr.py`,
   `test_network.py`. (Keep tests for 02-specific things — selfplay
   loop, replay buffer — in 02.)
4. Run `cd chess_core && uv run python -m pytest tests/`. **Gate: all
   pass.**

### Phase 2 — refactor 02-alphazero-chess to use `chess_core`

02 has the canonical implementations, so this should be the cheapest
migration (mostly delete + change imports).

1. `experiments/02-selfplay/pyproject.toml` declares the dep:
   ```toml
   dependencies = ["chess-core @ file://${PROJECT_ROOT}/chess_core"]
   ```
   (Or `[tool.uv.sources] chess-core = { path = "../../chess_core", editable = true }` — uv-native syntax preferred.)
2. Delete `src/azchess/board.py`, `network.py`, `mcts.py`, `arena.py`, `config.py`.
3. In `src/azchess/`, find all `from .board import ...` / `from .network import ...` etc. → replace with `from chess_core.board import ...`.
4. Update `scripts/*.py` and `tests/*.py` likewise.
5. `mv 02-alphazero-chess experiments/02-selfplay` (path move).
6. `cd experiments/02-selfplay && uv sync && uv run python -m pytest tests/`. **Gate: all 42 - moved-out tests pass.**
7. Run a 30-second smoke test of `scripts/selfplay_loop_mp.py` with a
   small iter count, verify it doesn't crash on `import`.

### Phase 3 — refactor 02b-distill to use `chess_core`

Same pattern. 02b's copies of shared files are identical to 02c's, so
the diff vs the new `chess_core` (which started from 02's superset)
needs reconciliation: if 02 had a feature 02b doesn't, that's fine
(02b just doesn't use it). If 02b had a feature 02 doesn't… that
would have surfaced in Phase 0.

1. Delete duplicate files from `src/azdistill/`.
2. Update imports.
3. `mv 02b-alphazero-stockfish-distill experiments/02b-distill-hard`.
4. `uv sync && pytest`. **Gate: all 6 tests pass.**
5. Smoke-test: load an existing 02b checkpoint, run a single eval game,
   verify the model loads and plays.

### Phase 4 — refactor 02c-distill-scaled to use `chess_core`

Identical to Phase 3 mechanically. Importantly: keep `stockfish_data.py`
and `train_supervised.py` in the experiment package — they're not
shared.

1-4. Same steps.

5. **Critical check**: the 02c-30ep checkpoint at
   `aws_results/run02_30ep/distilled_epoch029.pt` must still load.
   `state_dict()` is module-path-agnostic; verify with a load test
   in `tests/test_compat_old_ckpts.py`.

### Phase 5 — shared `library/games/` + catalog

This is the second half of what you asked for: making data generation
overnight-runnable.

1. **Move 02c's library** from `02c-distill-scaled/data/library/` to
   `library/games/`. Add an `__OLD_LOCATION` symlink for safety during
   transition.
2. **All three experiments learn `--library-root`.** Today only 02c
   has it. Plumb it through 02 (replay buffer can read prior datasets
   for warm-start) and 02b (no library awareness today).
3. **Catalog generator** at `chess_core/scripts/catalog.py`:
   - walks `library/games/`
   - for each leaf dir, reads `metadata.json`
   - emits `library/CATALOG.md` with a table:

     ```
     | sf  | depth | mpv | T   | games | seed | positions | size | path |
     |-----|------:|----:|----:|------:|-----:|----------:|-----:|------|
     | 18  | 10    | 8   | 1.0 | 4000  | 42   | 243k      | 1.5G | sf-18/d10-mpv8-T1/g4000-seed42 |
     | 18  | 15    | 8   | 1.0 | 4000  | 42   | (in-prog) | 0.4G | sf-18/d15-mpv8-T1/g4000-seed42 |
     ```
   - emits `library/catalog.json` for programmatic access
   - exit code 0 even if some chunks are incomplete (logs warnings)
4. **Pre-commit hook (optional):** regenerate `CATALOG.md` if any
   `metadata.json` changed. Otherwise: regenerated by each datagen
   pipeline run.
5. **Train scripts gain `--from-library`:**
   ```bash
   uv run python scripts/train.py \
     --from-library sf-18/d10-mpv8-T1/g4000-seed42 \
     --epochs 30 ...
   ```
   resolves to `library/games/sf-18/d10-mpv8-T1/g4000-seed42/data.npz`,
   reads `metadata.json` to populate sf-version + mpv + T into the
   checkpoint path under `checkpoints/`.

### Phase 6 — overnight multi-config datagen orchestrator

This is the runbook for unattended overnight runs.

1. **New script:** `chess_core/scripts/overnight_datagen.sh`:
   ```bash
   KEY_NAME=mykey CONFIGS=configs/overnight_1.yaml \
     bash chess_core/scripts/overnight_datagen.sh
   ```
2. **Config file format** (`configs/overnight_1.yaml`):
   ```yaml
   instance_type: c7i.8xlarge
   region: us-east-1
   use_spot: true
   configs:
     - { depth: 10, multipv: 8, temperature: 1.0, n_games: 4000, seed: 42 }
     - { depth: 15, multipv: 8, temperature: 1.0, n_games: 4000, seed: 42 }
     - { depth: 10, multipv: 8, temperature: 0.3, n_games: 4000, seed: 42 }
   ```
3. **Behavior:** for each config in sequence:
   - skip if `library/games/<resolved_path>/data.npz` already exists
     (idempotent)
   - terraform apply CPU box → rsync project → run gen with
     `--library-root` → rsync library tree back → terraform destroy
   - regenerate `library/CATALOG.md`
   - log to `library/runs/<timestamp>/`
4. **Failure recovery.** If a config's run is killed (spot reclaim),
   `finalize_library_path()` salvages chunks; the partial dataset
   becomes a "partial" entry in CATALOG.md. Re-running the script
   with the same config completes it (workers detect existing chunks
   — *this needs implementing*; today re-runs start from chunk_idx=0
   per worker, which would duplicate data. **Tracked as work item in
   this phase.**)

### Cost estimate for overnight datagen

Three new datasets (the experiments queued in 02c results.md):

| Config | Purpose | Wall (c7i.8xlarge) | Spot cost |
|---|---|---:|---:|
| d10 mpv8 T=1.0 g4000 | regenerate baseline (was 02c's) | ~30 min | $0.17 |
| d15 mpv8 T=1.0 g4000 | depth-15 teacher (Exp C) | ~3 h | $1.00 |
| d10 mpv8 T=0.3 g4000 | sharper softmax (Exp B) | ~30 min | $0.17 |
| Bootstrap × 3 | | 9 min | $0.05 |
| **Total** | | **~4.5 h** | **~$1.39** |

Fits in one overnight slot (10pm → 7am has slack for failures and
re-runs). Cheap enough that re-running is no big deal.

## Test impact

- **02-alphazero-chess**: 42 tests today. After refactor: ~24 stay in
  experiments/02-selfplay (selfplay-loop + replay-buffer specific), ~18
  migrate to `chess_core/tests/`.
- **02b-alphazero-stockfish-distill**: 6 tests today. All stay in
  experiments/02b-distill-hard.
- **02c-distill-scaled**: 63 tests today. ~30 stay in
  experiments/02c-distill-soft (stockfish_data, train_supervised). ~33
  migrate to `chess_core/tests/` (board, network).
- **chess_core/tests/**: ~50 new shared tests after migration.

Gate at each phase: `pytest` in the affected package must be green.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Old checkpoints fail to load due to architecture drift | Medium | Phase 0 audit + `test_compat_old_ckpts.py` smoke tests in each phase |
| 02's batched MCTS has hidden assumptions 02b/02c eval breaks on | Low | 02b/02c eval uses plain MCTS code path; batched-MCTS is opt-in via flag |
| `uv sync` workspace setup with editable local deps doesn't work | Low | uv supports `tool.uv.sources` for this; documented and tested |
| Mid-migration the repo is broken | High during refactor | Each phase is one PR; revert restores previous state |
| Library path change breaks `aws_results/run02_30ep/` references | Low | That dir is preserved as-is; old runs are read-only history |
| Catalog regeneration is slow at large library size | Low (today: 1 entry) | Only re-reads `metadata.json` per leaf; O(N_datasets) |

## What I am NOT proposing (and why)

- **Don't merge experiments into one mega-CLI.** The point of having
  three separate experiments is the per-experiment story
  (results.md, scripts, CLI). Mashing them into one CLI would obscure
  the narrative.
- **Don't move 01-ha-world-models into `experiments/`.** It's not chess,
  shares no code with the others, and renumbering it would break the
  conceptual "01 / 02 / 03" arc the top-level README narrates.
- **Don't introduce S3 for the library yet.** The current laptop-as-
  intermediary works at this scale (~few GB datasets). S3 makes sense
  if multiple machines need to read the library concurrently; one
  engineer + one laptop doesn't need it.
- **Don't auto-rename existing checkpoints into the new index scheme.**
  Phase 5 documents the *new* scheme; old `aws_results/` and
  `checkpoints_v*/` directories stay where they are with a README
  pointer. Migrating historical artifacts is busywork without payoff.
- **Don't gate Phase 6 (overnight datagen) on Phases 1–5.** The
  overnight orchestrator can be built today against the existing
  02c library layout. It's listed under Phase 6 only for natural
  ordering; if you want to start there, we can.

## Suggested order if we want to ship fast

If the goal is "have an overnight library run going as soon as possible,
with refactor as a follow-up":

1. **Phase 6 first** — build `overnight_datagen.sh` against today's
   02c-rooted library path. Get a useful dataset growing tonight.
2. **Phase 0** — divergence audit.
3. **Phase 1** — `chess_core/` extraction.
4. Phases 2 / 3 / 4 in any order.
5. **Phase 5** — move library to `library/games/` (one path rename in
   the overnight script; cheap once Phase 1 is done).

That sequence keeps the cost-benefit positive at every step: the night
after Phase 6 lands, you wake up with three new datasets ready for
Experiment A / B / C ablations.

## Decision points for you

Before I touch anything, four things to confirm:

1. **Package name** `chess_core` ok, or do you prefer something else
   (e.g., `azchess_core`, `wm_chess`)?
2. **Top-level layout**: `experiments/02-selfplay/` vs keeping the
   experiments flat at repo root (`02-selfplay/`)? Flat keeps the
   conceptual "01 / 02 / 03" numbering legible from `ls`; nested is
   tidier.
3. **Ship-fast order or by-phase order?** I'd recommend ship-fast (start
   with Phase 6 datagen so we earn data while we refactor).
4. **Single-PR or multi-PR refactor?** I'd recommend multi-PR (one per
   phase) so each is small and revertible.

Once you sign off on those four, the rest is mechanical.
