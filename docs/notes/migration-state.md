# Migration state

`MIGRATION_PLAN.md` was substantially implemented before May 2026 —
`wm_chess/` is the chess_core, experiments depend on it via editable
paths. The plan itself is obsolete; cleanup is what remained, and is
now done too.

`MIGRATION_PLAN.md` proposes extracting a `chess_core/` package and
refactoring three chess experiments to use it. **This was already
done in a prior session.** As of May 2026:

- `wm_chess/src/wm_chess/` contains `board.py`, `network.py`, `mcts.py`,
  `arena.py`, `config.py` — the canonical shared core (the proposed
  `chess_core/`, just named `wm_chess`).
- `experiments/selfplay/` only contains `selfplay.py`, `replay.py`,
  `train.py` (experiment-specific).
- `experiments/distill-hard/` only contains `stockfish_data.py`,
  `train_supervised.py`.
- `experiments/distill-soft/` only contains `stockfish_data.py`,
  `train_supervised.py`.
- Each experiment's `pyproject.toml` has
  `[tool.uv.sources] wm-chess = { path = "../../wm_chess", editable = true }`.
- `library/` exists at root with the catalog system.

## What used to be unfinished — all now done

- ~~No top-level `pyproject.toml` / uv workspace — 5 separate
  `uv.lock` files.~~ Done: single root workspace + one `uv.lock`.
- ~~01-ha-world-models and 03-muzero-chess clutter the root.~~ Both
  removed.
- ~~`infra/` is deprecated (replaced by `infra-eks/`) but still
  referenced by 6 scripts.~~ Removed; references patched.
- ~~No `.env` / `.env.example` for the AWS values.~~ Both exist.
- ~~Top-level README still lists 01 and 03.~~ Updated.
- ~~Obsolete `MIGRATION_PLAN.md` itself.~~ Deleted.

**Why:** Reading `MIGRATION_PLAN.md` literally would have led to
redoing work that was done. The note exists so future-me checks the
actual repo state first.

**How to apply:** If someone references "the migration plan," it's
historical. `wm_chess/src/` *is* the migrated core; the cleanup phases
are documented in the closed task list and the May 2026 git log.
