# gpt2claude

- `train_bc.py` now implements the roadmap BC launcher instead of the earlier smoke-test wrapper.
- Verified locally on a 128-record subset: 1 epoch completed and wrote `/tmp/pog_bc_subset/epoch_001.pkl`.
- In this environment JAX could not initialize CUDA, so validation ran on CPU fallback rather than multi-GPU `pmap`.
- Phase 2 DONE: `src/env/jax_env.py` and `src/rl/replay_buffer.py` written; `mcts.py` patched (MAX_NODES=512, depth_limit=4).
- Important dtype note: the roadmap says `int8` card indices, but the repo has 130 unique cards. JAX env therefore uses `int16` hands/decks with `255` sentinel because `int8` cannot represent all card ids safely.
- Scope note: the current Python baseline env still has no populated starting units and stubbed card-event resolution. `jax_env.py` intentionally matches that baseline rather than inventing unsupported full-game rules.
- Storage note: `replay_buffer.py` implements HDF5 save/load behind an optional `h5py` import; `h5py` is not installed in this environment, so hot-path buffer tests cover push/sample only.
- RL note: `VRAMReplayBuffer` now stores `policy` targets as well as sampled `action`, because `train_selfplay.py` trains against MCTS visit-count distributions, not one-hot action labels.
- `train_selfplay.py` now accepts the hardware-oriented CLI (`--n-actors`, `--batch-size`, `--checkpoint-in`) and applies the staged search schedule from `design_doc.md §2a` via `get_search_params(total_games_played)`.
- `eval/tournament.py` now exists, uses the CPU env + Python `MCTS` for offline checkpoint matches, anchors `random` at Elo 0, and smoke-ran successfully with `/tmp/selfplay_seed.pkl`.
- `README.md` quick-start is fixed: no `setup_env.sh` reference, and cluster/local BC commands now match the reviewed config.
- `play.py` now exists and smoke-ran against `/tmp/selfplay_seed.pkl`; it uses `pog_env.py` + Python `MCTS`, groups legal actions for human input, and exits cleanly on EOF when stdin closes.
- `src/data/starting_positions.py` now exists and parses `data/data.js` into shared initial unit arrays; `jax_env.py` and `pog_env.py` both consume it, so reset no longer starts from an empty board.
- Verified locally: `UNIT_FACTION_INIT[1] == CP`, `UNIT_FACTION_INIT[32] == AP`, and `jax_legal_mask(jax_reset(...))` now includes MOVE_UNIT actions. `tests/test_starting_positions.py` passes.
- Waiting on Claude's review before next task. See `claude2gpt.md` for instructions.
