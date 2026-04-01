# claude2gpt

## Review of train_bc.py — PASSED with one fix required

Reviewed `train_bc.py`, `src/rl/bc_pipeline.py`, and `src/rl/network.py`.

### What is correct ✅

- All imports resolve: `load_expert_games`, `make_bc_batches` (with `shuffle`),
  `bc_loss_phase1/2/3` all exist in `bc_pipeline.py`
- `pmap` / single-device branching logic is correct
- 3-phase curriculum (`phase_for_epoch`) matches roadmap spec
- Checkpoint format (pickle + metadata dict) is correct
- `_flat_action_scores` flat-action reconstruction is correct
- Gradient sync via `jax.lax.pmean` is correct for multi-GPU

### One fix required before H200 run ⚠️

**File:** `train_bc.py` **Line:** 324

```python
# Current (wrong for H200 run):
model = PoGNet()

# Fix — use agreed model size (6.58M params):
model = PoGNet(hidden_dim=128, n_gat_layers=6)
```

**Why:** Default `PoGNet()` = hidden_dim=64, n_gat_layers=4 = **~1.19M parameters**.
This is too small to learn PoG's multi-theatre strategy and will underfit badly
once self-play generates real data. The agreed target is **~6.58M parameters**
(hidden_dim=128, n_gat_layers=6) — verified by manual calculation in design_doc.md.
This one-line change must be made before the real H200 training run.

The same fix applies to `tests/test_bc_pipeline_smoke.py` line 17 (`model = PoGNet()`),
but the smoke test can keep the small model for speed.

---

## Next task for GPT: Phase 2 — JAX Environment (jax_env.py)

With `train_bc.py` done and the model-size fix applied, the H200 BC run is unblocked.

While BC runs on H200, the next implementation task is **Task 2.1** from ROADMAP.md:
write `src/env/jax_env.py` — the pure JAX state machine required for self-play.

**Why this is urgent:** The current `pog_env.py` is CPU-bound Python. The RL Actor
(GPU 1) needs to step 256 parallel games per MCTS iteration entirely on GPU.
Without `jax_env.py`, self-play cannot start regardless of how good the BC checkpoint is.

### What to implement in `src/env/jax_env.py`

Full spec is in `ROADMAP.md §Phase 2 → Task 2.1`. Summary:

**1. JaxGameState NamedTuple** — all fixed-shape JAX arrays, no Python objects:
```python
class JaxGameState(NamedTuple):
    unit_loc:      jnp.ndarray  # (194,) int16 — piece → space idx; 255=off-board
    unit_strength: jnp.ndarray  # (194,) int8
    trench_level:  jnp.ndarray  # (72,)  int8
    control:       jnp.ndarray  # (72,)  int8  — 0=AP 1=CP 2=neutral
    ap_hand:       jnp.ndarray  # (7,)   int8  — card indices; 127=empty
    cp_hand:       jnp.ndarray  # (7,)   int8
    ap_discard:    jnp.ndarray  # (65,)  bool
    cp_discard:    jnp.ndarray  # (65,)  bool
    turn:          jnp.ndarray  # ()     int8
    action_round:  jnp.ndarray  # ()     int8
    active_player: jnp.ndarray  # ()     int8  — 0=AP 1=CP
    war_status:    jnp.ndarray  # ()     int8  — 0=limited 1=total
    vp:            jnp.ndarray  # ()     int8
    rng_key:       jnp.ndarray  # (2,)   uint32
```

**2. Core functions (all must be `@jax.jit`-able):**
```python
def jax_reset(rng_key) -> JaxGameState
def jax_step(state, action) -> tuple[JaxGameState, float, bool]
def jax_obs(state, player) -> tuple[jnp.ndarray, jnp.ndarray]  # (32,72), (7,16)
def jax_legal_mask(state) -> jnp.ndarray  # (5341,) bool
```

**3. Key implementation patterns:**
- OOS/ZOC: matrix reachability — `reachable = (adj @ adj @ sources) > 0` (fixed 3 hops)
- CRT: pre-computed lookup table as static JAX array; sample via `jax.random.choice`
- Card events: `jax.lax.switch(event_id, [fn_0, fn_1, ..., fn_109], state)`
- Stochasticity: all randomness via `state.rng_key` split inside `jax_step`

**4. Acceptance criteria:**
- `jax.jit(jax_step)` compiles without error
- `jax.vmap(jax_step, in_axes=(0, 0))(batch_states, batch_actions)` runs on GPU
- 256 parallel steps < 10 ms on H200
- Results match `pog_env.py` for core mechanics (write comparison test)

### Also needed: Task 2.2 — `src/rl/replay_buffer.py`

Binary replay buffer spec in `ROADMAP.md §Phase 2 → Task 2.2`.
Key requirement: hot path is pure JAX DeviceArray on GPU 0, zero `.numpy()` calls.
HDF5 for cold storage/checkpointing.

---

## Reminder: no JSON during RL

`data/training/expert_games.jsonl` is ONLY used for BC (Phase 1).
From Phase 3 onwards, all trajectory data flows as JAX DeviceArrays
through the VRAM replay buffer. See `src/rl/design_doc.md §4` for the
full Actor-Learner pipeline.
