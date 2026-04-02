# claude2gpt

---

## Task 8.1 — Multi-GPU self-play via pmap  ← DO NOW

### What happened

On the cluster (3× H200, 143 GB each), running:
```bash
export CUDA_VISIBLE_DEVICES=2
python train_selfplay.py --bc-checkpoint checkpoints/bc/epoch_040.pkl \
  --n-actors 256 --buffer-capacity 500000 --batch-size 2048 --iterations 500
```
crashed immediately with:
```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 86.89GiB.
```

**Root cause:** `jax.vmap` over 256 actors wraps `jax_mcts_search`, which internally contains `jax.lax.fori_loop(0, n_simulations, ...)`. XLA vectorises the entire fori_loop body over all 256 actors simultaneously, materialising 256 × 256 sims × 5 rollout steps × GATv2 forward passes in one shot → ~87 GB on a single device.

Single H200 (143 GB) is just enough if you halve actors to 128 (≈43 GB). But the right fix is **pmap** across 3 GPUs so each device handles 64 actors (≈22 GB).

### Goal

Modify **`train_selfplay.py` only** to use `pmap(vmap(...))` when `n_devices > 1`, and keep `vmap(...)` when `n_devices == 1`. The `n_devices == 1` path is testable locally (RTX 5060, which has only one GPU). The `n_devices > 1` path follows standard JAX `pmap` semantics and is activated on the cluster.

### Testing strategy (local machine has 1 GPU — cannot test pmap path directly)

1. Run the existing smoke test — it must still pass on the `n_devices == 1` path:
   ```bash
   python train_selfplay.py \
     --n-actors 2 --buffer-capacity 512 --min-buffer-size 16 \
     --batch-size 16 --mcts-sims 2 --depth-limit 1 --learner-steps 1 \
     --max-steps 8 --iterations 1 \
     --checkpoint-dir /tmp/pog_selfplay_smoke_fast
   ```
   Expected: finite losses, checkpoint written (same as before).

2. Verify `pytest -q tests/test_selfplay_smoke.py` still passes (72 total tests).

3. The pmap path is validated by JAX API correctness on the cluster; no local multi-GPU test needed.

### Precise code changes to `train_selfplay.py`

**Change 1 — compute `n_per_device` after `n_actors`**

In `main()`, after the line `n_actors = args.n_actors or args.num_envs or (256 if multi_gpu else 16)`:

```python
# NEW: actors-per-device for pmap sharding
n_per_device = n_actors // n_local_devices
if n_actors % n_local_devices != 0:
    # Round up to nearest multiple
    n_actors = (n_actors // n_local_devices + 1) * n_local_devices
    n_per_device = n_actors // n_local_devices
    print(f"Rounded n_actors to {n_actors} (divisible by {n_local_devices} devices)")
```

**Change 2 — build sharded env functions based on n_local_devices**

Replace the three lines:
```python
batched_mask = jax.jit(jax.vmap(jax_legal_mask))
batched_step = jax.jit(jax.vmap(jax_step))
batched_obs = jax.jit(jax.vmap(jax_obs, in_axes=(0, 0)))
```
With:
```python
if n_local_devices > 1:
    # pmap over devices, vmap over actors-per-device
    # Input shape: (n_devices, n_per_device, ...)
    batched_mask = jax.pmap(jax.vmap(jax_legal_mask))
    batched_step = jax.pmap(jax.vmap(jax_step))
    batched_obs  = jax.pmap(jax.vmap(jax_obs, in_axes=(0, 0)))
else:
    batched_mask = jax.jit(jax.vmap(jax_legal_mask))
    batched_step = jax.jit(jax.vmap(jax_step))
    batched_obs  = jax.jit(jax.vmap(jax_obs, in_axes=(0, 0)))
```

**Change 3 — replicate params for actor use**

After `actor_params = learner_state.params`, add:
```python
if n_local_devices > 1:
    actor_params_rep = jax.device_put_replicated(actor_params, jax.local_devices())
else:
    actor_params_rep = actor_params
```

**Change 4 — build batched_search inside the iteration loop**

The `search_fn` is rebuilt each iteration (because `search_params` can change). Replace:
```python
search_fn = functools.partial(
    jax_mcts_search,
    params=actor_params,
    ...
)
batched_search = jax.jit(jax.vmap(search_fn))
```
With:
```python
search_fn = functools.partial(
    jax_mcts_search,
    params=actor_params,   # unreplicated — pmap broadcasts closed-over scalars/arrays
    ...
)
if n_local_devices > 1:
    batched_search = jax.pmap(jax.vmap(search_fn))
else:
    batched_search = jax.jit(jax.vmap(search_fn))
```

Note: `actor_params` here is the **unreplicated** params (from `learner_state.params`). JAX `pmap` will automatically broadcast closed-over non-mapped arrays to all devices. Do **not** pass `actor_params_rep` into `partial` — that would cause shape errors.

**Change 5 — reset states with shaped keys**

Replace:
```python
states = jax.vmap(jax_reset)(jax.random.split(reset_key, n_actors))
```
With:
```python
all_reset_keys = jax.random.split(reset_key, n_actors)
if n_local_devices > 1:
    shaped_keys = all_reset_keys.reshape(n_local_devices, n_per_device, 2)
    states = jax.pmap(jax.vmap(jax_reset))(shaped_keys)
else:
    states = jax.vmap(jax_reset)(all_reset_keys)
```

**Change 6 — add a `_flat(x)` helper and use it in the actor loop**

Add this helper right before the `while` loop (inside the `for iteration` block):
```python
def _flat(x):
    """Flatten (n_devices, n_per_device, ...) -> (n_actors, ...) when using pmap."""
    if n_local_devices > 1:
        return x.reshape(n_actors, *x.shape[2:])
    return x
```

**Change 7 — fix the actor loop inner body**

Replace the observation / mask / policy / action lines inside `while`:
```python
active_players = states.active_player
obs_batch, card_ctx_batch = batched_obs(states, active_players)
mask_batch = batched_mask(states)
policy_batch = batched_search(states)

sample_key, action_key = jax.random.split(sample_key)
action_keys = jax.random.split(action_key, n_actors)
tempered = jnp.power(policy_batch, 1.0 / jnp.maximum(temperature, 1e-6))
tempered = tempered / jnp.maximum(jnp.sum(tempered, axis=-1, keepdims=True), 1e-8)
action_batch = jax.vmap(lambda p, k: jax.random.choice(k, N_ACTIONS, p=p))(tempered, action_keys)
next_states, reward_batch, done_batch = batched_step(states, action_batch)

obs_np = np.asarray(jax.device_get(obs_batch))
ctx_np = np.asarray(jax.device_get(card_ctx_batch))
mask_np = np.asarray(jax.device_get(mask_batch))
policy_np = np.asarray(jax.device_get(policy_batch))
action_np = np.asarray(jax.device_get(action_batch))
next_states_host = jax.device_get(next_states)
done_np = np.asarray(jax.device_get(done_batch))
```
With:
```python
active_players = states.active_player
obs_batch, card_ctx_batch = batched_obs(states, active_players)
mask_batch = batched_mask(states)
policy_batch = batched_search(states)

# Flatten from (n_devices, n_per_device, ...) -> (n_actors, ...) for sampling
policy_flat = _flat(policy_batch)
sample_key, action_key = jax.random.split(sample_key)
action_keys = jax.random.split(action_key, n_actors)
tempered = jnp.power(policy_flat, 1.0 / jnp.maximum(temperature, 1e-6))
tempered = tempered / jnp.maximum(jnp.sum(tempered, axis=-1, keepdims=True), 1e-8)
action_flat = jax.vmap(lambda p, k: jax.random.choice(k, N_ACTIONS, p=p))(tempered, action_keys)

# Reshape actions back to (n_devices, n_per_device) for batched_step
if n_local_devices > 1:
    action_batch = action_flat.reshape(n_local_devices, n_per_device)
else:
    action_batch = action_flat

next_states, reward_batch, done_batch = batched_step(states, action_batch)

obs_np = np.asarray(jax.device_get(_flat(obs_batch)))
ctx_np = np.asarray(jax.device_get(_flat(card_ctx_batch)))
mask_np = np.asarray(jax.device_get(_flat(mask_batch)))
policy_np = np.asarray(jax.device_get(policy_flat))
action_np = np.asarray(jax.device_get(action_flat))
next_states_host = jax.device_get(jax.tree_util.tree_map(_flat, next_states))
done_np = np.asarray(jax.device_get(_flat(done_batch)))
```

The `for i in range(n_actors)` slot loop that follows is **unchanged** — it already indexes into (n_actors, ...) arrays.

**Change 8 — fix `states = next_states` at the end of the while loop**

The states variable stays in shaped form `(n_devices, n_per_device, ...)` for the next pmap call. No change needed — `next_states` was never flattened, only `next_states_host` was.

**Change 9 — sync actor params after learner update**

Replace:
```python
if (not multi_gpu) or (learner_updates % args.sync_every == 0):
    actor_params = learner_state.params
```
With:
```python
if n_local_devices == 1 or (learner_updates % args.sync_every == 0):
    actor_params = learner_state.params
    # Update search_fn binding — handled by rebuilding search_fn each iteration
```

Note: `search_fn` is rebuilt at the top of each `for iteration` block using `actor_params`, so the sync automatically takes effect next iteration. No additional change needed.

### Cluster run command (after code is pushed)

```bash
# On cluster: use GPUs 1, 2, 3 (GPU 0 is occupied — 108 GB used)
export CUDA_VISIBLE_DEVICES=1,2,3
nohup python train_selfplay.py \
  --bc-checkpoint checkpoints/bc/epoch_040.pkl \
  --n-actors 192 \
  --buffer-capacity 500000 \
  --batch-size 2048 \
  --iterations 500 \
  > selfplay.log 2>&1 &
echo $!
```

`n_actors=192` → 64 actors per H200 → ~22 GB per device. Well within 143 GB each.

Monitor: `tail -f selfplay.log`

### Acceptance criteria

- `python -m py_compile train_selfplay.py` passes ✓
- Existing smoke test passes on local 1-GPU machine ✓
- 72 existing tests pass ✓
- `jax.local_devices()` prints 3 devices on cluster, `n_per_device=64` ✓
- No OOM on cluster ✓

**Do not touch** anything outside `train_selfplay.py` (no new files, no test changes).

---

## Review of Tasks 7.6 + 7.7 — ALL PASSED ✅

**7.6 — pog_env.py alignment:**
- `_ap_played` / `_cp_played` lists track OPS-played cards; reshuffled back into deck at turn wrap ✓
- `_play_event` respects `remove_after_event`; permanent discards to `_ap_permanent_discard` set ✓
- `_play_event` calls `_bring_unit_on_map` for reinforcement-named cards ✓
- `_play_ops(op_type=2)` calls `_bring_unit_on_map` for SR ✓
- `tests/test_pog_env_alignment.py` passes ✓

**7.7 — Self-play smoke test:**
- `train_selfplay.py` cold-start (no BC checkpoint) runs 1 iteration cleanly ✓
- Finite losses confirmed: `loss=5.3199`, `policy=0.6384`, `value=2.9483` ✓
- Checkpoint written to `/tmp/pog_selfplay_smoke_fast/iter_001.pkl` ✓

**70/70 tests pass** ✓

---

## Task 7.8 — CARD_VP_DELTA: apply VP adjustments from event cards  ← DO NOW

This is the **last jax_env.py task** before the project waits for BC results.
After this task the simulation is engineering-complete.

### Problem

One class of events — cards that give an immediate VP adjustment — is still a
no-op. Only one card is a clear, unconditional, one-time VP adjustment:

| Card | Faction | Event text (simplified) | Our delta |
|---|---|---|---|
| Reichstag Truce | CP | "Add 1 VP" | **−1** (CP winning = negative in our convention) |

All other VP-related cards have per-turn accumulation, prerequisites, or side-
effects that make them too complex for this phase. Delta = 0 for them.

No `JaxGameState` schema change is needed. Apply delta directly to `state.vp`
inside `do_event`; it persists until the next territorial `_recompute_vp` call.

### Step 1 — Add `card_vp_delta` to `_load_static_tables()`

In `_load_static_tables()`, add alongside the other `card_*` initialisations
(near line 128):

```python
card_vp_delta = []
```

Inside the `if sid not in seen_cards:` block (near line 142), after the
`card_reinf_count.append(...)` line:

```python
name_lower = entry["name"].lower()
# Only Reichstag Truce has a clear, unconditional, one-time VP adjustment.
# CP card "Add 1 VP" → positive VP goes to CP → delta = -1 in our convention
# (our convention: positive vp = AP winning, negative = CP winning)
if "reichstag truce" in name_lower:
    card_vp_delta.append(-1)
else:
    card_vp_delta.append(0)
```

Add to the `return` dict:

```python
"card_vp_delta": np.array(card_vp_delta, dtype=np.int8),
```

### Step 2 — Add module-level constant

After `CARD_REINF_COUNT` (around line 203):

```python
CARD_VP_DELTA = jnp.asarray(_STATIC["card_vp_delta"], dtype=jnp.int8)
```

### Step 3 — Apply delta in `do_event`

Current `do_event` (lines 641–653):

```python
def do_event(s):
    card_idx = (action - ACT_EVENT_START).astype(jnp.int16)
    clamped = jnp.clip(card_idx, 0, CARD_REMOVE_AFTER_EVENT.shape[0] - 1)
    permanent = CARD_REMOVE_AFTER_EVENT[clamped]
    s = _remove_active_card(s, card_idx, permanent)
    n_reinf = CARD_REINF_COUNT[clamped].astype(jnp.int8)
    s = jax.lax.cond(
        n_reinf > 0,
        lambda st: _bring_units_on_map(st, st.active_player, n_reinf),
        lambda st: st,
        s,
    )
    return s
```

Add the VP delta after the reinforcement block:

```python
def do_event(s):
    card_idx = (action - ACT_EVENT_START).astype(jnp.int16)
    clamped = jnp.clip(card_idx, 0, CARD_REMOVE_AFTER_EVENT.shape[0] - 1)
    permanent = CARD_REMOVE_AFTER_EVENT[clamped]
    s = _remove_active_card(s, card_idx, permanent)
    n_reinf = CARD_REINF_COUNT[clamped].astype(jnp.int8)
    s = jax.lax.cond(
        n_reinf > 0,
        lambda st: _bring_units_on_map(st, st.active_player, n_reinf),
        lambda st: st,
        s,
    )
    # VP delta (one-time; persists until next territorial _recompute_vp)
    delta = CARD_VP_DELTA[clamped].astype(jnp.int8)
    s = s._replace(vp=jnp.clip(s.vp + delta, -127, 127).astype(jnp.int8))
    return s
```

### Step 4 — Test: `tests/test_card_vp_delta.py`

```python
import jax, jax.numpy as jnp
from src.env.jax_env import (
    ACT_EVENT_START, CARD_FACTION, CARD_VP_DELTA,
    FACTION_CP, jax_reset, jax_step,
)


def _find_reichstag() -> int:
    """Return the global card index for Reichstag Truce."""
    for i in range(CARD_VP_DELTA.shape[0]):
        if int(CARD_VP_DELTA[i]) == -1:
            return i
    raise RuntimeError("Reichstag Truce not found in CARD_VP_DELTA")


def test_reichstag_truce_decrements_vp():
    card_idx = _find_reichstag()
    assert int(CARD_FACTION[card_idx]) == FACTION_CP, "Reichstag Truce must be CP"

    state = jax_reset(jax.random.PRNGKey(0))
    vp_before = int(state.vp)
    state = state._replace(
        active_player=jnp.asarray(FACTION_CP, dtype=jnp.int8),
        cp_hand=jnp.asarray([card_idx, 255, 255, 255, 255, 255, 255], dtype=jnp.int16),
    )
    action = jnp.asarray(ACT_EVENT_START + card_idx, dtype=jnp.int32)
    new_state, _, _ = jax_step(state, action)
    assert int(new_state.vp) == vp_before - 1, (
        f"Reichstag Truce should decrement vp by 1: before={vp_before}, after={int(new_state.vp)}"
    )


def test_card_vp_delta_array_has_expected_nonzero():
    nonzero = int(jnp.sum(CARD_VP_DELTA != 0))
    assert nonzero >= 1, "At least Reichstag Truce should have a nonzero delta"
```

### Acceptance criteria

- `tests/test_card_vp_delta.py` (2 tests) passes ✓
- All 70 existing tests pass ✓
- `jax.jit(jax_step)` compiles ✓
- `CARD_VP_DELTA` is all-zeros except Reichstag Truce (delta=−1) ✓

**Do not touch** anything outside `src/env/jax_env.py` and the new test file.

---

## After Task 7.8: project is engineering-complete

The simulation is done. The pipeline is verified. What's left is data and training:

### Waiting on: BC checkpoint from cluster

When `checkpoints/bc/epoch_010.pkl` arrives:

```bash
# On cluster (256 actors, H200):
python train_selfplay.py \
  --bc-checkpoint checkpoints/bc/epoch_010.pkl \
  --n-actors 256 --buffer-capacity 500000 \
  --batch-size 2048 --iterations 500

# On local RTX 5060 (16 actors):
python train_selfplay.py \
  --bc-checkpoint checkpoints/bc/epoch_010.pkl \
  --n-actors 16 --buffer-capacity 50000 \
  --batch-size 256 --iterations 100
```

### Waiting on: more training data (user action required)

```bash
# Scrape more expert games from RTT server:
python scrape_rtt_expert.py
# Then parse into training records:
python src/data/rtt_parser.py data/rtt_games/ --output data/training/expert_games.jsonl
# Target: 2000+ records (currently ~195 — BC will overfit badly at 195)
```

### After self-play training: evaluate

```bash
python eval/tournament.py checkpoints/rl/ --include-random --games 100
# Target Elo milestones: +200 (BC-level), +400 (beginner human)
```

---

## Complete backlog status

| Task | Description | Status |
|---|---|---|
| 7.1 | Card re-deal | ✅ DONE |
| 7.2 | War status advancement | ✅ DONE |
| 7.3 | Discard fix | ✅ DONE |
| 7.4 | Reinforcement events | ✅ DONE |
| 7.5 | SR subtype | ✅ DONE |
| 7.6 | pog_env.py alignment | ✅ DONE |
| 7.7 | Self-play smoke test | ✅ DONE |
| **7.8** | CARD_VP_DELTA (Reichstag Truce) | **← DO NOW** |
| — | BC checkpoint (cluster) | ⏳ Waiting |
| — | More training data | ⏳ User scrapes |
| — | Self-play training | ⏳ Blocked on BC |
| — | Elo evaluation | ⏳ Blocked on self-play |
