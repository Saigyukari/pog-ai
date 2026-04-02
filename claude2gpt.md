# claude2gpt

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
