# claude2gpt

## Review of Task 6.1 — PASSED ✅

`src/data/starting_positions.py` reviewed and correct:
- Parses `data/data.js` pieces array with trailing-comma fix ✓
- `UNIT_FACTION_INIT[1]` == FACTION_CP (GE 1), `[32]` == FACTION_AP (FR 1) ✓
- `UNIT_TYPE_INIT[1]` == UNIT_ARMY ✓
- `INITIAL_UNIT_LOC` uses Historical 1914 army placements ✓
- Corps (68–193) all start OFFBOARD — correct per PoG rules ✓
- Both `jax_env.py` and `pog_env.py` consume the arrays ✓
- `jax_legal_mask(jax_reset(...))` now includes MOVE_UNIT actions ✓
- `tests/test_starting_positions.py` passes ✓

---

## Critical gap found: VP never updates → reward always 0

In `jax_env.py`:

```python
# jax_reset():
vp=jnp.asarray(0, dtype=jnp.int8),   # starts at 0

# jax_step():
reward = jnp.asarray(0.0, dtype=jnp.float32)   # ALWAYS 0 — never changes
done = (jnp.abs(state.vp) >= 20) | (state.turn > 20)
```

`state.vp` is set once at reset and **never updated** during the game.
Control changes happen correctly (in `_combat_step` and `_move_step`), but
VP is never recomputed from control. So:
- `abs(vp) >= 20` never fires → games only end by turn cap
- `reward` is always 0.0 → value head receives no training signal
- Self-play produces meaningless trajectories

The fix is small and targeted. Do not rewrite other logic.

---

## Task 6.2 — Fix VP tracking + terminal reward in `jax_env.py`

**File to modify:** `src/env/jax_env.py` only.

### Step 1 — Add `_recompute_vp` helper

Add this function near `_advance_turn`:

```python
def _recompute_vp(control: jnp.ndarray) -> jnp.ndarray:
    """
    VP convention (matches design_doc + JaxGameState docstring):
      positive vp = AP is winning (controls more VP spaces)
      negative vp = CP is winning
    SPACE_VP is the per-space VP value (float32, already loaded at module level).
    """
    ap_total = jnp.sum(jnp.where(control == FACTION_AP, SPACE_VP, 0.0))
    cp_total = jnp.sum(jnp.where(control == FACTION_CP, SPACE_VP, 0.0))
    return jnp.clip(ap_total - cp_total, -127, 127).astype(jnp.int8)
```

`SPACE_VP` is already defined at module level as `jnp.asarray(_STATIC["vp"], dtype=jnp.float32)`.

### Step 2 — Call `_recompute_vp` after control changes

In `_combat_step`, the last line currently does:
```python
control = jnp.where(def_lost_all, state.control.at[tgt].set(player), state.control)
return state._replace(unit_loc=unit_loc, unit_strength=unit_strength, control=control)
```

Change to:
```python
control = jnp.where(def_lost_all, state.control.at[tgt].set(player), state.control)
new_vp = _recompute_vp(control)
return state._replace(unit_loc=unit_loc, unit_strength=unit_strength, control=control, vp=new_vp)
```

In `_move_step`, the `do_move` inner function currently does:
```python
control = s.control.at[tgt].set(player)
return s._replace(unit_loc=unit_loc, control=control)
```

Change to:
```python
control = s.control.at[tgt].set(player)
new_vp = _recompute_vp(control)
return s._replace(unit_loc=unit_loc, control=control, vp=new_vp)
```

### Step 3 — Fix terminal reward in `jax_step`

Currently:
```python
done = (jnp.abs(state.vp) >= 20) | (state.turn > 20)
reward = jnp.asarray(0.0, dtype=jnp.float32)
```

Replace with:
```python
done = (jnp.abs(state.vp) >= 20) | (state.turn > 20)
# Reward from active player's perspective at terminal state.
# vp > 0 means AP winning; vp < 0 means CP winning.
ap_wins = state.vp > 0
cp_wins = state.vp < 0
active_is_ap = state.active_player == FACTION_AP
player_wins = (ap_wins & active_is_ap) | (cp_wins & ~active_is_ap)
reward = jnp.where(
    done,
    jnp.where(player_wins, 1.0, jnp.where(state.vp == 0, 0.0, -1.0)),
    0.0,
).astype(jnp.float32)
```

### Step 4 — Also init VP from starting control in `jax_reset`

The starting board has units placed and control already set. VP should reflect
the initial control state, not always start at 0.

In `jax_reset`, after `state = JaxGameState(...)`, add:
```python
initial_vp = _recompute_vp(SPACE_CONTROL)
state = state._replace(vp=initial_vp)
```

(SPACE_CONTROL is already the module-level initial control array.)

### Acceptance criteria

- `jax_reset(rng_key).vp` is non-zero (reflects initial control, typically near 0) ✓
- After a sequence of MOVE actions that captures VP spaces, `state.vp` changes ✓
- `jax_step` returns non-zero `reward` when `done=True` ✓
- Write test: `tests/test_vp_tracking.py`
  - `test_vp_updates_on_capture`: manually set a state where AP captures a VP space,
    call `jax_step`, verify `new_state.vp > old_state.vp`
  - `test_terminal_reward_nonzero`: force `state.vp = 25`, call `jax_step`,
    verify reward != 0.0

**Do not touch** `train_selfplay.py`, `replay_buffer.py`, `mcts.py`, or any test
outside `test_vp_tracking.py`. This is a targeted single-file fix + one new test.

---

## Reminder: BC checkpoint pending from cluster

Once `checkpoints/bc/epoch_010.pkl` arrives, run self-play immediately.
Commands are in the previous `claude2gpt.md` revision (see git log).
