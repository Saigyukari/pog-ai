# claude2gpt

## Review of Task 6.2 — PASSED ✅

`src/env/jax_env.py` VP tracking verified correct:
- `_recompute_vp(control)` helper added at line 463 — matches spec exactly ✓
- Called in `_combat_step` after control update (line 515) ✓
- Called in `_move_step`'s `do_move` after control update (line 535) ✓
- Terminal reward returns +1/-1 from active player's perspective (lines 577–586) ✓
- `jax_reset` initialises VP from `SPACE_CONTROL` via `_recompute_vp` (line 370) ✓
- `tests/test_vp_tracking.py` passes (4 tests total) ✓
- Confirmed: `jax_reset(...).vp == -1` (non-zero, reflects initial board control) ✓

---

## Critical gap found: hands empty after turn 1 → no card play for turns 2–20

In `jax_env.py`, `_advance_turn` never re-deals:

```python
def _advance_turn(state: JaxGameState) -> JaxGameState:
    next_action_round = state.action_round + 1
    wrap = next_action_round > 7          # turn boundary
    next_turn = state.turn + wrap
    next_action_round = jnp.where(wrap, 1, next_action_round)
    next_player = (1 - state.active_player)
    return state._replace(...)            # ap_hand and cp_hand never touched
```

`jax_reset` deals an initial hand (AP 6, CP 7). After both players exhaust
their starting cards — by action round 7 of turn 1 — both hands are all-255
(empty sentinel). For the remaining 19 turns only MOVE_UNIT actions remain
legal. Card-play is the primary strategic decision in PoG; without re-dealing,
self-play learns almost nothing about card selection.

The fix is small and self-contained.

---

## Task 7.1 — Card re-deal at turn boundary in `jax_env.py`

**File to modify:** `src/env/jax_env.py` only.

### Step 1 — Add `_deal_hand_from_deck` helper

Add this function near `_sample_hand` (around line 337):

```python
def _deal_hand_from_deck(
    deck: jnp.ndarray,
    discard_mask: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> jnp.ndarray:
    """
    Draw HAND_SIZE cards from deck, skipping permanently removed cards.

    deck: (N,) int16  — full faction physical deck (AP_PHYSICAL_DECK or CP_PHYSICAL_DECK)
    discard_mask: (N,) bool  — True = card permanently removed (remove_after_event)
    Returns: (HAND_SIZE,) int16  — hand with EMPTY_CARD (255) padding if needed
    """
    available = jnp.where(
        discard_mask,
        jnp.asarray(EMPTY_CARD, dtype=jnp.int16),
        deck,
    )
    perm = jax.random.permutation(rng_key, deck.shape[0])
    shuffled = available[perm]
    # Sort so EMPTY_CARD (255) sinks to the end, non-empty cards float up
    order = jnp.argsort(shuffled == EMPTY_CARD, stable=True)
    compacted = shuffled[order]
    return compacted[:HAND_SIZE]
```

`AP_PHYSICAL_DECK`, `CP_PHYSICAL_DECK`, `EMPTY_CARD`, `HAND_SIZE` are all
already defined at module level.

### Step 2 — Change `_advance_turn` signature and add re-deal

Current signature:
```python
def _advance_turn(state: JaxGameState) -> JaxGameState:
```

Replace the entire function with:

```python
def _advance_turn(state: JaxGameState, deal_key: jnp.ndarray) -> JaxGameState:
    next_action_round = state.action_round + jnp.asarray(1, dtype=jnp.int8)
    wrap = next_action_round > 7
    next_turn = state.turn + wrap.astype(jnp.int8)
    next_action_round = jnp.where(wrap, jnp.asarray(1, dtype=jnp.int8), next_action_round)
    next_player = (1 - state.active_player).astype(jnp.int8)

    # Re-deal both hands at every turn boundary.
    ap_key, cp_key = jax.random.split(deal_key)
    new_ap_hand = _deal_hand_from_deck(AP_PHYSICAL_DECK, state.ap_discard, ap_key)
    new_cp_hand = _deal_hand_from_deck(CP_PHYSICAL_DECK, state.cp_discard, cp_key)
    ap_hand = jnp.where(wrap, new_ap_hand, state.ap_hand)
    cp_hand = jnp.where(wrap, new_cp_hand, state.cp_hand)

    return state._replace(
        turn=next_turn,
        action_round=next_action_round,
        active_player=next_player,
        ap_hand=ap_hand,
        cp_hand=cp_hand,
    )
```

### Step 3 — Pass `deal_key` from `jax_step`

In `jax_step`, the current key split is:
```python
key, step_key = jax.random.split(state.rng_key)
```

Change to:
```python
key, step_key, deal_key = jax.random.split(state.rng_key, 3)
```

And update the `_advance_turn` call at the bottom of `jax_step`:
```python
# was: state = _advance_turn(state)._replace(rng_key=key)
state = _advance_turn(state, deal_key)._replace(rng_key=key)
```

That is the only call site for `_advance_turn` — there is one in `jax_step`, verify
no other call sites exist before editing.

### Step 4 — Write `tests/test_card_deal.py`

```python
import jax
import jax.numpy as jnp

from src.env.jax_env import EMPTY_CARD, FACTION_AP, jax_reset, jax_step


def test_reset_deals_nonempty_hands():
    state = jax_reset(jax.random.PRNGKey(0))
    assert bool(jnp.any(state.ap_hand != EMPTY_CARD))
    assert bool(jnp.any(state.cp_hand != EMPTY_CARD))


def test_hand_refreshed_after_turn_boundary():
    """Force action_round to 7, empty ap_hand, step with PASS → wrap → re-deal."""
    state = jax_reset(jax.random.PRNGKey(1))
    # Force last action round of turn 1 with empty AP hand
    state = state._replace(
        action_round=jnp.asarray(7, dtype=jnp.int8),
        ap_hand=jnp.full((7,), EMPTY_CARD, dtype=jnp.int16),
    )
    new_state, _, _ = jax_step(state, jnp.asarray(0, dtype=jnp.int32))  # PASS
    # After step: action_round wraps to 1 (turn 2), AP hand should be re-dealt
    assert int(new_state.turn) == 2
    assert int(new_state.action_round) == 1
    assert bool(jnp.any(new_state.ap_hand != EMPTY_CARD)), "AP hand still empty after re-deal"
```

### Acceptance criteria

- `jax_reset(rng).ap_hand` has non-255 entries ✓ (already true, just verify test passes)
- After stepping through action_round 7 with PASS, `new_state.ap_hand` is non-empty ✓
- `jax.jit(jax_step)` still compiles without error ✓
- `tests/test_card_deal.py` passes ✓
- All existing tests still pass ✓

**Do not touch** `train_selfplay.py`, `replay_buffer.py`, `mcts.py`, `pog_env.py`,
or any test outside `test_card_deal.py`. This is a single-file fix + one new test.
