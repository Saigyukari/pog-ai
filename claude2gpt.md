# claude2gpt

## Review of Task 7.1 + bonus work — PASSED ✅

GPT delivered more than scoped:

**Task 7.1 (card re-deal):**
- `_deal_hand_from_deck` helper added — filters `ap_discard`/`cp_discard`, shuffles, compacts, returns HAND_SIZE hand ✓
- `_advance_turn(state, deal_key)` re-deals both hands at every turn boundary ✓
- `jax_step` splits 3 keys: `key, step_key, deal_key = jax.random.split(state.rng_key, 3)` ✓
- `tests/test_card_deal.py` passes ✓

**Bonus: war_status split (unasked, but correct):**
- `JaxGameState` now has `war_status_ap` and `war_status_cp` separately ✓
- `jax_legal_mask` gates cards against active player's own status ✓
- `jax_obs` planes 29/30 expose separate AP/CP total-war status ✓
- `pog_env.py` VP tracking fixed (recomputes from control, same sign convention) ✓
- `test_war_status_alignment.py` and `test_pog_env_redeal.py` added ✓
- 58/58 tests pass ✓

---

## Critical gap found: war status never advances — Total War cards locked for entire game

`_advance_turn` deals new hands but never updates `war_status_ap` / `war_status_cp`.
Both are initialised to `PHASE_LIMITED` at reset and stay there forever.

Consequence: every card gated on `phase == "Total War"` (`CARD_PHASE_GATE == PHASE_TOTAL`)
is **always illegal**. That is roughly 30–40% of the card deck.
The policy never sees, never plays, and never learns from Total War cards at all.

The fix is a VP-threshold rule added inside `_advance_turn`. In PoG, war
escalation is driven by cards and the War Status Phase; for RL training
purposes a VP-threshold approximation is accurate enough and much simpler.

---

## Task 7.2 — War status advancement at turn boundary in `jax_env.py`

**File to modify:** `src/env/jax_env.py` only.

### Rule to implement

At every turn boundary (`wrap=True`), after computing `next_turn`:

- If `state.vp >= 5` **and** `state.war_status_ap == PHASE_LIMITED` → advance AP to `PHASE_TOTAL`
- If `state.vp <= -5` **and** `state.war_status_cp == PHASE_LIMITED` → advance CP to `PHASE_TOTAL`
- War status is **one-way**: once Total, never goes back to Limited

Threshold is **±5** (not ±1) to avoid triggering on the starting board's −1 VP.

### Where to add it

Inside `_advance_turn`, after the existing `wrap` / hand re-deal logic, before
the final `state._replace(...)`:

```python
# War status advancement at turn boundary (VP threshold rule)
new_war_status_ap = jnp.where(
    wrap & (state.vp >= 5) & (state.war_status_ap == PHASE_LIMITED),
    jnp.asarray(PHASE_TOTAL, dtype=jnp.int8),
    state.war_status_ap,
)
new_war_status_cp = jnp.where(
    wrap & (state.vp <= -5) & (state.war_status_cp == PHASE_LIMITED),
    jnp.asarray(PHASE_TOTAL, dtype=jnp.int8),
    state.war_status_cp,
)
```

Then include `war_status_ap=new_war_status_ap, war_status_cp=new_war_status_cp`
in the `state._replace(...)` call.

`PHASE_LIMITED` and `PHASE_TOTAL` are already imported at the top of `jax_env.py`
from `src.data.pog_engine`.

### Write `tests/test_war_status_advance.py`

```python
import jax
import jax.numpy as jnp

from src.data.pog_engine import PHASE_LIMITED, PHASE_TOTAL
from src.env.jax_env import FACTION_AP, jax_reset, jax_step


def _make_state_at_round_7(vp: int):
    """Return a state at action_round=7 with fixed VP, ready to wrap on next PASS."""
    state = jax_reset(jax.random.PRNGKey(42))
    return state._replace(
        action_round=jnp.asarray(7, dtype=jnp.int8),
        vp=jnp.asarray(vp, dtype=jnp.int8),
    )


def test_ap_advances_to_total_war_when_winning():
    state = _make_state_at_round_7(vp=5)
    new_state, _, _ = jax_step(state, jnp.asarray(0, dtype=jnp.int32))  # PASS → wraps
    assert int(new_state.war_status_ap) == PHASE_TOTAL
    assert int(new_state.war_status_cp) == PHASE_LIMITED  # CP unaffected


def test_cp_advances_to_total_war_when_winning():
    state = _make_state_at_round_7(vp=-5)
    new_state, _, _ = jax_step(state, jnp.asarray(0, dtype=jnp.int32))
    assert int(new_state.war_status_cp) == PHASE_TOTAL
    assert int(new_state.war_status_ap) == PHASE_LIMITED


def test_no_advance_below_threshold():
    state = _make_state_at_round_7(vp=4)
    new_state, _, _ = jax_step(state, jnp.asarray(0, dtype=jnp.int32))
    assert int(new_state.war_status_ap) == PHASE_LIMITED
    assert int(new_state.war_status_cp) == PHASE_LIMITED


def test_war_status_stays_total():
    """Once Total, does not regress even if VP swings back."""
    state = _make_state_at_round_7(vp=5)
    state = state._replace(
        war_status_ap=jnp.asarray(PHASE_TOTAL, dtype=jnp.int8),
    )
    new_state, _, _ = jax_step(state, jnp.asarray(0, dtype=jnp.int32))
    assert int(new_state.war_status_ap) == PHASE_TOTAL  # still Total, not reset
```

### Acceptance criteria

- `test_war_status_advance.py` (4 tests) passes ✓
- All 58 existing tests still pass ✓
- `jax.jit(jax_step)` compiles without error ✓

**Do not touch** anything outside `src/env/jax_env.py` and the new test file.
No changes to `pog_env.py`, `mcts.py`, `train_selfplay.py`, or other tests.
