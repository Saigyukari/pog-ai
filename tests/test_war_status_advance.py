import jax
import jax.numpy as jnp

from src.data.pog_engine import PHASE_LIMITED, PHASE_TOTAL
from src.env.jax_env import jax_reset, jax_step


def _make_state_at_round_7(vp: int):
    """Return a state at action_round=7 with fixed VP, ready to wrap on next PASS."""
    state = jax_reset(jax.random.PRNGKey(42))
    return state._replace(
        action_round=jnp.asarray(7, dtype=jnp.int8),
        vp=jnp.asarray(vp, dtype=jnp.int8),
    )


def test_ap_advances_to_total_war_when_winning():
    state = _make_state_at_round_7(vp=5)
    new_state, _, _ = jax_step(state, jnp.asarray(0, dtype=jnp.int32))
    assert int(new_state.war_status_ap) == PHASE_TOTAL
    assert int(new_state.war_status_cp) == PHASE_LIMITED


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
    state = _make_state_at_round_7(vp=5)
    state = state._replace(
        war_status_ap=jnp.asarray(PHASE_TOTAL, dtype=jnp.int8),
    )
    new_state, _, _ = jax_step(state, jnp.asarray(0, dtype=jnp.int32))
    assert int(new_state.war_status_ap) == PHASE_TOTAL
