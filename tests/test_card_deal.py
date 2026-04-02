import jax
import jax.numpy as jnp

from src.env.jax_env import EMPTY_CARD, jax_reset, jax_step


def test_reset_deals_nonempty_hands():
    state = jax_reset(jax.random.PRNGKey(0))
    assert bool(jnp.any(state.ap_hand != EMPTY_CARD))
    assert bool(jnp.any(state.cp_hand != EMPTY_CARD))


def test_hand_refreshed_after_turn_boundary():
    """Force action_round to 7, empty ap_hand, step with PASS -> wrap -> re-deal."""
    state = jax_reset(jax.random.PRNGKey(1))
    state = state._replace(
        action_round=jnp.asarray(7, dtype=jnp.int8),
        ap_hand=jnp.full((7,), EMPTY_CARD, dtype=jnp.int16),
    )
    new_state, _, _ = jax_step(state, jnp.asarray(0, dtype=jnp.int32))
    assert int(new_state.turn) == 2
    assert int(new_state.action_round) == 1
    assert bool(jnp.any(new_state.ap_hand != EMPTY_CARD)), "AP hand still empty after re-deal"
