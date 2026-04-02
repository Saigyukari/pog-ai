import jax
import jax.numpy as jnp

from src.env.jax_env import (
    ACT_OPS_START,
    CARD_FACTION,
    FACTION_AP,
    OFFBOARD,
    jax_reset,
    jax_step,
)


def test_sr_action_brings_unit_on_map():
    state = jax_reset(jax.random.PRNGKey(0))
    card_idx = int(jnp.where(CARD_FACTION == FACTION_AP, jnp.arange(CARD_FACTION.shape[0]), 999).min())
    offboard_before = int(jnp.sum(state.unit_loc == OFFBOARD))
    state = state._replace(
        active_player=jnp.asarray(FACTION_AP, dtype=jnp.int8),
        ap_hand=jnp.asarray([card_idx, 255, 255, 255, 255, 255, 255], dtype=jnp.int16),
    )
    sr_action = jnp.asarray(ACT_OPS_START + card_idx * 3 + 2, dtype=jnp.int32)
    new_state, _, _ = jax_step(state, sr_action)
    offboard_after = int(jnp.sum(new_state.unit_loc == OFFBOARD))
    assert offboard_after < offboard_before
