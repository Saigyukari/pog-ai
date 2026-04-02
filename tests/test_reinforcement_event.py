import jax
import jax.numpy as jnp

from src.env.jax_env import (
    ACT_EVENT_START,
    CARD_FACTION,
    CARD_REINF_COUNT,
    FACTION_AP,
    OFFBOARD,
    jax_reset,
    jax_step,
)


def test_reinforcement_card_brings_unit_on_map():
    for i in range(CARD_REINF_COUNT.shape[0]):
        if int(CARD_REINF_COUNT[i]) > 0 and int(CARD_FACTION[i]) == FACTION_AP:
            card_idx = i
            break
    else:
        raise RuntimeError("no AP reinforcement card found")

    state = jax_reset(jax.random.PRNGKey(0))
    offboard_before = int(jnp.sum(state.unit_loc == OFFBOARD))
    state = state._replace(
        active_player=jnp.asarray(FACTION_AP, dtype=jnp.int8),
        ap_hand=jnp.asarray([card_idx, 255, 255, 255, 255, 255, 255], dtype=jnp.int16),
    )
    action = jnp.asarray(ACT_EVENT_START + card_idx, dtype=jnp.int32)
    new_state, _, _ = jax_step(state, action)
    offboard_after = int(jnp.sum(new_state.unit_loc == OFFBOARD))
    assert offboard_after < offboard_before
