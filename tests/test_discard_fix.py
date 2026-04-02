import jax
import jax.numpy as jnp

from src.env.jax_env import (
    ACT_EVENT_START,
    ACT_OPS_START,
    AP_UNIQUE_LOCAL,
    CARD_FACTION,
    CARD_REMOVE_AFTER_EVENT,
    FACTION_AP,
    jax_reset,
    jax_step,
)


def _find_card(remove: bool, faction_eq_ap: bool) -> int:
    for i in range(CARD_REMOVE_AFTER_EVENT.shape[0]):
        rae = bool(CARD_REMOVE_AFTER_EVENT[i])
        fap = int(CARD_FACTION[i]) == FACTION_AP
        if rae == remove and fap == faction_eq_ap:
            return i
    raise RuntimeError("card not found")


def test_remove_after_event_card_is_permanently_discarded():
    state = jax_reset(jax.random.PRNGKey(0))
    card_idx = _find_card(remove=True, faction_eq_ap=True)
    state = state._replace(
        active_player=jnp.asarray(FACTION_AP, dtype=jnp.int8),
        ap_hand=jnp.asarray([card_idx, 255, 255, 255, 255, 255, 255], dtype=jnp.int16),
    )
    action = jnp.asarray(ACT_EVENT_START + card_idx, dtype=jnp.int32)
    new_state, _, _ = jax_step(state, action)
    local = int(AP_UNIQUE_LOCAL[card_idx])
    assert bool(new_state.ap_discard[local])


def test_reusable_card_not_permanently_discarded_on_ops():
    state = jax_reset(jax.random.PRNGKey(1))
    card_idx = _find_card(remove=False, faction_eq_ap=True)
    state = state._replace(
        active_player=jnp.asarray(FACTION_AP, dtype=jnp.int8),
        ap_hand=jnp.asarray([card_idx, 255, 255, 255, 255, 255, 255], dtype=jnp.int16),
    )
    ops_action = jnp.asarray(ACT_OPS_START + card_idx * 3, dtype=jnp.int32)
    new_state, _, _ = jax_step(state, ops_action)
    local = int(AP_UNIQUE_LOCAL[card_idx])
    assert not bool(new_state.ap_discard[local])
