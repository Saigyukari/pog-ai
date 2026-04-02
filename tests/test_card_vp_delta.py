import jax
import jax.numpy as jnp

from src.env.jax_env import (
    ACT_EVENT_START,
    CARD_FACTION,
    CARD_VP_DELTA,
    FACTION_CP,
    jax_reset,
    jax_step,
)


def _find_reichstag() -> int:
    for i in range(CARD_VP_DELTA.shape[0]):
        if int(CARD_VP_DELTA[i]) == -1:
            return i
    raise RuntimeError("Reichstag Truce not found in CARD_VP_DELTA")


def test_reichstag_truce_decrements_vp():
    card_idx = _find_reichstag()
    assert int(CARD_FACTION[card_idx]) == FACTION_CP

    state = jax_reset(jax.random.PRNGKey(0))
    vp_before = int(state.vp)
    state = state._replace(
        active_player=jnp.asarray(FACTION_CP, dtype=jnp.int8),
        cp_hand=jnp.asarray([card_idx, 255, 255, 255, 255, 255, 255], dtype=jnp.int16),
    )
    action = jnp.asarray(ACT_EVENT_START + card_idx, dtype=jnp.int32)
    new_state, _, _ = jax_step(state, action)
    assert int(new_state.vp) == vp_before - 1


def test_card_vp_delta_array_has_expected_nonzero():
    nonzero = int(jnp.sum(CARD_VP_DELTA != 0))
    assert nonzero >= 1
