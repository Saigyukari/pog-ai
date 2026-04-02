import jax
import jax.numpy as jnp

from src.data.starting_positions import UNIT_FACTION_INIT, UNIT_STRENGTH_INIT
from src.env.jax_env import (
    FACTION_AP,
    INITIAL_UNIT_LOC,
    JaxGameState,
    SPACE_CONTROL,
    UNIT_TYPE,
    _recompute_vp,
    jax_step,
)


def _make_state() -> JaxGameState:
    rng_key = jax.random.PRNGKey(0)
    return JaxGameState(
        unit_loc=INITIAL_UNIT_LOC,
        unit_strength=UNIT_STRENGTH_INIT,
        trench_level=jnp.zeros((72,), dtype=jnp.int8),
        oos_mask=jnp.zeros((72,), dtype=jnp.bool_),
        control=SPACE_CONTROL,
        ap_hand=jnp.full((7,), 255, dtype=jnp.int16),
        cp_hand=jnp.full((7,), 255, dtype=jnp.int16),
        ap_discard=jnp.zeros((65,), dtype=jnp.bool_),
        cp_discard=jnp.zeros((65,), dtype=jnp.bool_),
        turn=jnp.asarray(1, dtype=jnp.int8),
        action_round=jnp.asarray(1, dtype=jnp.int8),
        active_player=jnp.asarray(FACTION_AP, dtype=jnp.int8),
        war_status_ap=jnp.asarray(0, dtype=jnp.int8),
        war_status_cp=jnp.asarray(0, dtype=jnp.int8),
        vp=_recompute_vp(SPACE_CONTROL),
        rng_key=rng_key,
    )


def test_vp_updates_on_capture():
    state = _make_state()

    # Force a simple adjacent AP->CP VP capture on a known edge (1 -> 0).
    src = 1
    tgt = 0
    control = state.control.at[src].set(FACTION_AP).at[tgt].set(1)
    state = state._replace(
        unit_loc=state.unit_loc.at[32].set(src),
        unit_strength=state.unit_strength.at[32].set(3),
        control=control,
        vp=_recompute_vp(control),
    )

    action = jnp.asarray(441 + src * 72 + tgt, dtype=jnp.int32)
    new_state, _, _ = jax_step(state, action)
    assert int(new_state.vp) > int(state.vp)


def test_terminal_reward_nonzero():
    state = _make_state()._replace(vp=jnp.asarray(25, dtype=jnp.int8))
    new_state, reward, done = jax_step(state, jnp.asarray(0, dtype=jnp.int32))
    assert bool(done)
    assert float(reward) != 0.0
