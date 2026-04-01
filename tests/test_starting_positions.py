import jax
import jax.numpy as jnp

from src.data.pog_engine import FACTION_AP, FACTION_CP, UNIT_ARMY
from src.data.starting_positions import (
    INITIAL_UNIT_LOC,
    UNIT_FACTION_INIT,
    UNIT_STRENGTH_INIT,
    UNIT_TYPE_INIT,
)
from src.env.jax_env import ACT_MOVE_START, jax_legal_mask, jax_reset


def test_starting_position_arrays_basic_values():
    assert UNIT_FACTION_INIT.shape == (194,)
    assert UNIT_FACTION_INIT[1] == FACTION_CP
    assert UNIT_FACTION_INIT[32] == FACTION_AP
    assert UNIT_TYPE_INIT[1] == UNIT_ARMY
    assert UNIT_STRENGTH_INIT[1] > 0
    assert INITIAL_UNIT_LOC.shape == (194,)


def test_jax_reset_has_move_actions():
    state = jax_reset(jax.random.PRNGKey(0))
    legal = jax_legal_mask(state)
    assert bool(jnp.any(legal[ACT_MOVE_START:]))
