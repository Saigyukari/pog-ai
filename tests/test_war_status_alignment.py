import jax
import jax.numpy as jnp

from src.data.pog_engine import PHASE_LIMITED, PHASE_TOTAL
from src.env.jax_env import jax_legal_mask, jax_obs, jax_reset
from src.env.pog_env import FACTION_AP, FACTION_CP, PogEnv


def test_jax_obs_exposes_distinct_ap_cp_war_status_planes():
    state = jax_reset(jax.random.PRNGKey(0))._replace(
        war_status_ap=jnp.asarray(PHASE_TOTAL, dtype=jnp.int8),
        war_status_cp=jnp.asarray(PHASE_LIMITED, dtype=jnp.int8),
    )
    obs, _ = jax_obs(state, 0)
    assert float(obs[29, 0]) == 1.0
    assert float(obs[30, 0]) == 0.0


def test_jax_legal_mask_uses_active_player_war_status():
    state = jax_reset(jax.random.PRNGKey(1))

    env = PogEnv("pog_map_graph.json", "pog_cards_db.json")
    env.reset(seed=0)
    ap_total_idx = next(i for i, card in enumerate(env._cards_db) if card["faction"] == FACTION_AP and card["phase_gate"] == PHASE_TOTAL)
    cp_total_idx = next(i for i, card in enumerate(env._cards_db) if card["faction"] == FACTION_CP and card["phase_gate"] == PHASE_TOTAL)

    state = state._replace(
        ap_hand=jnp.asarray([ap_total_idx, 255, 255, 255, 255, 255, 255], dtype=jnp.int16),
        cp_hand=jnp.asarray([cp_total_idx, 255, 255, 255, 255, 255, 255], dtype=jnp.int16),
        active_player=jnp.asarray(FACTION_AP, dtype=jnp.int8),
        war_status_ap=jnp.asarray(PHASE_LIMITED, dtype=jnp.int8),
        war_status_cp=jnp.asarray(PHASE_TOTAL, dtype=jnp.int8),
    )
    ap_mask = jax_legal_mask(state)
    assert not bool(ap_mask[1 + ap_total_idx])

    state = state._replace(
        active_player=jnp.asarray(FACTION_CP, dtype=jnp.int8),
        war_status_ap=jnp.asarray(PHASE_TOTAL, dtype=jnp.int8),
        war_status_cp=jnp.asarray(PHASE_TOTAL, dtype=jnp.int8),
    )
    cp_mask = jax_legal_mask(state)
    assert bool(cp_mask[1 + cp_total_idx])


def test_cpu_env_reset_initializes_vp_from_control():
    env = PogEnv("pog_map_graph.json", "pog_cards_db.json")
    env.reset(seed=0)
    assert env.vp_track == env._recompute_vp_track()
    assert env.vp_track != 0
