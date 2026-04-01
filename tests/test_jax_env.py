import jax
import jax.numpy as jnp

from src.env.jax_env import jax_legal_mask, jax_obs, jax_reset, jax_step


def test_jax_env_reset_obs_and_mask_shapes():
    state = jax_reset(jax.random.PRNGKey(0))

    obs, card_ctx = jax_obs(state, 0)
    mask = jax_legal_mask(state)

    assert obs.shape == (32, 72)
    assert card_ctx.shape == (7, 16)
    assert mask.shape == (5341,)
    assert bool(mask[0])


def test_jax_env_step_jit_and_vmap():
    state = jax_reset(jax.random.PRNGKey(1))
    action = jnp.asarray(0, dtype=jnp.int32)

    next_state, reward, done = jax.jit(jax_step)(state, action)
    assert next_state.action_round == 2
    assert reward.shape == ()
    assert done.shape == ()

    stacked = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
    _, rewards, dones = jax.vmap(jax_step, in_axes=(0, 0))(stacked, jnp.array([0, 0], dtype=jnp.int32))
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
