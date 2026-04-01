import functools

import jax
import jax.numpy as jnp

from src.env.jax_env import jax_reset
from src.rl.mcts import jax_mcts_search
from src.rl.network import PoGNet, load_adjacency_matrix


def test_jax_mcts_search_jit_and_vmap():
    model = PoGNet()
    adj = load_adjacency_matrix("pog_map_graph.json")
    state = jax_reset(jax.random.PRNGKey(0))

    dummy_obs = jnp.zeros((1, 32, 72), dtype=jnp.float32)
    dummy_cards = jnp.zeros((1, 7, 16), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(1), dummy_obs, dummy_cards, adj)

    search = functools.partial(
        jax_mcts_search,
        params=params,
        adj=adj,
        model=model,
        n_simulations=4,
        depth_limit=2,
    )

    policy = jax.jit(search)(state)
    assert policy.shape == (5341,)
    assert jnp.isclose(jnp.sum(policy), 1.0, atol=1e-5)

    states = jax.vmap(jax_reset)(jax.random.split(jax.random.PRNGKey(2), 2))
    policies = jax.jit(jax.vmap(search))(states)
    assert policies.shape == (2, 5341)
