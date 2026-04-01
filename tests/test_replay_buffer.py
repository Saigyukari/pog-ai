import jax
import jax.numpy as jnp
import pytest

from src.rl.replay_buffer import VRAMReplayBuffer


def test_replay_buffer_push_and_sample():
    buf = VRAMReplayBuffer(capacity=8)
    obs = jnp.ones((3, 32, 72), dtype=jnp.float32)
    card_ctx = jnp.ones((3, 7, 16), dtype=jnp.float32)
    mask = jnp.zeros((3, 5341), dtype=jnp.bool_)
    action = jnp.array([1, 2, 3], dtype=jnp.int32)
    reward = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    done = jnp.array([False, False, True], dtype=jnp.bool_)

    buf.push(obs, card_ctx, mask, action, reward, done)
    sample = buf.sample(2, jax.random.PRNGKey(0))

    assert len(buf) == 3
    assert sample[0].shape == (2, 32, 72)
    assert sample[1].shape == (2, 7, 16)
    assert sample[2].shape == (2, 5341)
    assert sample[3].shape == (2,)


def test_replay_buffer_sample_empty_raises():
    buf = VRAMReplayBuffer(capacity=4)
    with pytest.raises(ValueError):
        buf.sample(1, jax.random.PRNGKey(0))
