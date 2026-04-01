"""
replay_buffer.py — Device-array replay buffer for self-play RL.

Hot-path operations (`push`, `sample`) keep data as JAX arrays and avoid any
`.numpy()` conversion. Cold-storage HDF5 persistence is implemented behind an
optional `h5py` import because this environment does not currently ship with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp


@dataclass
class ReplayBufferState:
    obs: jnp.ndarray
    card_ctx: jnp.ndarray
    mask: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    position: jnp.ndarray
    size: jnp.ndarray


class VRAMReplayBuffer:
    """
    Fixed-size replay buffer stored as JAX arrays.

    The buffer is mutable at the Python object level but all payload arrays stay
    resident as device arrays.
    """

    def __init__(self, capacity: int = 500_000):
        self.capacity = int(capacity)
        self.state = ReplayBufferState(
            obs=jnp.zeros((capacity, 32, 72), dtype=jnp.float32),
            card_ctx=jnp.zeros((capacity, 7, 16), dtype=jnp.float32),
            mask=jnp.zeros((capacity, 5341), dtype=jnp.bool_),
            action=jnp.zeros((capacity,), dtype=jnp.int32),
            reward=jnp.zeros((capacity,), dtype=jnp.float32),
            done=jnp.zeros((capacity,), dtype=jnp.bool_),
            position=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
        )

    def __len__(self) -> int:
        return int(self.state.size)

    def push(self, obs, card_ctx, mask, action, reward, done):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        card_ctx = jnp.asarray(card_ctx, dtype=jnp.float32)
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        action = jnp.asarray(action, dtype=jnp.int32)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.bool_)

        if obs.ndim == 2:
            obs = obs[None, ...]
            card_ctx = card_ctx[None, ...]
            mask = mask[None, ...]
            action = action[None, ...]
            reward = reward[None, ...]
            done = done[None, ...]

        n = int(obs.shape[0])
        indices = (jnp.arange(n, dtype=jnp.int32) + self.state.position) % self.capacity

        self.state = ReplayBufferState(
            obs=self.state.obs.at[indices].set(obs),
            card_ctx=self.state.card_ctx.at[indices].set(card_ctx),
            mask=self.state.mask.at[indices].set(mask),
            action=self.state.action.at[indices].set(action),
            reward=self.state.reward.at[indices].set(reward),
            done=self.state.done.at[indices].set(done),
            position=(self.state.position + n) % self.capacity,
            size=jnp.minimum(self.state.size + n, self.capacity),
        )

    def sample(self, batch_size: int, rng_key) -> tuple[jnp.ndarray, ...]:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        batch_size = int(batch_size)
        rng_key = jnp.asarray(rng_key, dtype=jnp.uint32)
        idx = jax.random.randint(rng_key, (batch_size,), 0, self.state.size)
        return (
            self.state.obs[idx],
            self.state.card_ctx[idx],
            self.state.mask[idx],
            self.state.action[idx],
            self.state.reward[idx],
            self.state.done[idx],
        )

    def save_hdf5(self, path: str):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("save_hdf5 requires h5py to be installed.") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        size = len(self)
        payload = jax.device_get(
            (
                self.state.obs[:size],
                self.state.card_ctx[:size],
                self.state.mask[:size],
                self.state.action[:size],
                self.state.reward[:size],
                self.state.done[:size],
                self.state.position,
                self.state.size,
            )
        )

        with h5py.File(path, "w") as handle:
            handle.attrs["capacity"] = self.capacity
            handle.create_dataset("obs", data=payload[0])
            handle.create_dataset("card_ctx", data=payload[1])
            handle.create_dataset("mask", data=payload[2])
            handle.create_dataset("action", data=payload[3])
            handle.create_dataset("reward", data=payload[4])
            handle.create_dataset("done", data=payload[5])
            handle.attrs["position"] = int(payload[6])
            handle.attrs["size"] = int(payload[7])

    def load_hdf5(self, path: str):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("load_hdf5 requires h5py to be installed.") from exc

        with h5py.File(path, "r") as handle:
            capacity = int(handle.attrs.get("capacity", self.capacity))
            if capacity != self.capacity:
                raise ValueError(f"HDF5 capacity {capacity} does not match buffer capacity {self.capacity}.")
            size = int(handle.attrs["size"])
            position = int(handle.attrs["position"])

            obs = self.state.obs.at[:size].set(jnp.asarray(handle["obs"][...], dtype=jnp.float32))
            card_ctx = self.state.card_ctx.at[:size].set(jnp.asarray(handle["card_ctx"][...], dtype=jnp.float32))
            mask = self.state.mask.at[:size].set(jnp.asarray(handle["mask"][...], dtype=jnp.bool_))
            action = self.state.action.at[:size].set(jnp.asarray(handle["action"][...], dtype=jnp.int32))
            reward = self.state.reward.at[:size].set(jnp.asarray(handle["reward"][...], dtype=jnp.float32))
            done = self.state.done.at[:size].set(jnp.asarray(handle["done"][...], dtype=jnp.bool_))

        self.state = ReplayBufferState(
            obs=obs,
            card_ctx=card_ctx,
            mask=mask,
            action=action,
            reward=reward,
            done=done,
            position=jnp.asarray(position, dtype=jnp.int32),
            size=jnp.asarray(size, dtype=jnp.int32),
        )
