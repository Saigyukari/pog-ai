#!/usr/bin/env python3
"""
train_selfplay.py — AlphaZero-style self-play trainer for Paths of Glory.

This script is intentionally pragmatic:
  - actor data generation uses the JAX env + vmapped `jax_mcts_search`
  - learner updates use the VRAM replay buffer with MCTS policy targets
  - parameter sync is explicit and periodic (stale actor params)

The current game environment is still mechanically minimal, so this script is a
working training loop scaffold rather than a final-performance implementation.
"""

from __future__ import annotations

import argparse
import functools
import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.env.jax_env import jax_legal_mask, jax_obs, jax_reset, jax_step
from src.rl.mcts import jax_mcts_search
from src.rl.network import PoGNet, create_train_state, load_adjacency_matrix
from src.rl.replay_buffer import VRAMReplayBuffer

N_ACTIONS = 5341
N_PLANES = 32
N_SPACES = 72


@dataclass
class EpisodeSlot:
    obs: list
    card_ctx: list
    mask: list
    policy: list
    action: list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-play RL for PoG.")
    parser.add_argument("--bc-checkpoint", help="BC checkpoint pickle produced by train_bc.py")
    parser.add_argument("--checkpoint-in", help="Optional alias for --bc-checkpoint.")
    parser.add_argument("--map", dest="map_json", default="pog_map_graph.json", help="Path to board graph JSON.")
    parser.add_argument("--iterations", type=int, default=10, help="Outer self-play iterations.")
    parser.add_argument("--n-actors", type=int, help="Parallel self-play environments.")
    parser.add_argument("--num-envs", type=int, help="Deprecated alias for --n-actors.")
    parser.add_argument("--mcts-sims", type=int, help="Override MCTS simulations per move.")
    parser.add_argument("--depth-limit", type=int, help="Override JAX MCTS depth limit.")
    parser.add_argument("--temperature", type=float, help="Override actor sampling temperature.")
    parser.add_argument("--max-steps", type=int, default=64, help="Hard cap on plies per actor batch.")
    parser.add_argument("--buffer-capacity", type=int, default=50_000, help="Replay buffer capacity.")
    parser.add_argument("--min-buffer-size", type=int, default=1_024, help="Minimum samples before learner updates.")
    parser.add_argument("--batch-size", type=int, default=256, help="Learner minibatch size.")
    parser.add_argument("--learner-batch-size", type=int, help="Deprecated alias for --batch-size.")
    parser.add_argument("--learner-steps", type=int, default=8, help="Learner updates per iteration.")
    parser.add_argument("--sync-every", type=int, default=8, help="Broadcast learner params to actor every N learner steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learner Adam LR.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/selfplay", help="Output directory for self-play checkpoints.")
    parser.add_argument("--save-every", type=int, default=1, help="Save every N iterations.")
    parser.add_argument("--total-games-played", type=int, default=0, help="Resume counter for search schedule staging.")
    args = parser.parse_args()
    args.bc_checkpoint = args.checkpoint_in or args.bc_checkpoint
    if not args.bc_checkpoint:
        parser.error("one of --bc-checkpoint or --checkpoint-in is required")
    if args.learner_batch_size is not None:
        args.batch_size = args.learner_batch_size
    return args


def load_params(checkpoint_path: str):
    with open(checkpoint_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a train_bc-style payload.")
    return payload["params"]


def _flat_action_scores(policy_heads) -> jnp.ndarray:
    card_logits, atype_logits, src_logits, tgt_logits = policy_heads
    batch = card_logits.shape[0]
    scores = jnp.full((batch, N_ACTIONS), -1e9, dtype=card_logits.dtype)
    scores = scores.at[:, 0].set(0.0)
    scores = scores.at[:, 1:111].set(card_logits + atype_logits[:, 0:1])
    scores = scores.at[:, 111:441].set((card_logits[:, :, None] + atype_logits[:, None, :]).reshape(batch, 330))
    move_scores = src_logits[:, :, None] + tgt_logits[:, None, :]
    scores = scores.at[:, 441:].set(move_scores.reshape(batch, -1)[:, : N_ACTIONS - 441])
    return scores


def alphazero_loss(params, batch, model: PoGNet, adj: jnp.ndarray, lambda_v: float = 1.0, lambda_reg: float = 1e-4):
    obs, card_ctx, legal_mask, target_policy, action, outcome, done = batch
    policy_heads, value = model.apply(params, obs, card_ctx, adj)

    flat_scores = _flat_action_scores(policy_heads)
    masked_scores = jnp.where(legal_mask, flat_scores, -1e9)
    log_probs = jax.nn.log_softmax(masked_scores, axis=-1)
    policy_loss = -jnp.mean(jnp.sum(target_policy * log_probs, axis=-1))

    value_loss = jnp.mean((value[:, 0] - outcome) ** 2)
    l2 = sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(params))
    total = policy_loss + lambda_v * value_loss + lambda_reg * l2

    return total, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "l2_loss": lambda_reg * l2,
        "total_loss": total,
    }


def build_learner_step(model: PoGNet, adj: jnp.ndarray):
    @jax.jit
    def learner_step(state, batch):
        def loss_fn(params):
            loss, metrics = alphazero_loss(params, batch, model, adj)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        metrics["total_loss"] = loss
        return new_state, metrics

    return learner_step


def save_checkpoint(path: Path, params, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"params": jax.device_get(params), "metadata": metadata}, handle)


def final_outcome_from_state(state) -> float:
    vp = int(jax.device_get(state.vp))
    if vp > 0:
        return 1.0
    if vp < 0:
        return -1.0
    return 0.0


def get_search_params(total_games: int) -> tuple[int, int, float]:
    """Returns (depth_limit, n_simulations, temperature)."""
    if total_games < 2_000:
        return 6, 256, 1.0
    if total_games < 10_000:
        return 5, 192, 1.0
    return 4, 128, 0.5


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.bc_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint_path}")

    n_local_devices = len(jax.local_devices())
    multi_gpu = n_local_devices > 1
    n_actors = args.n_actors or args.num_envs or (256 if multi_gpu else 16)
    if args.buffer_capacity == 50_000 and multi_gpu:
        args.buffer_capacity = 500_000

    adj = load_adjacency_matrix(args.map_json)
    model = PoGNet(hidden_dim=128, n_gat_layers=6)
    params = load_params(str(checkpoint_path))
    learner_state = create_train_state(
        rng=jax.random.PRNGKey(args.seed),
        model=model,
        dummy_spatial=jnp.zeros((1, N_PLANES, N_SPACES), dtype=jnp.float32),
        dummy_cards=jnp.zeros((1, 7, 16), dtype=jnp.float32),
        adj=adj,
        learning_rate=args.lr,
    ).replace(params=params)

    actor_params = learner_state.params
    learner_step = build_learner_step(model, adj)
    replay = VRAMReplayBuffer(capacity=args.buffer_capacity)
    batched_mask = jax.jit(jax.vmap(jax_legal_mask))
    batched_step = jax.jit(jax.vmap(jax_step))
    batched_obs = jax.jit(jax.vmap(jax_obs, in_axes=(0, 0)))

    rng = jax.random.PRNGKey(args.seed)
    learner_updates = 0
    total_games_played = int(args.total_games_played)

    print(
        f"Devices={jax.local_devices()}  mode={'multi-gpu' if multi_gpu else 'single-device'}"
        f"  n_actors={n_actors}  buffer_capacity={args.buffer_capacity}  batch_size={args.batch_size}"
    )

    for iteration in range(1, args.iterations + 1):
        rng, reset_key, sample_key = jax.random.split(rng, 3)
        states = jax.vmap(jax_reset)(jax.random.split(reset_key, n_actors))
        slots = [EpisodeSlot([], [], [], [], []) for _ in range(n_actors)]
        done_mask = np.zeros((n_actors,), dtype=bool)

        sched_depth, sched_sims, sched_temp = get_search_params(total_games_played)
        depth_limit = args.depth_limit if args.depth_limit is not None else sched_depth
        n_simulations = args.mcts_sims if args.mcts_sims is not None else sched_sims
        temperature = args.temperature if args.temperature is not None else sched_temp

        search_fn = functools.partial(
            jax_mcts_search,
            params=actor_params,
            adj=adj,
            model=model,
            n_simulations=n_simulations,
            depth_limit=depth_limit,
        )
        batched_search = jax.jit(jax.vmap(search_fn))

        actor_steps = 0
        while actor_steps < args.max_steps and not np.all(done_mask):
            active_players = states.active_player
            obs_batch, card_ctx_batch = batched_obs(states, active_players)
            mask_batch = batched_mask(states)
            policy_batch = batched_search(states)

            sample_key, action_key = jax.random.split(sample_key)
            action_keys = jax.random.split(action_key, n_actors)
            tempered = jnp.power(policy_batch, 1.0 / jnp.maximum(temperature, 1e-6))
            tempered = tempered / jnp.maximum(jnp.sum(tempered, axis=-1, keepdims=True), 1e-8)
            action_batch = jax.vmap(lambda p, k: jax.random.choice(k, N_ACTIONS, p=p))(tempered, action_keys)
            next_states, reward_batch, done_batch = batched_step(states, action_batch)

            obs_np = np.asarray(jax.device_get(obs_batch))
            ctx_np = np.asarray(jax.device_get(card_ctx_batch))
            mask_np = np.asarray(jax.device_get(mask_batch))
            policy_np = np.asarray(jax.device_get(policy_batch))
            action_np = np.asarray(jax.device_get(action_batch))
            next_states_host = jax.device_get(next_states)
            done_np = np.asarray(jax.device_get(done_batch))

            completed_games = 0
            for i in range(n_actors):
                if done_mask[i]:
                    continue
                slots[i].obs.append(obs_np[i])
                slots[i].card_ctx.append(ctx_np[i])
                slots[i].mask.append(mask_np[i])
                slots[i].policy.append(policy_np[i])
                slots[i].action.append(int(action_np[i]))

                if bool(done_np[i]) or actor_steps + 1 >= args.max_steps:
                    outcome = final_outcome_from_state(jax.tree_util.tree_map(lambda x: x[i], next_states_host))
                    count = len(slots[i].action)
                    replay.push(
                        obs=np.stack(slots[i].obs),
                        card_ctx=np.stack(slots[i].card_ctx),
                        mask=np.stack(slots[i].mask),
                        policy=np.stack(slots[i].policy),
                        action=np.asarray(slots[i].action, dtype=np.int32),
                        reward=np.full((count,), outcome, dtype=np.float32),
                        done=np.array([False] * (count - 1) + [True], dtype=bool),
                    )
                    done_mask[i] = True
                    completed_games += 1

            states = next_states
            actor_steps += 1
            total_games_played += completed_games

        metrics_rows = []
        if len(replay) >= args.min_buffer_size:
            for _ in range(args.learner_steps):
                rng, batch_key = jax.random.split(rng)
                batch = replay.sample(args.batch_size, batch_key)
                learner_state, metrics = learner_step(learner_state, batch)
                metrics_rows.append({k: float(v) for k, v in metrics.items()})
                learner_updates += 1
                if (not multi_gpu) or (learner_updates % args.sync_every == 0):
                    actor_params = learner_state.params

        metric_str = ""
        if metrics_rows:
            summary = {
                key: float(np.mean([row[key] for row in metrics_rows]))
                for key in metrics_rows[0]
            }
            metric_str = (
                f"  loss={summary['total_loss']:.4f}"
                f"  policy={summary['policy_loss']:.4f}"
                f"  value={summary['value_loss']:.4f}"
            )

        print(
            f"Iter {iteration:03d}  buffer={len(replay)}"
            f"  actor_steps={actor_steps}"
            f"  learner_updates={learner_updates}"
            f"  total_games={total_games_played}"
            f"  depth={depth_limit}"
            f"  sims={n_simulations}"
            f"  temp={temperature:.2f}{metric_str}"
        )

        if iteration % args.save_every == 0:
            out = Path(args.checkpoint_dir) / f"iter_{iteration:03d}.pkl"
            save_checkpoint(
                out,
                learner_state.params,
                {
                    "iteration": iteration,
                    "buffer_size": len(replay),
                    "learner_updates": learner_updates,
                    "total_games_played": total_games_played,
                    "bc_checkpoint": str(checkpoint_path),
                },
            )
            print(f"Saved checkpoint: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
