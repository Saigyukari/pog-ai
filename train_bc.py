#!/usr/bin/env python3
"""
Behavioral cloning trainer for Paths of Glory.

Implements the roadmap Phase 1 launcher:
  - loads expert JSONL data
  - initialises PoGNet via create_train_state()
  - runs the 3-phase BC curriculum
  - uses pmap when multiple local devices are available
  - logs policy loss, value loss, and flat top-1 action accuracy
  - saves checkpoints every N epochs to checkpoints/bc/
"""

from __future__ import annotations

import argparse
import functools
import math
import pickle
import random
from pathlib import Path
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training import train_state

from src.rl.bc_pipeline import (
    bc_loss_phase1,
    bc_loss_phase2,
    bc_loss_phase3,
    load_expert_games,
    make_bc_batches,
)
from src.rl.network import PoGNet, create_train_state, load_adjacency_matrix

N_SPACES = 72
N_ACTIONS = 5341


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PoGNet with behavioral cloning.")
    parser.add_argument(
        "--data",
        default="data/training/expert_games.jsonl",
        help="Path to expert JSONL records.",
    )
    parser.add_argument(
        "--map",
        dest="map_json",
        default="pog_map_graph.json",
        help="Path to pog_map_graph.json.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of BC epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Global batch size before device sharding.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation split fraction in [0, 1).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/bc",
        help="Directory for periodic checkpoints.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Checkpoint frequency in epochs.",
    )
    return parser.parse_args()


def split_records(records: Sequence[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    items = list(records)
    if not items:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) == 1 or val_frac <= 0:
        return items, []

    n_val = max(1, int(round(len(items) * val_frac)))
    n_val = min(n_val, len(items) - 1)
    return items[n_val:], items[:n_val]


def phase_for_epoch(epoch: int) -> int:
    if epoch <= 10:
        return 1
    if epoch <= 30:
        return 2
    return 3


def _value_loss(value: jnp.ndarray, outcome: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((value[:, 0] - outcome) ** 2)


def _flat_action_scores(policy_heads: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    card_logits, atype_logits, src_logits, tgt_logits = policy_heads

    batch = card_logits.shape[0]
    scores = jnp.full((batch, N_ACTIONS), -1e9, dtype=card_logits.dtype)

    # PASS uses a neutral baseline score. It is only selected when all modeled
    # actions are worse after masking.
    scores = scores.at[:, 0].set(0.0)

    # EVENT: 1 + card_idx
    scores = scores.at[:, 1:111].set(card_logits + atype_logits[:, 0:1])

    # OPS: 111 + card_idx * 3 + op_type
    ops_scores = card_logits[:, :, None] + atype_logits[:, None, :]
    scores = scores.at[:, 111:441].set(ops_scores.reshape(batch, 330))

    # MOVE_UNIT: 441 + src * 72 + tgt
    move_scores = src_logits[:, :, None] + tgt_logits[:, None, :]
    scores = scores.at[:, 441:].set(move_scores.reshape(batch, N_SPACES * N_SPACES)[:, : N_ACTIONS - 441])

    return scores


def _batch_metrics(params, batch: tuple[jnp.ndarray, ...], model: PoGNet, adj: jnp.ndarray) -> dict[str, jnp.ndarray]:
    spatial_obs, card_context, legal_mask, action_taken, outcome = batch
    policy_heads, value = model.apply(params, spatial_obs, card_context, adj)

    policy_loss = bc_loss_phase1(params, batch, model, adj)
    value_loss = _value_loss(value, outcome)

    masked_scores = jnp.where(legal_mask, _flat_action_scores(policy_heads), -1e9)
    pred_action = jnp.argmax(masked_scores, axis=-1)
    accuracy = jnp.mean(pred_action == action_taken)

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "accuracy": accuracy,
    }


def _reshape_for_pmap(array: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    batch_size = array.shape[0]
    remainder = batch_size % n_devices
    if remainder:
        pad = n_devices - remainder
        pad_block = jnp.repeat(array[-1:], pad, axis=0)
        array = jnp.concatenate([array, pad_block], axis=0)
    per_device = array.shape[0] // n_devices
    return array.reshape((n_devices, per_device) + array.shape[1:])


def shard_batch(batch: tuple[jnp.ndarray, ...], n_devices: int) -> tuple[jnp.ndarray, ...]:
    return tuple(_reshape_for_pmap(x, n_devices) for x in batch)


def unreplicate_params(state_or_params):
    return jax_utils.unreplicate(state_or_params)


def checkpoint_payload(
    params,
    epoch: int,
    args: argparse.Namespace,
    train_size: int,
    val_size: int,
) -> dict:
    return {
        "params": jax.device_get(params),
        "metadata": {
            "epoch": epoch,
            "data": args.data,
            "map_json": args.map_json,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "train_size": train_size,
            "val_size": val_size,
        },
    }


def save_checkpoint(
    checkpoint_dir: Path,
    params,
    epoch: int,
    args: argparse.Namespace,
    train_size: int,
    val_size: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch_{epoch:03d}.pkl"
    payload = checkpoint_payload(params, epoch, args, train_size, val_size)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def build_train_step_single(model: PoGNet, adj: jnp.ndarray):
    @jax.jit
    def train_step(state: train_state.TrainState, batch: tuple[jnp.ndarray, ...], phase_index: jnp.ndarray):
        def loss_fn(params):
            loss = jax.lax.switch(
                phase_index,
                (
                    lambda p: bc_loss_phase1(p, batch, model, adj),
                    lambda p: bc_loss_phase2(p, batch, model, adj),
                    lambda p: bc_loss_phase3(p, batch, model, adj),
                ),
                params,
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        metrics = _batch_metrics(new_state.params, batch, model, adj)
        metrics["train_loss"] = loss
        return new_state, metrics

    return train_step


def build_eval_step_single(model: PoGNet, adj: jnp.ndarray):
    @jax.jit
    def eval_step(params, batch: tuple[jnp.ndarray, ...]):
        metrics = _batch_metrics(params, batch, model, adj)
        metrics["train_loss"] = bc_loss_phase2(params, batch, model, adj)
        return metrics

    return eval_step


def build_train_step_pmap(model: PoGNet, adj: jnp.ndarray):
    @functools.partial(jax.pmap, axis_name="devices")
    def train_step(state: train_state.TrainState, batch: tuple[jnp.ndarray, ...], phase: jnp.ndarray):
        def loss_fn(params):
            loss = jax.lax.switch(
                phase,
                (
                    lambda p: bc_loss_phase1(p, batch, model, adj),
                    lambda p: bc_loss_phase2(p, batch, model, adj),
                    lambda p: bc_loss_phase3(p, batch, model, adj),
                ),
                params,
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        new_state = state.apply_gradients(grads=grads)

        metrics = _batch_metrics(new_state.params, batch, model, adj)
        metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="devices"), metrics)
        metrics["train_loss"] = loss
        return new_state, metrics

    return train_step


def build_eval_step_pmap(model: PoGNet, adj: jnp.ndarray):
    @functools.partial(jax.pmap, axis_name="devices")
    def eval_step(params, batch: tuple[jnp.ndarray, ...]):
        metrics = _batch_metrics(params, batch, model, adj)
        metrics["train_loss"] = bc_loss_phase2(params, batch, model, adj)
        return jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="devices"), metrics)

    return eval_step


def collect_epoch_metrics(
    metric_dicts: Iterable[dict[str, jnp.ndarray]],
    *,
    pmapped: bool,
) -> dict[str, float]:
    rows = []
    for metrics in metric_dicts:
        if pmapped:
            row = {k: float(np.asarray(v)[0]) for k, v in metrics.items()}
        else:
            row = {k: float(v) for k, v in metrics.items()}
        rows.append(row)

    if not rows:
        return {
            "train_loss": math.nan,
            "policy_loss": math.nan,
            "value_loss": math.nan,
            "accuracy": math.nan,
        }

    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> int:
    args = parse_args()

    data_path = Path(args.data)
    map_path = Path(args.map_json)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not map_path.is_file():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    records = load_expert_games(str(data_path))
    if not records:
        raise ValueError(f"No records found in {data_path}")

    train_records, val_records = split_records(records, args.val_frac, args.seed)
    adj = load_adjacency_matrix(str(map_path))
    model = PoGNet(hidden_dim=128, n_gat_layers=6)

    devices = jax.local_devices()
    n_devices = len(devices)
    use_pmap = n_devices > 1

    print(f"Devices: {devices}")
    print(f"Loaded {len(records)} records from {data_path}")
    print(f"Train records: {len(train_records)}")
    print(f"Val records:   {len(val_records)}")
    print(
        f"Training mode: {'pmap' if use_pmap else 'single-device'} "
        f"across {n_devices} local device(s)"
    )

    dummy_spatial = jnp.zeros((1, 32, N_SPACES), dtype=jnp.float32)
    dummy_cards = jnp.zeros((1, 7, 16), dtype=jnp.float32)
    state = create_train_state(
        rng=jax.random.PRNGKey(args.seed),
        model=model,
        dummy_spatial=dummy_spatial,
        dummy_cards=dummy_cards,
        adj=adj,
        learning_rate=args.lr,
    )

    if use_pmap:
        state = jax_utils.replicate(state, devices=devices)
        train_step = build_train_step_pmap(model, adj)
        eval_step = build_eval_step_pmap(model, adj)
    else:
        train_step = build_train_step_single(model, adj)
        eval_step = build_eval_step_single(model, adj)

    checkpoint_dir = Path(args.checkpoint_dir)

    for epoch in range(1, args.epochs + 1):
        phase = phase_for_epoch(epoch)
        train_metrics = []

        for batch in make_bc_batches(train_records, batch_size=args.batch_size, shuffle=True):
            if use_pmap:
                sharded_batch = shard_batch(batch, n_devices)
                phase_index = jnp.full((n_devices,), phase - 1, dtype=jnp.int32)
                state, metrics = train_step(state, sharded_batch, phase_index)
            else:
                state, metrics = train_step(state, batch, jnp.asarray(phase - 1, dtype=jnp.int32))
            train_metrics.append(metrics)

        train_summary = collect_epoch_metrics(train_metrics, pmapped=use_pmap)

        val_summary = None
        if val_records:
            val_metrics = []
            for batch in make_bc_batches(val_records, batch_size=args.batch_size, shuffle=False):
                if use_pmap:
                    metrics = eval_step(state.params, shard_batch(batch, n_devices))
                else:
                    metrics = eval_step(state.params, batch)
                val_metrics.append(metrics)
            val_summary = collect_epoch_metrics(val_metrics, pmapped=use_pmap)

        line = (
            f"Epoch {epoch:03d} [Phase {phase}] "
            f"policy_loss={train_summary['policy_loss']:.4f} "
            f"value_loss={train_summary['value_loss']:.4f} "
            f"top1_acc={train_summary['accuracy']:.4f}"
        )
        if val_summary:
            line += (
                f" | val_policy_loss={val_summary['policy_loss']:.4f} "
                f"val_value_loss={val_summary['value_loss']:.4f} "
                f"val_top1_acc={val_summary['accuracy']:.4f}"
            )
        print(line)

        if epoch % args.save_every == 0:
            params = unreplicate_params(state).params if use_pmap else state.params
            path = save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                params=params,
                epoch=epoch,
                args=args,
                train_size=len(train_records),
                val_size=len(val_records),
            )
            print(f"Saved checkpoint: {path}")

    final_epoch = args.epochs
    if final_epoch % args.save_every != 0:
        params = unreplicate_params(state).params if use_pmap else state.params
        path = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            params=params,
            epoch=final_epoch,
            args=args,
            train_size=len(train_records),
            val_size=len(val_records),
        )
        print(f"Saved final checkpoint: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
