"""
bc_pipeline.py — Behavioral Cloning pre-training pipeline for Paths of Glory.

3-phase anti-collapse curriculum:
  Phase 1 (epochs  1-10): Policy head only, value head frozen.         λ_v = 0.0
  Phase 2 (epochs 11-30): Policy + value head, partial value signal.   λ_v = 0.1
  Phase 3 (epochs 31+  ): Full AlphaZero self-play targets integrated. λ_v = 1.0

Rationale: Naively training the value head from sparse game outcomes during BC
causes it to collapse to predicting mean ≈ 0. Freezing it initially lets the
policy head develop meaningful features first; Phase 2 then bootstraps the
value head off those features.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import random
from typing import List, Iterator, Tuple, Dict, Optional

N_SPACES  = 72
N_CARDS   = 110
N_ACTIONS = 5341

# ─────────────────────────────────────────────
# Action space decomposition
#
# Flat layout:
#   [0]        PASS
#   [1:111]    EVENT   → card_idx = flat − 1
#   [111:441]  OPS     → card_idx = (flat−111)//3, op_type = (flat−111)%3
#   [441:5341] MOVE    → src = (flat−441)//N_SPACES, tgt = (flat−441)%N_SPACES
# ─────────────────────────────────────────────

def decompose_action(flat_action: int) -> Tuple[int, int, int, int]:
    """
    Map flat action index → (card_idx, action_type, src_space, tgt_space).

    action_type encoding:
      -1 = PASS
       0 = EVENT
       1 = OPS_MOVE
       2 = OPS_ATTACK
       3 = OPS_SR
       4 = MOVE_UNIT
    Unused fields are returned as 0.
    """
    if flat_action == 0:
        return (0, -1, 0, 0)
    elif flat_action < 111:
        return (flat_action - 1, 0, 0, 0)
    elif flat_action < 441:
        offset   = flat_action - 111
        card_idx = offset // 3
        op_type  = offset % 3          # 0=MOVE, 1=ATTACK, 2=SR
        return (card_idx, 1 + op_type, 0, 0)
    else:
        offset = flat_action - 441
        src    = offset // N_SPACES
        tgt    = offset % N_SPACES
        return (0, 4, src, tgt)


def compose_action(card_idx: int, action_type: int,
                   src_space: int, tgt_space: int) -> int:
    """
    Inverse of decompose_action.

    action_type: −1=PASS, 0=EVENT, 1=OPS_MOVE, 2=OPS_ATTACK, 3=OPS_SR, 4=MOVE_UNIT
    """
    if action_type == -1:
        return 0
    elif action_type == 0:
        return 1 + card_idx
    elif 1 <= action_type <= 3:
        op_type = action_type - 1
        return 111 + card_idx * 3 + op_type
    elif action_type == 4:
        return 441 + src_space * N_SPACES + tgt_space
    else:
        raise ValueError(f"Unknown action_type {action_type}")


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_expert_games(jsonl_path: str) -> List[Dict]:
    """
    Load JSONL expert game records. One JSON object per line.

    Required fields per record:
      game_id:      str
      turn:         int
      action_round: int
      player:       "AP" | "CP"
      obs_tensor:   list[list[float]]  shape (32, N_SPACES)
      legal_mask:   list[bool]         shape (N_ACTIONS,)
      action_taken: int
      outcome:      int  {−1, 0, 1}

    Optional:
      card_context: list[list[float]]  shape (7, 16)
    """
    records: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_bc_batches(
    records:    List[Dict],
    batch_size: int = 32,
    shuffle:    bool = True,
) -> Iterator[Tuple[jnp.ndarray, ...]]:
    """
    Yield (spatial_obs, card_context, legal_mask, action_taken, outcome) batches.

    Shapes:
      spatial_obs:  (B, 32, N_SPACES) float32
      card_context: (B, 7, 16)        float32
      legal_mask:   (B, N_ACTIONS)    bool
      action_taken: (B,)              int32
      outcome:      (B,)              float32
    """
    if shuffle:
        records = records[:]
        random.shuffle(records)

    for start in range(0, len(records), batch_size):
        batch = records[start: start + batch_size]
        if not batch:
            break

        obs_list, ctx_list, mask_list, act_list, out_list = [], [], [], [], []
        for rec in batch:
            obs  = np.array(rec["obs_tensor"],  dtype=np.float32)
            ctx  = np.array(rec.get("card_context", np.zeros((7, 16))), dtype=np.float32)
            mask = np.array(rec["legal_mask"],  dtype=bool)
            act  = int(rec["action_taken"])
            out  = float(rec["outcome"])
            obs_list.append(obs);  ctx_list.append(ctx)
            mask_list.append(mask); act_list.append(act); out_list.append(out)

        yield (
            jnp.array(np.stack(obs_list)),
            jnp.array(np.stack(ctx_list)),
            jnp.array(np.stack(mask_list)),
            jnp.array(act_list,  dtype=jnp.int32),
            jnp.array(out_list,  dtype=jnp.float32),
        )


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

def _policy_loss(policy_heads: Tuple, action_taken: jnp.ndarray) -> jnp.ndarray:
    """
    Cross-entropy loss for factored 4-sub-head policy.
    Decomposes each flat action into (card, action_type, src, tgt) and
    applies CE loss to each sub-head independently.
    """
    card_logits, atype_logits, src_logits, tgt_logits = policy_heads

    # Vectorised decomposition (pure jnp — no Python branching)
    f = action_taken.astype(jnp.int32)

    card_tgt  = jnp.where(f < 111, f - 1,
                jnp.where(f < 441, (f - 111) // 3, 0))
    atype_tgt = jnp.where(f == 0,  0,
                jnp.where(f < 111, 0,
                jnp.where(f < 441, (f - 111) % 3, 3)))
    src_tgt   = jnp.where(f >= 441, (f - 441) // N_SPACES, 0)
    tgt_tgt   = jnp.where(f >= 441, (f - 441) % N_SPACES,  0)

    def ce(logits: jnp.ndarray, targets: jnp.ndarray, n_classes: int) -> jnp.ndarray:
        one_hot   = jax.nn.one_hot(targets, n_classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    return (
        ce(card_logits,   card_tgt,  N_CARDS)  +
        ce(atype_logits,  atype_tgt, 3)        +
        ce(src_logits,    src_tgt,   N_SPACES) +
        ce(tgt_logits,    tgt_tgt,   N_SPACES)
    ) / 4.0


def bc_loss_phase1(params, batch, model, adj):
    """Phase 1: policy cross-entropy only (value head ignored)."""
    spatial_obs, card_context, legal_mask, action_taken, outcome = batch
    policy_heads, _ = model.apply(params, spatial_obs, card_context, adj)
    return _policy_loss(policy_heads, action_taken)


def bc_loss_phase2(params, batch, model, adj, lambda_v: float = 0.1):
    """Phase 2: policy + λ_v × MSE(value, discounted outcome)."""
    spatial_obs, card_context, legal_mask, action_taken, outcome = batch
    policy_heads, value = model.apply(params, spatial_obs, card_context, adj)
    policy_l = _policy_loss(policy_heads, action_taken)
    value_l  = jnp.mean((value[:, 0] - outcome) ** 2)
    return policy_l + lambda_v * value_l


def bc_loss_phase3(params, batch, model, adj, lambda_v: float = 1.0):
    """Phase 3: full targets — same as Phase 2 but λ_v=1.0 for MCTS integration."""
    return bc_loss_phase2(params, batch, model, adj, lambda_v=lambda_v)


def batch_component_accuracy(params, batch, model, adj) -> float:
    """
    Mean per-component accuracy for the factored policy heads.
    This is a smoke-test metric, not a true flat-action Top-1 score.
    """
    spatial_obs, card_context, legal_mask, action_taken, outcome = batch
    (card_logits, atype_logits, src_logits, tgt_logits), _ = model.apply(
        params, spatial_obs, card_context, adj)

    pred_card  = np.array(jnp.argmax(card_logits, axis=-1), dtype=np.int32)
    pred_atype = np.array(jnp.argmax(atype_logits, axis=-1), dtype=np.int32)
    pred_src   = np.array(jnp.argmax(src_logits, axis=-1), dtype=np.int32)
    pred_tgt   = np.array(jnp.argmax(tgt_logits, axis=-1), dtype=np.int32)

    truth = np.array(action_taken, dtype=np.int32)
    card_tgt  = np.where(
        truth == 0, 0,
        np.where(truth < 111, truth - 1, np.where(truth < 441, (truth - 111) // 3, 0))
    )
    atype_tgt = np.where(truth == 0, 0, np.where(truth < 111, 0, np.where(truth < 441, (truth - 111) % 3, 3)))
    src_tgt   = np.where(truth >= 441, (truth - 441) // N_SPACES, 0)
    tgt_tgt   = np.where(truth >= 441, (truth - 441) % N_SPACES, 0)

    card_acc  = np.mean(pred_card == card_tgt)
    atype_acc = np.mean(pred_atype == np.clip(atype_tgt, 0, 2))
    src_acc   = np.mean(pred_src == src_tgt)
    tgt_acc   = np.mean(pred_tgt == tgt_tgt)
    return float(np.mean([card_acc, atype_acc, src_acc, tgt_acc]))


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train_bc(
    model,
    train_records:  List[Dict],
    adj:            jnp.ndarray,
    n_epochs:       int   = 50,
    batch_size:     int   = 32,
    learning_rate:  float = 1e-4,
    rng_seed:       int   = 0,
    val_records:    Optional[List[Dict]] = None,
) -> dict:
    """
    Full 3-phase BC training loop.

    Args:
        model:         PoGNet instance (uninitialised)
        train_records: list of expert game record dicts
        adj:           (N_SPACES, N_SPACES) static adjacency matrix
        n_epochs:      total epochs (phases: 1-10, 11-30, 31+)
        batch_size:    batch size
        learning_rate: Adam LR
        rng_seed:      PRNG seed
        val_records:   optional validation set for loss reporting

    Returns:
        final trained params dict
    """
    import optax
    from flax.training import train_state as flax_ts

    rng           = jax.random.PRNGKey(rng_seed)
    dummy_spatial = jnp.zeros((1, 32, N_SPACES))
    dummy_cards   = jnp.zeros((1, 7, 16))
    params        = model.init(rng, dummy_spatial, dummy_cards, adj)
    tx            = optax.adam(learning_rate)
    state         = flax_ts.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def step_p1(st, b):
        def loss_fn(params):
            return bc_loss_phase1(params, b, model, adj)
        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads), loss

    @jax.jit
    def step_p2(st, b):
        def loss_fn(params):
            return bc_loss_phase2(params, b, model, adj)
        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads), loss

    @jax.jit
    def step_p3(st, b):
        def loss_fn(params):
            return bc_loss_phase3(params, b, model, adj)
        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads), loss

    for epoch in range(1, n_epochs + 1):
        if epoch <= 10:
            phase, step_fn = 1, step_p1
        elif epoch <= 30:
            phase, step_fn = 2, step_p2
        else:
            phase, step_fn = 3, step_p3

        losses = []
        accs   = []
        for batch in make_bc_batches(train_records, batch_size=batch_size):
            state, loss = step_fn(state, batch)
            losses.append(float(loss))
            accs.append(batch_component_accuracy(state.params, batch, model, adj))

        val_str = ""
        if val_records and losses:
            val_losses = []
            val_accs   = []
            for batch in make_bc_batches(val_records, batch_size=batch_size, shuffle=False):
                val_losses.append(float(bc_loss_phase2(state.params, batch, model, adj)))
                val_accs.append(batch_component_accuracy(state.params, batch, model, adj))
            val_str = (
                f"  val_loss={np.mean(val_losses):.4f}"
                f"  val_component_acc={np.mean(val_accs):.3f}"
            )

        if losses:
            print(f"Epoch {epoch:3d} [Phase {phase}]  "
                  f"train_loss={np.mean(losses):.4f}"
                  f"  train_component_acc={np.mean(accs):.3f}{val_str}")

    return state.params
