"""
network.py — JAX/Flax GATv2 neural network for Paths of Glory RL.

Architecture (all finalized):
  - 4 × GATv2Conv layers, 4 heads, hidden dim 64 → 256 after head concat
  - Global context (card embeddings + game flags) injected at every layer
  - Pre-LayerNorm residuals (no BatchNorm)
  - Factored 4-sub-head policy output
  - Value head: Linear(256→128) → ReLU → Linear(128→1) → tanh
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import json
import numpy as np
from typing import Tuple

N_SPACES = 72
N_PLANES = 32
N_CARDS  = 110
HIDDEN   = 256   # n_heads * head_dim = 4 * 64


# ─────────────────────────────────────────────
# GATv2 Attention Layer
# ─────────────────────────────────────────────

class GATv2Conv(nn.Module):
    """
    Single GATv2 graph attention layer.

    Attention formula:
        e_ij = W_a^T · LeakyReLU(W_l · h_i + W_r · h_j)
    Masking: non-edge pairs receive −1e9 before softmax.
    Output shape: (N_nodes, num_heads * out_features)
    """
    out_features: int   # per-head output dimension
    num_heads:    int   # number of attention heads

    @nn.compact
    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        # x:   (N, F_in)   — node features
        # adj: (N, N)       — binary float32 adjacency (1=edge, 0=no edge)
        N = x.shape[0]

        W_l = nn.Dense(self.num_heads * self.out_features, use_bias=False)
        W_r = nn.Dense(self.num_heads * self.out_features, use_bias=False)
        W_a = self.param("W_a", nn.initializers.glorot_uniform(),
                         (self.num_heads, self.out_features))

        h_l = W_l(x).reshape(N, self.num_heads, self.out_features)   # (N, H, F)
        h_r = W_r(x).reshape(N, self.num_heads, self.out_features)

        # GATv2: e_ij = W_a^T · LeakyReLU(h_l_i + h_r_j)
        h_l_exp = h_l[:, None, :, :]   # (N, 1, H, F)
        h_r_exp = h_r[None, :, :, :]   # (1, N, H, F)
        e = jax.nn.leaky_relu(h_l_exp + h_r_exp, negative_slope=0.2)  # (N, N, H, F)

        # Attention score per head
        scores = jnp.einsum("ijhf,hf->ijh", e, W_a)   # (N, N, H)

        # Mask non-edges
        adj_exp = adj[:, :, None]                      # (N, N, 1)
        scores  = jnp.where(adj_exp > 0, scores, -1e9)

        # Softmax over source (axis=1 = "which neighbor")
        attn = jax.nn.softmax(scores, axis=1)          # (N, N, H)

        # Aggregate: weighted sum of neighbour features
        out = jnp.einsum("ijh,jhf->ihf", attn, h_r)   # (N, H, F)
        return out.reshape(N, self.num_heads * self.out_features)


# ─────────────────────────────────────────────
# Main Network
# ─────────────────────────────────────────────

class PoGNet(nn.Module):
    """
    Paths of Glory policy/value network.

    Inputs:
      spatial_obs:  (batch, N_PLANES, N_SPACES) float32
      card_context: (batch, 7, 16)              float32
      adj:          (N_SPACES, N_SPACES)         static float32 — passed non-trained

    Outputs:
      policy: (card_logits, action_type_logits, src_logits, tgt_logits)
      value:  (batch, 1) float32 in [−1, 1]
    """
    n_spaces:       int = N_SPACES
    n_planes:       int = N_PLANES
    hidden_dim:     int = 64        # per-head dim; concat dim = hidden_dim * n_heads
    n_heads:        int = 4
    n_gat_layers:   int = 4
    card_slots:     int = 7
    card_embed_dim: int = 16

    @nn.compact
    def __call__(self,
                 spatial_obs:  jnp.ndarray,
                 card_context: jnp.ndarray,
                 adj:          jnp.ndarray,
                 training:     bool = False) -> Tuple:

        B    = spatial_obs.shape[0]
        hdim = self.hidden_dim * self.n_heads   # 256

        # ── Global context ──────────────────────────────────────────────
        # (B, 7, 16) → (B, 112) → Linear → (B, 256)
        ctx        = card_context.reshape(B, self.card_slots * self.card_embed_dim)
        global_ctx = nn.Dense(hdim)(ctx)
        global_ctx = nn.LayerNorm()(global_ctx)
        global_ctx = jax.nn.relu(global_ctx)

        # ── Input projection ────────────────────────────────────────────
        # (B, N_PLANES, N_SPACES) → (B, N_SPACES, N_PLANES) → (B, N_SPACES, hdim)
        x = spatial_obs.transpose(0, 2, 1)    # (B, N, P)
        x = nn.Dense(hdim)(x)                 # (B, N, 256)

        # ── 4 × GATv2 layers ───────────────────────────────────────────
        gat_layers = [
            GATv2Conv(out_features=self.hidden_dim, num_heads=self.n_heads)
            for _ in range(self.n_gat_layers)
        ]
        layer_norms_pre  = [nn.LayerNorm() for _ in range(self.n_gat_layers)]
        ctx_projections  = [nn.Dense(hdim)  for _ in range(self.n_gat_layers)]
        out_projections  = [nn.Dense(hdim)  for _ in range(self.n_gat_layers)]

        for gat, ln_pre, ctx_proj, out_proj in zip(
                gat_layers, layer_norms_pre, ctx_projections, out_projections):

            h = ln_pre(x)                                          # Pre-LN
            h = h + ctx_proj(global_ctx)[:, None, :]              # Context injection (B,1,256) broadcast
            h_gat = jax.vmap(lambda hi: gat(hi, adj))(h)          # (B, N, 256)
            h_gat = out_proj(h_gat)
            x     = x + h_gat                                     # Residual

        # ── Graph readout ───────────────────────────────────────────────
        graph_emb = jnp.mean(x, axis=1)    # (B, 256)

        # ── Factored 4-sub-head policy ──────────────────────────────────
        card_logits        = nn.Dense(N_CARDS)(graph_emb)           # (B, 110)
        action_type_logits = nn.Dense(3)(graph_emb)                 # (B, 3)
        src_logits         = nn.Dense(self.n_spaces)(graph_emb)     # (B, N_SPACES)
        tgt_logits         = nn.Dense(self.n_spaces)(graph_emb)     # (B, N_SPACES)

        # ── Value head ──────────────────────────────────────────────────
        v = nn.Dense(128)(graph_emb)
        v = jax.nn.relu(v)
        v = nn.Dense(1)(v)
        value = jnp.tanh(v)                # (B, 1)

        return (card_logits, action_type_logits, src_logits, tgt_logits), value


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Apply boolean mask then softmax.
    logits: (..., N), mask: (..., N) bool — True = legal action.
    Illegal positions receive −inf before softmax.
    """
    masked = jnp.where(mask, logits, -jnp.inf)
    return jax.nn.softmax(masked, axis=-1)


def load_adjacency_matrix(map_json_path: str) -> jnp.ndarray:
    """
    Load pog_map_graph.json → (N_SPACES, N_SPACES) float32 adjacency matrix.
    Handles both old format (list of dicts) and new format ({spaces: [...]}).
    """
    with open(map_json_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        spaces_raw = []
        for item in data:
            for name, info in item.items():
                spaces_raw.append({"id": name.upper().replace(" ", "_"), **info})
    else:
        spaces_raw = data.get("spaces", [])

    n        = len(spaces_raw)
    id_to_idx = {s["id"]: i for i, s in enumerate(spaces_raw)}
    adj = np.zeros((n, n), dtype=np.float32)

    for i, s in enumerate(spaces_raw):
        for conn in s.get("connections", []):
            cid = conn.upper().replace(" ", "_")
            if cid in id_to_idx:
                j = id_to_idx[cid]
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Pad to N_SPACES if needed
    if n < N_SPACES:
        full = np.zeros((N_SPACES, N_SPACES), dtype=np.float32)
        full[:n, :n] = adj
        adj = full

    return jnp.array(adj)


def create_train_state(rng:           jax.Array,
                       model:         PoGNet,
                       dummy_spatial: jnp.ndarray,
                       dummy_cards:   jnp.ndarray,
                       adj:           jnp.ndarray,
                       learning_rate: float = 1e-4) -> train_state.TrainState:
    """Initialise Flax TrainState with Adam optimiser."""
    params = model.init(rng, dummy_spatial, dummy_cards, adj)
    tx     = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("Loading adjacency matrix...")
    map_path = os.path.join(os.path.dirname(__file__), "../../pog_map_graph.json")
    adj      = load_adjacency_matrix(map_path)
    n_spaces = adj.shape[0]
    print(f"  adj shape: {adj.shape}, edges: {int(adj.sum()) // 2}")

    print("Initialising PoGNet...")
    rng   = jax.random.PRNGKey(42)
    model = PoGNet(n_spaces=n_spaces)

    dummy_spatial = jnp.zeros((2, N_PLANES, n_spaces))
    dummy_cards   = jnp.zeros((2, 7, 16))

    params = model.init(rng, dummy_spatial, dummy_cards, adj)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Parameter count: {n_params:,}")

    (c, a, s, t), v = model.apply(params, dummy_spatial, dummy_cards, adj)
    print(f"  card_logits:        {c.shape}")
    print(f"  action_type_logits: {a.shape}")
    print(f"  src_logits:         {s.shape}")
    print(f"  tgt_logits:         {t.shape}")
    print(f"  value:              {v.shape}")
    print("Smoke test PASSED.")
