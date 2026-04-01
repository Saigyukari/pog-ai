"""
mcts.py — JAX-compatible Monte Carlo Tree Search for Paths of Glory.

Design constraints:
  - ALL arrays pre-allocated at startup (static shapes for jit)
  - Simulation loop uses jax.lax.fori_loop (no Python-level loop under jit)
  - Env stepping uses jax.pure_callback (Python env stays outside jit scope)
  - Tree capacity: MAX_NODES = 512 nodes per search

Tree arrays (all shape [MAX_NODES, ...]):
  N  [MAX_NODES, N_ACTIONS]  int32   visit counts
  W  [MAX_NODES, N_ACTIONS]  float32 total action values
  P  [MAX_NODES, N_ACTIONS]  float32 prior policy from network
  children [MAX_NODES, N_ACTIONS] int32  child node index (-1=unexpanded)
  parent   [MAX_NODES]       int32  parent node index (-1=root)
  is_terminal [MAX_NODES]    bool
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Callable, Optional

from src.env.jax_env import jax_legal_mask, jax_obs, jax_step

N_ACTIONS = 5341
MAX_NODES = 512
C_PUCT    = 1.5      # exploration constant


class MCTSTree(NamedTuple):
    """Immutable JAX-compatible MCTS tree state."""
    N:           jnp.ndarray   # (MAX_NODES, N_ACTIONS) int32
    W:           jnp.ndarray   # (MAX_NODES, N_ACTIONS) float32 — total values
    P:           jnp.ndarray   # (MAX_NODES, N_ACTIONS) float32 — priors
    children:    jnp.ndarray   # (MAX_NODES, N_ACTIONS) int32 — child idx or -1
    parent:      jnp.ndarray   # (MAX_NODES,)           int32
    node_count:  jnp.ndarray   # () int32 — number of allocated nodes


class RootSearchState(NamedTuple):
    visits: jnp.ndarray
    values: jnp.ndarray
    prior: jnp.ndarray
    rng_key: jnp.ndarray


def create_tree() -> MCTSTree:
    """Allocate a zeroed MCTS tree."""
    return MCTSTree(
        N          = jnp.zeros((MAX_NODES, N_ACTIONS), dtype=jnp.int32),
        W          = jnp.zeros((MAX_NODES, N_ACTIONS), dtype=jnp.float32),
        P          = jnp.zeros((MAX_NODES, N_ACTIONS), dtype=jnp.float32),
        children   = jnp.full((MAX_NODES, N_ACTIONS), -1, dtype=jnp.int32),
        parent     = jnp.full((MAX_NODES,), -1, dtype=jnp.int32),
        node_count = jnp.array(1, dtype=jnp.int32),   # root = node 0
    )


def reset_tree(tree: MCTSTree) -> MCTSTree:
    """Reset tree for a new root position (reuse allocated memory)."""
    return MCTSTree(
        N          = jnp.zeros_like(tree.N),
        W          = jnp.zeros_like(tree.W),
        P          = jnp.zeros_like(tree.P),
        children   = jnp.full_like(tree.children, -1),
        parent     = jnp.full_like(tree.parent, -1),
        node_count = jnp.array(1, dtype=jnp.int32),
    )


# ─────────────────────────────────────────────────────────────────────
# UCB scoring (pure JAX — jit-able)
# ─────────────────────────────────────────────────────────────────────

@jax.jit
def ucb_scores(tree: MCTSTree, node: int, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Compute PUCT scores for all actions from a node.

    PUCT(a) = Q(a) + C_PUCT * P(a) * sqrt(sum_N) / (1 + N(a))
    Illegal actions receive -inf.
    """
    n_node = tree.N[node]                          # (N_ACTIONS,) visits per action
    w_node = tree.W[node]                          # (N_ACTIONS,) total value
    p_node = tree.P[node]                          # (N_ACTIONS,) priors

    n_total = jnp.sum(n_node).astype(jnp.float32)
    q = jnp.where(n_node > 0, w_node / n_node.astype(jnp.float32), 0.0)
    u = C_PUCT * p_node * jnp.sqrt(n_total + 1) / (1.0 + n_node.astype(jnp.float32))

    scores = q + u
    return jnp.where(legal_mask, scores, -jnp.inf)


# ─────────────────────────────────────────────────────────────────────
# MCTS search (Python-level loop — wraps JAX ops)
# Using Python loop here for clarity; can wrap with lax.fori_loop for
# full jit if env is replaced with a JAX-native state machine.
# ─────────────────────────────────────────────────────────────────────

class MCTS:
    """
    MCTS search object.

    Usage:
        mcts = MCTS(model, params, adj, n_simulations=400)
        probs = mcts.search(env, rng_key)
        action = np.random.choice(len(probs), p=probs)
    """

    def __init__(self, model, params, adj: jnp.ndarray,
                 n_simulations: int = 400,
                 depth_limit: int = 4,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_eps:   float = 0.25,
                 temperature:     float = 1.0):
        self.model           = model
        self.params          = params
        self.adj             = adj
        self.n_simulations   = n_simulations
        self.depth_limit     = depth_limit
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps   = dirichlet_eps
        self.temperature     = temperature
        self.tree            = create_tree()

    def _network_eval(self, env, rng):
        """Run network forward pass on current env state."""
        obs       = env._build_spatial_obs()
        card_ctx  = env._build_card_context(env.active_player)
        legal     = env.action_mask(env._agent_name(env.active_player))

        spatial_b   = jnp.array(obs)[None]        # (1, 32, N_SPACES)
        card_ctx_b  = jnp.array(card_ctx)[None]   # (1, 7, 16)

        (card_l, atype_l, src_l, tgt_l), value = self.model.apply(
            self.params, spatial_b, card_ctx_b, self.adj)

        # Combine factored sub-head logits into flat N_ACTIONS prior
        # Simple heuristic: weight flat-to-factored mapping
        prior = np.zeros(N_ACTIONS, dtype=np.float32)
        prior[0] = 0.01   # PASS always gets small prior

        # Events [1:111]: card logits (softmaxed)
        card_probs = np.array(jax.nn.softmax(card_l[0]))
        for i in range(min(110, card_probs.shape[0])):
            prior[1 + i] = card_probs[i]

        # OPS [111:441]: card × type
        atype_probs = np.array(jax.nn.softmax(atype_l[0]))
        for ci in range(min(110, card_probs.shape[0])):
            for ot in range(3):
                flat = 111 + ci * 3 + ot
                if flat < 441:
                    prior[flat] = card_probs[ci] * atype_probs[ot]

        # MOVE [441:5341]: src × tgt
        src_probs = np.array(jax.nn.softmax(src_l[0]))
        tgt_probs = np.array(jax.nn.softmax(tgt_l[0]))
        for si in range(src_probs.shape[0]):
            for ti in range(tgt_probs.shape[0]):
                flat = 441 + si * src_probs.shape[0] + ti
                if flat < N_ACTIONS:
                    prior[flat] = src_probs[si] * tgt_probs[ti]

        # Mask and renormalise
        prior = prior * legal.astype(np.float32)
        total = prior.sum()
        if total > 0:
            prior /= total

        v = float(value[0, 0])
        return prior, v, legal

    def search(self, env, rng_key: jnp.ndarray) -> np.ndarray:
        """
        Run MCTS from current env state.
        Returns visit-count policy vector shape (N_ACTIONS,).
        """
        import copy

        self.tree = reset_tree(self.tree)

        # Evaluate root
        root_prior, root_value, root_legal = self._network_eval(env, rng_key)

        # Add Dirichlet noise to root prior for exploration
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * int(root_legal.sum()))
        noisy_prior = root_prior.copy()
        legal_idxs  = np.where(root_legal)[0]
        noisy_prior[legal_idxs] = (
            (1 - self.dirichlet_eps) * root_prior[legal_idxs] +
            self.dirichlet_eps * noise
        )

        # Store root priors
        self.tree = self.tree._replace(
            P=self.tree.P.at[0].set(jnp.array(noisy_prior))
        )

        for sim in range(self.n_simulations):
            sim_env   = copy.deepcopy(env)
            node      = 0
            depth     = 0
            path: list = []   # [(node, action)]

            # ── Selection ─────────────────────────────────────────
            while True:
                if depth >= self.depth_limit:
                    break
                legal = sim_env.action_mask(sim_env._agent_name(sim_env.active_player))
                scores = np.array(ucb_scores(self.tree, node, jnp.array(legal)))
                action = int(np.argmax(scores))

                path.append((node, action))
                child = int(self.tree.children[node, action])

                if child == -1:
                    break   # unexpanded leaf

                node = child
                try:
                    _, _, done, _, _ = sim_env.step(action)
                    depth += 1
                    if any(done.values()):
                        break
                except Exception:
                    break

            # ── Expansion ─────────────────────────────────────────
            prior, value, _ = self._network_eval(sim_env, rng_key)

            nc = int(self.tree.node_count)
            if nc < MAX_NODES:
                # Expand new node
                self.tree = self.tree._replace(
                    P          = self.tree.P.at[nc].set(jnp.array(prior)),
                    parent     = self.tree.parent.at[nc].set(path[-1][0]),
                    node_count = jnp.array(nc + 1, dtype=jnp.int32),
                )
                parent_node, parent_action = path[-1]
                self.tree = self.tree._replace(
                    children=self.tree.children.at[parent_node, parent_action].set(nc)
                )

            # ── Backup ────────────────────────────────────────────
            for bnode, baction in reversed(path):
                self.tree = self.tree._replace(
                    N=self.tree.N.at[bnode, baction].add(1),
                    W=self.tree.W.at[bnode, baction].add(value),
                )
                value = -value   # alternating players

        # Return visit-count policy
        root_visits = np.array(self.tree.N[0], dtype=np.float32)
        if self.temperature == 0:
            policy = np.zeros(N_ACTIONS)
            policy[int(np.argmax(root_visits))] = 1.0
        else:
            root_visits = root_visits ** (1.0 / self.temperature)
            total = root_visits.sum()
            policy = root_visits / total if total > 0 else root_visits
        return policy


def _flat_action_scores_jax(policy_heads) -> jnp.ndarray:
    card_logits, atype_logits, src_logits, tgt_logits = policy_heads
    batch = card_logits.shape[0]

    scores = jnp.full((batch, N_ACTIONS), -1e9, dtype=card_logits.dtype)
    scores = scores.at[:, 0].set(0.0)
    scores = scores.at[:, 1:111].set(card_logits + atype_logits[:, 0:1])

    ops_scores = card_logits[:, :, None] + atype_logits[:, None, :]
    scores = scores.at[:, 111:441].set(ops_scores.reshape(batch, 330))

    move_scores = src_logits[:, :, None] + tgt_logits[:, None, :]
    scores = scores.at[:, 441:].set(move_scores.reshape(batch, -1)[:, : N_ACTIONS - 441])
    return scores


def _dirichlet_noise_for_legal(
    rng_key: jnp.ndarray,
    legal_mask: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    gamma = jax.random.gamma(rng_key, alpha, shape=(N_ACTIONS,))
    masked = gamma * legal_mask.astype(jnp.float32)
    total = jnp.sum(masked)
    fallback = legal_mask.astype(jnp.float32) / jnp.maximum(1.0, jnp.sum(legal_mask.astype(jnp.float32)))
    return jnp.where(total > 0, masked / total, fallback)


def _network_eval_jax(state, params, adj: jnp.ndarray, model) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    obs, card_ctx = jax_obs(state, state.active_player)
    legal = jax_legal_mask(state)
    policy_heads, value = model.apply(params, obs[None, ...], card_ctx[None, ...], adj)
    scores = _flat_action_scores_jax(policy_heads)[0]
    masked = jnp.where(legal, scores, -1e9)
    probs = jax.nn.softmax(masked, axis=-1)
    probs = probs * legal.astype(jnp.float32)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
    return probs, value[0, 0], legal


def _rollout_value(state, params, adj: jnp.ndarray, model, depth_limit: int) -> jnp.ndarray:
    """
    Truncated leaf evaluator for JAX MCTS.

    This is intentionally lightweight: it advances up to `depth_limit - 1` greedy
    plies using the JAX env and returns the terminal/leaf network value translated
    back into the root player's frame.
    """
    root_player = state.active_player

    def body(_, carry):
        curr_state, done = carry

        def do_step(step_state):
            prior, _, legal = _network_eval_jax(step_state, params, adj, model)
            masked = jnp.where(legal, prior, -1.0)
            action = jnp.argmax(masked)
            next_state, _, step_done = jax_step(step_state, action.astype(jnp.int32))
            return next_state, step_done

        next_state, next_done = jax.lax.cond(done, lambda s: (s, done), do_step, curr_state)
        return next_state, next_done

    leaf_state, _ = jax.lax.fori_loop(
        0,
        jnp.maximum(depth_limit - 1, 0),
        body,
        (state, jnp.asarray(False)),
    )
    _, leaf_value, _ = _network_eval_jax(leaf_state, params, adj, model)
    same_player = leaf_state.active_player == root_player
    return jnp.where(same_player, leaf_value, -leaf_value)


def jax_mcts_search(
    state,
    params,
    adj: jnp.ndarray,
    model,
    n_simulations: int = 128,
    depth_limit: int = 4,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> jnp.ndarray:
    """
    Pure-JAX root-search MCTS variant for vmapped actor usage.

    The existing `MCTS` class remains the CPU/Python implementation. This helper
    is additive and uses the JAX env (`jax_step`) so it can be wrapped by
    `jax.vmap` for batched actor rollouts.
    """
    root_prior, _, root_legal = _network_eval_jax(state, params, adj, model)
    noise_key, sim_key = jax.random.split(state.rng_key)
    noise = _dirichlet_noise_for_legal(noise_key, root_legal, dirichlet_alpha)
    prior = ((1.0 - dirichlet_eps) * root_prior + dirichlet_eps * noise) * root_legal.astype(jnp.float32)
    prior = prior / jnp.maximum(jnp.sum(prior), 1e-8)

    init = RootSearchState(
        visits=jnp.zeros((N_ACTIONS,), dtype=jnp.int32),
        values=jnp.zeros((N_ACTIONS,), dtype=jnp.float32),
        prior=prior,
        rng_key=sim_key,
    )

    def sim_body(_, carry):
        visits, values, sim_prior, rng_key = carry
        total_n = jnp.sum(visits).astype(jnp.float32)
        q = jnp.where(visits > 0, values / visits.astype(jnp.float32), 0.0)
        u = C_PUCT * sim_prior * jnp.sqrt(total_n + 1.0) / (1.0 + visits.astype(jnp.float32))
        scores = jnp.where(root_legal, q + u, -jnp.inf)
        action = jnp.argmax(scores).astype(jnp.int32)

        rng_key, step_key = jax.random.split(rng_key)
        stepped_state = state._replace(rng_key=step_key)
        next_state, _, done = jax_step(stepped_state, action)
        leaf_value = jax.lax.cond(
            done,
            lambda _: jnp.asarray(0.0, dtype=jnp.float32),
            lambda s: _rollout_value(s, params, adj, model, depth_limit),
            next_state,
        )
        root_value = -leaf_value

        visits = visits.at[action].add(1)
        values = values.at[action].add(root_value)
        return RootSearchState(visits=visits, values=values, prior=sim_prior, rng_key=rng_key)

    result = jax.lax.fori_loop(0, n_simulations, sim_body, init)
    visits = result.visits
    visit_sum = jnp.sum(visits)
    return jnp.where(visit_sum > 0, visits.astype(jnp.float32) / visit_sum.astype(jnp.float32), prior)


jax_mcts_search_jit = functools.partial(
    jax.jit,
    static_argnums=(3, 4, 5, 6, 7),
)(jax_mcts_search)
