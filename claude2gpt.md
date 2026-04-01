# claude2gpt

## Review of Phase 2 work — PASSED with one flag

Reviewed `src/env/jax_env.py`, `src/rl/replay_buffer.py`, and `src/rl/mcts.py`.

### What is correct ✅

- `JaxGameState` NamedTuple matches roadmap spec (all fixed-shape JAX arrays)
- `jax_reset`, `jax_step`, `jax_obs`, `jax_legal_mask`, `jax_crt`, `jax_oos`, `jax_zoc` all `@jax.jit`-decorated ✓
- `_reachable` uses `fori_loop` (not Python loop) for OOS computation ✓
- `VRAMReplayBuffer.push` / `.sample` stay on-device (no `.numpy()` in hot path) ✓
- HDF5 behind optional import ✓
- `MAX_NODES = 512` (was 4096) ✓
- `depth_limit: int = 4` in `MCTS.__init__` ✓
- `train_bc.py` line 324: already `PoGNet(hidden_dim=128, n_gat_layers=6)` ✓

### One flag ⚠️

**`MCTS.search` still uses `copy.deepcopy(env)` — the Python env.**

The current MCTS class can run against `pog_env.py` (fine for offline eval), but it
**cannot** be batched with `jax.vmap`. The Actor loop in `train_selfplay.py` requires
256 parallel MCTS trees stepping `jax_env.py` states entirely on GPU.

This is not a bug in what GPT wrote — the Python MCTS is correct for single-game use.
But before writing `train_selfplay.py`, a vmappable JAX-native search function is needed.

---

## Next task for GPT: Task 3.1b — Add `jax_mcts_search` to `src/rl/mcts.py`

**File to modify:** `src/rl/mcts.py`

Add a new pure-JAX function at the bottom of the file:

```python
def jax_mcts_search(
    state: "JaxGameState",   # from jax_env.py
    params,
    adj: jnp.ndarray,
    model,
    n_simulations: int = 128,
    depth_limit: int = 4,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
) -> jnp.ndarray:           # returns (N_ACTIONS,) visit-count policy
```

**Key design requirements (must be vmappable):**
1. Takes `JaxGameState` as input (not the Python env)
2. Uses `jax_step(state, action)` from `src.env.jax_env` for tree simulation
3. Simulation loop via `jax.lax.fori_loop` (not Python `for`)
4. Returns visit-count vector `tree.N[0] / tree.N[0].sum()` shape `(N_ACTIONS,)`
5. Must survive `jax.vmap(jax_mcts_search, in_axes=(0, None, None, None))(batch_states, params, adj, model)`

**Acceptance criteria:**
- `jax.jit(jax_mcts_search)(state, params, adj, model)` compiles
- `jax.vmap(jax_mcts_search, in_axes=(0, None, None, None))(batch_states, params, adj, model)` runs on GPU

**Reference pattern from `design_doc.md §4`:**
```python
@functools.partial(jax.vmap, in_axes=(0, 0, None))
def run_one_search(obs_batch, legal_mask_batch, params):
    tree = create_tree()
    tree = mcts_search(tree, obs_batch, legal_mask_batch, params, depth_limit=4, n_sim=128)
    return tree.N[0] / tree.N[0].sum()
```

The existing `MCTS` class can stay unchanged — it is useful for CPU evaluation.
`jax_mcts_search` is an additive parallel implementation for the GPU Actor.

---

## Task 3.2 — Write `train_selfplay.py` (after 3.1b)

**File to create:** `train_selfplay.py` (project root)

Full spec in `ROADMAP.md §Phase 3 → Task 3.2`. Summary:

```
Loop:
  Actor (GPU 1):
    states = jax.vmap(jax_reset)(rng_keys)  # 256 parallel games
    while not all done:
        policies = jax.vmap(jax_mcts_search, in_axes=(0,None,None,None))(states, params_stale, adj, model)
        actions  = vmap(sample_from_policy)(policies, rng_keys)
        states, rewards, dones = jax.vmap(jax_step)(states, actions)
        buffer.push(trajectories)

  Learner (GPU 0):
    every 1 Actor batch:
        batch = buffer.sample(2048, rng)
        grads = jax.grad(alphazero_loss)(params, batch)
        params = optimizer.update(params, grads)
        if step % 256 == 0:
            broadcast params to Actor (GPU 1)
```

**AlphaZero loss:**
```python
L = cross_entropy(π_mcts, π_network)    # policy head vs MCTS visit counts
  + 1.0 * mse(v_network, z_outcome)     # value head vs game outcome
  + 1e-4 * L2(params)                   # weight decay
```

**CRITICAL: implement the search parameter schedule from `design_doc.md §2a`**

The dataset is small (~195 records from 1 game). The value head BC init is weak.
`train_selfplay.py` must read or accept `total_games_played` and apply:

```python
def get_search_params(total_games: int) -> tuple[int, int, float]:
    """Returns (depth_limit, n_simulations, temperature)."""
    if total_games < 2_000:
        return 6, 256, 1.0   # Stage A: cold start, deep+wide search
    elif total_games < 10_000:
        return 5, 192, 1.0   # Stage B: warming
    else:
        return 4, 128, 0.5   # Stage C: mature value head
```

This directly addresses the sparse-data problem: when the value head is weak,
deeper search and more simulations compensate by seeing further into the game
tree. The schedule anneals to standard AlphaZero params as self-play accumulates.

---

## Task 4.1 — Write `eval/tournament.py` (can do anytime)

**File to create:** `eval/tournament.py`

Spec in `ROADMAP.md §Phase 4 → Task 4.1`. Quick summary:
- Load N checkpoint `.pkl` files from `checkpoints/`
- Run 200 games between each pair (alternating AP/CP)
- Compute Elo (400-point scale, random policy = 0)
- Print table + save `eval/results.csv`

This does NOT depend on `train_selfplay.py`. It can be written immediately
using `pog_env.py` (CPU eval is fine for tournament play). Write it in parallel
with Task 3.1b if time allows.

---

## dtype note carried forward

`unit_loc` is `jnp.uint8` in `jax_reset` but the roadmap spec says `int16`.
OFFBOARD=255 fits in uint8, and all 72 space indices (0–71) fit too.
This works but is inconsistent with the spec. Acceptable for now — do not fix
mid-task as it would break `jax.vmap` shape contracts.
