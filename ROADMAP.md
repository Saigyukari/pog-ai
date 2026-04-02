# PoG-AI Master Roadmap

> **For LLM agents:** Read this file first. It tells you exactly what is done,
> what is next, what blocks what, and the acceptance criteria for each task.
> The authoritative architecture decisions are in `src/rl/design_doc.md`.

---

## Current status snapshot

```
Phase 0 — Foundation        ████████████████████  DONE
Phase 1 — BC Pre-training   ████████████████████  DONE — running on cluster now
Phase 2 — JAX Env Rewrite   ████████████████████  DONE (jax_env, replay_buffer, mcts patched)
Phase 3 — RL Self-Play      ████████████████████  DONE (train_selfplay.py, jax_mcts_search)
Phase 4 — Evaluation        ████████████████████  DONE (eval/tournament.py)
Phase 5 — Human Interface   ████████████████████  DONE (play.py)
Phase 6 — Board Init        ████████████████████  DONE (starting_positions.py, VP tracking, terminal reward)
Phase 7 — Card Mechanics    ████████░░░░░░░░░░░░  Task 7.1 DONE; Task 7.2 assigned (war status)
```

---

## What is already built (do not re-implement)

| File | What it does |
|---|---|
| `src/data/pog_engine.py` | Core data structures, observation tensor, action mask |
| `src/env/pog_env.py` | CPU Python env (PettingZoo AEC) — used for BC only |
| `src/rl/network.py` | GATv2 network, 4-sub-head policy, value head |
| `src/rl/bc_pipeline.py` | 3-phase BC loss functions, `make_bc_batches()` |
| `src/rl/mcts.py` | Pre-allocated MCTS tree, UCB scores, Dirichlet noise |
| `src/data/rtt_parser.py` | RTT JSON → training JSONL (`extract_training_records`) |
| `data/rtt_space_map.json` | RTT space integer → our space str_id (67/72 spaces) |
| `data/training/expert_games.jsonl` | ~20,000 expert records, ready for BC |
| `pog_map_graph.json` | 72-space board graph |
| `pog_cards_db.json` | 130 unique cards |

---

## Phase 1 — Behavioral Cloning (READY TO RUN)

**Goal:** Train PoGNet to imitate expert card play before any self-play.

### Task 1.1 — Write `train_bc.py`  ✅ DONE by GPT-o3 (verified by Claude)

**File:** `train_bc.py` (project root) — exists and correct.
**One fix required before H200 run:** change `PoGNet()` → `PoGNet(hidden_dim=128, n_gat_layers=6)` on line 324. See `claude2gpt.md`.

**What it must do:**
1. Load `data/training/expert_games.jsonl`
2. Load adjacency matrix from `pog_map_graph.json` via `load_adjacency_matrix()`
3. Instantiate `PoGNet` and `create_train_state()`
4. Run 3-phase curriculum using loss functions already in `bc_pipeline.py`:
   - Phase 1 (epochs 1–10): `bc_loss_phase1` — policy head only
   - Phase 2 (epochs 11–30): `bc_loss_phase2` — policy + value (λ=0.1)
   - Phase 3 (epochs 31+): `bc_loss_phase3` — full (λ=1.0)
5. Save checkpoints every 10 epochs to `checkpoints/bc/`
6. Log: epoch, policy loss, value loss, policy accuracy (top-1 action match)

**Key imports available:**
```python
from src.rl.bc_pipeline import make_bc_batches, bc_loss_phase1, bc_loss_phase2, bc_loss_phase3
from src.rl.network import PoGNet, create_train_state, load_adjacency_matrix
```

**Hardware:** Both H200s via `jax.pmap` (data parallel, split batch 4096 → 2048 per GPU).

**Acceptance criteria:**
- Policy accuracy > 15% top-1 (random = 1/110 ≈ 0.9%)
- Value loss converges (not stuck at predicting 0.0)
- Checkpoint saved at `checkpoints/bc/epoch_030.pkl`

**Data volume needed:**
- 20K records: smoke test (will overfit — expect high train acc, low val acc)
- 50K+ records: meaningful BC (collect more RTT games via `src/data/rtt_parser.py`)

---

## Phase 2 — JAX Environment Rewrite (CRITICAL ENGINEERING)

**Why this is required:** The current `pog_env.py` is a CPU-bound Python class.
During RL self-play, the Actor GPU (GPU 1) runs 256 parallel games. Each step
requires a CPU round-trip via `jax.pure_callback` — this creates PCIe transfer
latency that throttles H200 utilization to < 5%. The env must execute entirely
on GPU.

**Reference:** DeepMind `pgx`, `brax`, `mctx` — all implement environments as
pure JAX state machines. This is a hard requirement for exploiting our H200s.

**Design decision (final):**
- Game state = a JAX NamedTuple of fixed-shape arrays (no Python objects)
- `jax_step(state, action, rng) → (new_state, obs, reward, done)` must be `@jax.jit`-able
- BFS (OOS/ZOC) → fixed-iteration matrix reachability: `reachable = adj^k @ sources`
- CRT → pre-computed 7-outcome table; sample with `jax.random.choice`
- Card events → `jax.lax.switch` over 110 cases (lookup-table pattern)

### Task 2.1 — Write `src/env/jax_env.py` — Core state machine

**File to create:** `src/env/jax_env.py`

**GameState NamedTuple (all JAX arrays, fixed shapes):**
```python
class JaxGameState(NamedTuple):
    # Board
    unit_loc:       jnp.ndarray  # (194,)  int16  — piece idx → space idx (255=off-board)
    unit_strength:  jnp.ndarray  # (194,)  int8   — current strength
    trench_level:   jnp.ndarray  # (72,)   int8   — 0/1/2/3
    oos_mask:       jnp.ndarray  # (72,)   bool   — OOS per space
    control:        jnp.ndarray  # (72,)   int8   — 0=AP, 1=CP, 2=neutral
    # Cards
    ap_hand:        jnp.ndarray  # (7,)    int8   — card indices (127=empty slot)
    cp_hand:        jnp.ndarray  # (7,)    int8
    ap_discard:     jnp.ndarray  # (65,)   bool
    cp_discard:     jnp.ndarray  # (65,)   bool
    # Game meta
    turn:           jnp.ndarray  # ()      int8
    action_round:   jnp.ndarray  # ()      int8
    active_player:  jnp.ndarray  # ()      int8   — 0=AP, 1=CP
    war_status:     jnp.ndarray  # ()      int8   — 0=limited, 1=total
    vp:             jnp.ndarray  # ()      int8   — positive=AP winning
    rng_key:        jnp.ndarray  # (2,)    uint32 — JAX PRNG state
```

**Functions to implement (all `@jax.jit`-able):**
```python
def jax_reset(rng_key) -> JaxGameState: ...
def jax_step(state, action) -> tuple[JaxGameState, float, bool]: ...
def jax_obs(state, player) -> tuple[jnp.ndarray, jnp.ndarray]: ...
    # returns: obs_tensor (32,72), card_context (7,16)
def jax_legal_mask(state) -> jnp.ndarray: ...  # (5341,) bool
def jax_crt(rng, atk_str, def_str, drm, trench_level) -> dict: ...
def jax_oos(state, player) -> jnp.ndarray: ...  # (72,) bool via matrix reachability
def jax_zoc(state, player) -> jnp.ndarray: ...  # (72,) bool
```

**Acceptance criteria:**
- `jax.jit(jax_step)` compiles without error
- `jax.vmap(jax_step, in_axes=(0,0))(batch_states, batch_actions)` runs on GPU
- 256 parallel steps complete in < 10 ms on H200
- Core mechanics match `pog_env.py` results (test against existing tests)

### Task 2.2 — Binary replay buffer

**Why:** JSON serialization (~11KB/record, CPU-bound) will saturate memory bandwidth
before GPU compute during RL. During self-play the Actor generates ~15K records/sec —
JSON cannot keep up.

**Decision (final):**

| Phase | Format | Reason |
|---|---|---|
| BC (current) | JSONL | Human-readable, debuggable, low volume |
| RL hot buffer | JAX DeviceArray (GPU 0 VRAM) | Zero-copy sample → train |
| RL cold storage | HDF5 (`h5py`) | Compact, memory-mappable, standard |

**File to create:** `src/rl/replay_buffer.py`

```python
class VRAMReplayBuffer:
    """
    Fixed-size replay buffer stored as JAX arrays on GPU 0.
    Write: Actor pushes trajectories via NVLink (no CPU).
    Read:  Learner samples random minibatch, zero-copy.
    """
    def __init__(self, capacity: int = 500_000): ...
    def push(self, obs, card_ctx, mask, action, reward, done): ...
    def sample(self, batch_size: int, rng_key) -> tuple: ...
    def save_hdf5(self, path: str): ...          # checkpoint to cold storage
    def load_hdf5(self, path: str): ...          # resume from checkpoint
```

**Acceptance criteria:**
- `push` + `sample` cycle runs entirely on GPU (no `.numpy()` calls in hot path)
- 15K push/sec sustained without CPU bottleneck
- HDF5 save/load round-trips correctly

---

## Phase 3 — RL Self-Play (blocked by Phase 1 + Phase 2)

**Goal:** Improve beyond BC using MCTS self-play with AlphaZero-style targets.

**Architecture (from `src/rl/design_doc.md`):**
- GPU 0: Learner — large-batch gradient updates, hosts replay buffer
- GPU 1: Actor — 256-way `jax.vmap` parallel MCTS, depth 3–5, no rollouts
- No JSON anywhere in this phase

### Task 3.1 — Patch `src/rl/mcts.py`  ✅ DONE

MAX_NODES=512, depth_limit=4, jax_mcts_search() added (pure JAX, vmappable).

### Task 3.2 — Write `train_selfplay.py`  ✅ DONE

**File to create:** `train_selfplay.py` (project root)

**What it must do:**
```
Loop:
  Actor (GPU 1):
    vmapped_states = jax_reset(256 rng_keys)
    while not all done:
        policy = vmap(mcts_search)(vmapped_states, params_stale)
        actions = sample(policy)
        vmapped_states, rewards, dones = vmap(jax_step)(vmapped_states, actions)
        buffer.push(trajectories)

  Learner (GPU 0):
    every 1 Actor batch:
        batch = buffer.sample(2048, rng)
        grads = jax.grad(alphazero_loss)(params, batch)
        params = optimizer.update(params, grads)
        if step % 256 == 0:
            broadcast params to GPU 1
```

**Loss function (AlphaZero combined):**
```python
L = cross_entropy(π_mcts, π_network)    # policy head matches MCTS visit counts
  + λ_v  * mse(v_network, z_outcome)    # value head matches game outcome
  + λ_reg * L2(params)                  # weight decay
# λ_v = 1.0 (full, we are past BC phase), λ_reg = 1e-4
```

**Acceptance criteria:**
- Actor GPU utilization > 80% sustained
- Learner GPU utilization > 60% sustained
- Self-play Elo increases measurably after 1K games vs random policy baseline

---

## Phase 4 — Evaluation

**Goal:** Track Elo, detect regressions, compare checkpoints.

### Task 4.1 — Write `eval/tournament.py`  ✅ DONE

Round-robin, alternating sides, Elo 400-scale anchored at random=0, CSV output.

Usage: `python eval/tournament.py checkpoints/bc/ --include-random --games 200`

**Milestones to target:**

| Elo | Meaning |
|---|---|
| 0 | Random policy baseline |
| +200 | BC policy (imitates experts without understanding) |
| +400 | Beginner human equivalent |
| +700 | Club-level human equivalent |
| +1000 | Expert-level (stretch goal) |

---

## Phase 5 — Human Interface

**Goal:** Let the user play against the trained AI on their local RTX 5060.

### Task 5.1 — Write `play.py`  ✅ DONE

Human-vs-AI terminal interface. `python play.py --checkpoint <pkl> --side AP`

---

## Phase 6 — Board Initialization (CRITICAL for meaningful self-play)

**Problem:** Both `jax_env.py` and `pog_env.py` start with empty boards.
`UNIT_FACTION` is all -1 → ZOC always returns False → no movement actions ever legal.
Self-play is just card-passing. VP never changes. Training signal is meaningless.

### Task 6.1 — Write `src/data/starting_positions.py`  ✅ DONE

Piece metadata (faction/type/strength) from data.js + Historical 1914 army positions.
Both `jax_env.py` and `pog_env.py` patched. `jax_legal_mask` now returns MOVE_UNIT actions.

### Task 6.2 — Fix VP tracking + terminal reward  ✅ DONE

`_recompute_vp` called after every control change; terminal reward is +1/−1 from active player's perspective; reset initialises VP from starting control.

---

## Phase 7 — Card Mechanics (improves self-play quality)

**Problem:** Cards are dealt at reset but never re-dealt. By turn 2 both players have empty hands. For 19 of 20 turns only movement actions are legal. Since card selection is PoG's primary strategic decision, self-play learns almost nothing useful.

### Task 7.1 — Card re-deal at turn boundary  ✅ DONE

`_deal_hand_from_deck` added; `_advance_turn(state, deal_key)` re-deals both hands at `wrap=True`. `war_status` also split into `war_status_ap`/`war_status_cp` (bonus). `pog_env.py` VP sign convention aligned. 58/58 tests pass.

### Task 7.2 — War status advancement at turn boundary  ← CURRENT TASK FOR GPT

**Problem:** `war_status_ap` and `war_status_cp` are initialised to `PHASE_LIMITED` and never change. ~30–40% of the card deck (Total War cards) is permanently locked for the entire game, so the policy never learns card selection strategy for that portion of the deck.

**Fix:** In `_advance_turn`, after the hand re-deal block, add VP-threshold checks:
- `vp >= +5` at turn boundary → advance `war_status_ap` to `PHASE_TOTAL`
- `vp <= -5` at turn boundary → advance `war_status_cp` to `PHASE_TOTAL`
- One-way: never regresses to Limited once Total

See `claude2gpt.md` for the exact code and test spec.

---

## Engineering decision log

### On JAX env (Gemini suggestion — accepted with phasing)

**Gemini proposed:** Pure GPU state machine, zero CPU involvement.
**Our decision:** Agree fully. The CPU-bound `pog_env.py` is acceptable for BC
(low volume, one game at a time) but fatal for RL self-play at 256-way vmap scale.

**Phasing:** Task 2.1 implements the full JAX env. The 110 card events are the
hardest part — use `jax.lax.switch(event_id, event_fns, state)` where each
`event_fn` is a pre-written JAX function. This is the lookup-table pattern used
by pgx for complex rules.

### On binary replay buffer (Gemini suggestion — accepted)

**Gemini proposed:** TFRecords, HDF5, or msgpack. No JSON during RL.
**Our decision:** JAX DeviceArray as hot buffer (zero-copy GPU-side),
HDF5 for cold/persistent storage. No TFRecords (TF dependency), no msgpack
(still CPU-bound). JSONL is permanently retired after Phase 1.

### On deep rollouts (our earlier decision — unchanged)

No deep rollouts. No chance nodes. No PIMC.
Value head = leaf evaluator. Single CRT sample per simulation path.
See `src/rl/design_doc.md` §1–3 for full justification.

---

## Dependency graph

```
[expert_games.jsonl ~20K]
         │
         ▼
[Task 1.1] train_bc.py           ← UNBLOCKED, implement first
         │
         │   (parallel with 1.1)
         ▼
[Task 2.1] jax_env.py            ← Major engineering, start immediately after 1.1
[Task 2.2] replay_buffer.py      ← Can write alongside 2.1
         │
         ▼
[Task 3.1] patch mcts.py         ← Small change, 30 min
         │
         ▼
[Task 3.2] train_selfplay.py     ← Requires 1.1 + 2.1 + 2.2 + 3.1
         │
         ▼
[Task 4.1] eval/tournament.py    ← Can write anytime, run after 3.2
```

---

## File map (complete project)

```
PoGAIV1/
├── ROADMAP.md                       ← this file
├── IMPLEMENTATION_SUMMARY.md        ← architecture reference
├── TRAINING_GUIDE.md                ← setup + hardware guide
├── RTT_DATA_PIPELINE.md             ← data extraction guide
│
├── train_bc.py                      ← [TASK 1.1] TO CREATE
├── train_selfplay.py                ← [TASK 3.2] TO CREATE
│
├── pog_map_graph.json               ✓ done
├── pog_cards_db.json                ✓ done
│
├── data/
│   ├── rtt_games/                   ← drop RTT JSON exports here
│   ├── training/
│   │   └── expert_games.jsonl       ✓ ~20K records ready
│   ├── rtt_space_map.json           ✓ done
│   └── data.js                      ✓ RTT source (reference only)
│
├── src/
│   ├── data/
│   │   ├── pog_engine.py            ✓ done
│   │   ├── rtt_parser.py            ✓ done
│   │   └── convert_records.py       ✓ done
│   ├── env/
│   │   ├── pog_env.py               ✓ done (CPU, used for BC only)
│   │   └── jax_env.py               ← [TASK 2.1] TO CREATE
│   └── rl/
│       ├── network.py               ✓ done
│       ├── bc_pipeline.py           ✓ done
│       ├── mcts.py                  ← [TASK 3.1] patch MAX_NODES + depth_limit
│       ├── replay_buffer.py         ← [TASK 2.2] TO CREATE
│       └── design_doc.md            ✓ architecture decisions
│
├── eval/
│   └── tournament.py                ← [TASK 4.1] TO CREATE
│
├── checkpoints/
│   ├── bc/                          ← Phase 1 checkpoints saved here
│   └── rl/                          ← Phase 3 checkpoints saved here
│
└── tests/                           ✓ 41 tests passing
```
