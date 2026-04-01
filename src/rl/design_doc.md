# PoG-AI — MCTS Strategy & Hardware Design Document

**Status:** Finalized architecture decisions
**Hardware:** 2× NVIDIA H200 (141 GB HBM3e each, NVLink)
**Training data:** ~20,000 records from expert games (BC pre-train)

---

## 1 — The Deep Rollout Problem: Why We Abandon It

The concern is correct. Deep MCTS rollouts to terminal state are **incompatible with PoG**
for two compounding reasons.

### 1a. Variance explosion from stochasticity

PoG's CRT (Combat Results Table) fires multiple times per turn across multiple fronts.
Each result is a discrete random outcome (Engaged / A1 / EX / D1 / D1R / D2 / DE).
In a tree of depth D with K combats per ply, the number of stochastic branches grows
as `(7^K)^D`. At depth 10 with just 2 combats per turn: `7^2^10 ≈ 10^17` terminal
outcomes. No amount of sampling corrects for this — the variance of Monte Carlo
estimates over this space is so large that the value signal is buried in noise.

**Evidence from literature:**
- TD-Gammon (Tesauro 1995) — backgammon's 2d6 rolls create similar compounding
  variance. Tesauro's solution: **never do deep rollouts.** Train a value network on
  TD(λ) targets and use it directly as a leaf evaluator. No rollout. The value network
  *absorbs* the stochastic EV implicitly.
- Blood Bowl AI (Justesen et al., 2019) — explicit chance nodes per action work only
  because each action has a *single* dice event with bounded branching (2–3 outcomes).
  PoG combats are more complex. The Blood Bowl approach does not transfer.

**Decision:** No deep rollouts. Value head = leaf evaluator at all times.

### 1b. Strategy fusion from imperfect information

Each player holds 6–7 hidden cards. A deep rollout that conditions on a specific
opponent hand will find a locally optimal path that assumes the opponent *has* that hand.
When averaged over many such hands (PIMC), the resulting policy is a weighted average of
strategies optimized for incompatible assumptions — this is what Koller & Pfeffer (1997)
termed **strategy fusion**. The fused policy is demonstrably worse than ignoring the
hidden information entirely in many game situations.

**Evidence from literature:**
- **Libratus (Brown & Sandholm, 2017):** Heads-up poker. Solved via Counterfactual
  Regret Minimization (CFR), which explicitly tracks a *probability distribution* over
  opponent hands rather than sampling a single realization. CFR then minimizes regret
  across all hands simultaneously. The key insight: treat hidden info as a *distribution*,
  never a point sample.
- **Pluribus (Brown & Sandholm, 2019):** 6-player poker. Scales CFR with blueprint
  strategy + real-time subgame solving. The blueprint handles depth > 4 implicitly.
- **DeepNash (Perolat et al., 2022):** Stratego — 2-player, asymmetric hidden info
  (hidden piece identities). Uses Regularized Nash Dynamics (R-NaD), a regret-based
  objective. **DeepNash uses zero MCTS at all.** It trains purely with self-play via
  policy gradient + Nash regularization. The network learns to reason about hidden info
  implicitly through self-play against itself.

**Decision:** No PIMC. No explicit opponent hand sampling during MCTS.
The value head, trained via self-play under true hidden-info conditions, will implicitly
learn to evaluate positions as EV over the opponent's possible hands.

---

## 2 — Finalized MCTS Design

### Core principle

```
Search depth: 3–5 plies (one full action round ± one enemy response)
Leaf evaluation: GNN value head (no rollout)
Stochasticity: single-sample per simulation path (not tree-branched)
Hidden info: not explicitly modeled in tree — value head handles it implicitly
```

### Why depth 3–5?

One ply in PoG = one card play (event/ops/sr) + the resulting unit movements.
Depth 3 covers: *our card → our moves → enemy card → enemy moves → our card*.
This is one complete tactical loop: we see the full cause-effect of our strategic
decision plus one enemy counter-response before handing off to the value head.

At depth 5 we see two full rounds. Beyond that, stochastic CRT variance
dominates any signal the value head can return.

### Handling stochasticity: single-sample path

When a combat occurs during a MCTS simulation:
- Draw **one** CRT sample from the probability distribution
- Follow that single branch for the remainder of the simulation
- Do **not** branch the tree at the chance node

This is equivalent to "outcome sampling" from the MCTS literature (Lanctot et al., 2009).
Over N simulations (N=64–256), the visit counts naturally average over many CRT outcomes.
The value head at the leaf absorbs remaining variance.

Contrast with explicit chance nodes: branching on 7 CRT outcomes × K combats ×
depth D is exponentially more expensive with no practical benefit once we have a
strong value head.

### Handling hidden information: implicit EV via value head

The GNN value head sees only the *observable* board state (our hand, board positions,
trench levels, OOS status — all 32 feature planes). It does **not** see the opponent's
hand. After BC pre-training + self-play, it will have learned to assign values that
reflect the full distribution of possible opponent hands consistent with the game state.

This is the same approach used by AlphaZero for Chess/Go — the value head provides a
point estimate of EV; the MCTS then uses this to guide search without needing to
enumerate hidden information explicitly.

### MCTS configuration

```python
MAX_NODES  = 512     # was 4096 — truncated search needs far fewer nodes
DEPTH_LIMIT = 4      # hard cutoff; value head called at this depth (see §2a for schedule)
N_SIMULATIONS = 128  # per move decision (see §2a for schedule)
C_PUCT     = 1.5     # unchanged
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS   = 0.25
```

MAX_NODES reduction from 4096→512 is critical for vmap batching on GPU
(see §4 hardware plan).

### §2a — Search parameter schedule (small-data adaptation)

**Problem:** Optimal MCTS depth is inversely related to value head quality.
- Strong value head (post self-play): depth 4 is sufficient — it can evaluate
  positions accurately, so there is little gain beyond one tactical loop.
- Weak value head (early training, <1K self-play games): depth 4 returns
  near-random leaf values. Going deeper reaches positions where the outcome
  is more legible (winning/losing positions are more obvious structurally).

This matters especially when BC data is sparse (e.g., <500 records).
The value head trained on 1–2 games has learned the outcome of those specific
games, not general positional value. Deeper search partially compensates.

**The CRT variance limit:** Increasing depth beyond 6 in PoG hits the
stochasticity wall (§1a). At depth 6 with K=2 combats/ply: `7^2^6 ≈ 10^10`
terminal configurations over the search horizon — averaging over N_sim paths
is still tractable, but depth 8+ is not. **Hard cap: depth ≤ 6.**

**N_simulations** is a cleaner lever than depth for compensating a weak value
head: more simulations average over more CRT outcomes without deepening the
noise cone. Prefer increasing N_sim before increasing depth.

**Recommended schedule for `train_selfplay.py`:**

| Stage | Self-play games | depth_limit | N_sim | Temperature |
|---|---|---|---|---|
| A — Cold start | 0 – 2 K | 6 | 256 | 1.0 |
| B — Warming | 2 K – 10 K | 5 | 192 | 1.0 |
| C — Mature | 10 K + | 4 | 128 | 0.5 |

**Memory note (Stage A, depth=6, N_sim=256):**
MAX_NODES=512 may fill after ~85 simulations (6 new nodes/sim early on).
Once the tree is full, selection reuses existing nodes — the search becomes
more focused rather than failing. This is acceptable. No change to MAX_NODES
needed; 141 GB H200 has headroom if a larger tree (1024 nodes) is needed.

**BC adjustment for sparse data:**
With <500 records, run Phase 1 only (policy-head BC, epochs 1–10) and skip
Phase 2/3. The value head cannot learn anything meaningful from 1–2 game
outcomes and will overfit. Let self-play build the value head from scratch.
`train_bc.py --epochs 10` is the right invocation for sparse data.

### What we do NOT use

| Technique | Why rejected |
|---|---|
| Deep rollouts to terminal | Variance explosion, §1a |
| Explicit chance nodes | Exponential branching, §1a |
| PIMC (Perfect Information MC) | Strategy fusion, §1b |
| CFR / R-NaD | Correct but requires full-game tree abstraction; overkill for our BC→selfplay pipeline |
| DeepNash (zero MCTS) | Valid fallback if value head quality is poor; revisit at phase 3 |

---

## 3 — The AlphaStar Connection (Action Space)

Our 4-sub-head factored policy already correctly handles the massive action space:

```
card_logits   (B, 110)   which card
atype_logits  (B, 3)     event / ops / sr
src_logits    (B, 72)    source space
tgt_logits    (B, 72)    target space
```

The flat prior for MCTS is reconstructed as:
```
π(flat) = softmax(card)[card_idx]
        × softmax(atype)[op_type]
        × softmax(src)[src_idx]
        × softmax(tgt)[tgt_idx]
```

masked by `legal_mask` before normalization. This mirrors AlphaStar's
auto-regressive action head: factored sampling is both memory-efficient and
prevents the "curse of dimensionality" on a 5341-action space.

---

## 4 — 2× H200 Hardware Pipeline

### GPU roles

```
┌─────────────────────────────────────────────────────────┐
│  GPU 0 — LEARNER (141 GB)                               │
│                                                         │
│  • Holds current PoGNet parameters (θ)                  │
│  • Replay buffer: 500K records × ~11KB ≈ 5.5 GB         │
│  • Batch size: 4096 (BC) → 2048 (RL)                   │
│  • Optimizer: Adam (β1=0.9, β2=0.999, ε=1e-8)          │
│  • Gradient update every step                           │
│  • Broadcasts θ_new to GPU 1 every 256 gradient steps  │
└────────────────────┬────────────────────────────────────┘
                     │ NVLink param sync (~10 ms)
┌────────────────────▼────────────────────────────────────┐
│  GPU 1 — ACTOR (141 GB)                                 │
│                                                         │
│  • Holds stale copy of θ (lagged by ≤256 steps)         │
│  • jax.vmap over 256 parallel game environments         │
│  • Each env: truncated MCTS depth=4, N_sim=128          │
│  • Generates 256 × ~15 steps/game ≈ 3840 records/batch  │
│  • Pushes trajectories to GPU 0 replay buffer via PCIe  │
│                                                         │
│  Memory per vmap slot:                                  │
│    MCTSTree (512 nodes × 5341 actions × 4 arrays):      │
│      512 × 5341 × 4 × 4 bytes = ~44 MB per game        │
│    256 games × 44 MB = ~11 GB  ✓ fits in 141 GB        │
└─────────────────────────────────────────────────────────┘
```

### Training phases

**Phase 1 — Behavioral Cloning (BC)**
- Both GPUs used for data-parallel BC (JAX pmap across 2 GPUs)
- Dataset: `data/training/expert_games.jsonl`
- Batch: 4096 split across 2 GPUs = 2048 per GPU
- **Data volume determines curriculum:**
  - <500 records: Phase 1 only (policy-head, epochs 1–10). Skip value curriculum.
    `python train_bc.py --epochs 10`
  - 5K–20K records: Full 3-phase curriculum, 50 epochs.
    `python train_bc.py --epochs 50`
  - 50K+ records: Full curriculum, 200 epochs — target performance tier.
- Stop when policy cross-entropy plateaus

**Phase 2 — Self-Play RL**
- Switch to async Actor-Learner split (one GPU each)
- Actor generates truncated MCTS games, Learner trains on trajectories
- Loss: AlphaZero combined — `L = L_policy + λ_v × L_value + λ_reg × L_L2`
- Replay buffer: 500K records (FIFO), minimum fill 50K before RL starts
- Target: Elo gain measurable after ~10K self-play games

**Phase 3 — Evaluation**
- Periodically freeze θ_eval = snapshot of θ every 1K Learner steps
- Run round-robin tournament between checkpoints on CPU (eval/tournament.py)
- Elo computed with 400-point scale, anchor = random policy = 0

### JAX implementation notes for Actor (GPU 1)

```python
# Pseudocode — key design points for implementation

@functools.partial(jax.vmap, in_axes=(0, 0, None))
def run_one_search(obs_batch, legal_mask_batch, params):
    # obs_batch:       (32, 72) for this vmap slot
    # legal_mask_batch: (5341,) for this vmap slot
    # params:          shared across all vmap slots (None axis)
    tree = create_tree()          # 512-node pre-allocated tree
    tree = mcts_search(tree, obs_batch, legal_mask_batch, params,
                       depth_limit=4, n_sim=128)
    return tree.N[0] / tree.N[0].sum()   # root visit-count policy


# Batched inference — the network runs once per MCTS simulation
# across ALL 256 parallel games simultaneously:
@jax.jit
def batch_network_inference(obs_stack, card_stack, adj, params):
    # obs_stack:  (256, 32, 72)
    # card_stack: (256, 7, 16)
    return model.apply(params, obs_stack, card_stack, adj)
    # returns: (policy_heads, value) both batched over 256
```

The key efficiency gain: one `batch_network_inference` call services all 256 parallel
MCTS trees simultaneously, maximizing H200 tensor core utilization.

### Memory budget (GPU 1 — Actor)

| Component | Size |
|---|---|
| PoGNet parameters (fp16) | ~50 MB |
| 256 × MCTSTree (512 nodes) | ~11 GB |
| 256 × obs tensors (32×72 fp32) | ~150 MB |
| 256 × adjacency matrix (72×72) | ~5 MB |
| Activation buffers (4 GAT layers) | ~2 GB |
| **Total** | **~13.2 GB** |

Fits easily in 141 GB — leaves 127 GB headroom for larger batch sizes or
deeper trees if empirically justified.

### Expected throughput

| Metric | Estimate |
|---|---|
| Network inference (256 batch, H200) | ~0.5 ms |
| 128 MCTS sims × 4 depth = 512 net calls | ~256 ms per move |
| 256 parallel games × 15 moves avg | ~3840 records / 256 ms = **15K records/sec** |
| Learner gradient step (batch 2048, H200) | ~5 ms |
| Learner throughput | ~200 steps/sec = **400K samples/sec** |

Actor is the throughput bottleneck. Can scale by reducing N_sim=128→64 with
minimal Elo regression once the value head is strong.

---

## 5 — Decision Summary

| Question | Decision | Rationale |
|---|---|---|
| Deep rollouts? | **No** | Variance explosion (§1a) |
| Chance nodes? | **No** | Single-sample path instead (§2) |
| PIMC? | **No** | Strategy fusion (§1b) |
| CFR/R-NaD? | **No, revisit at phase 3** | Requires full-game abstraction |
| Search depth | **4–6 plies (scheduled)** | Start deep (weak value head), anneal to 4 as value matures (§2a) |
| Leaf evaluator | **GNN value head** | TD-Gammon precedent |
| Hidden info | **Implicit via value head** | DeepNash/AlphaZero precedent |
| GPU 0 role | **Learner** | Large-batch gradient updates |
| GPU 1 role | **Actor** | 256-way vmap parallel MCTS |
| Param sync | **Every 256 Learner steps** | Off-policy lag tolerable at this rate |
| BC batch size | **4096** (2048 per GPU, pmap) | Fills H200 tensor cores |
| RL batch size | **2048** from replay buffer | Balance stability vs freshness |

---

## 6 — Next Implementation Steps

In order:

1. **`train_bc.py`** — launch BC pre-training using bc_pipeline.py on expert_games.jsonl
2. **Reduce `MAX_NODES` 4096→512** in mcts.py for Actor vmap budget
3. **Add `depth_limit` parameter** to MCTS search loop (currently unlimited)
4. **`train_selfplay.py`** — async Actor-Learner loop using the GPU split above
5. **`eval/tournament.py`** — Elo evaluation between checkpoints

BC training is unblocked right now. Steps 2–5 can follow once BC converges.
