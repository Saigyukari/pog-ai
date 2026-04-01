# Paths of Glory RL Engine — Full Architecture Context
*Captured: 2026-04-01 | Branch: quantum_dyn_2 | Lead Architect: Claude Sonnet 4.6*

---

## 0. Project Overview

Building a blazing-fast, JAX-ready Reinforcement Learning environment for the CDG board game **"Paths of Glory" (PoG)**. Three-agent team: Data Architect, RL Environment Builder, RL Algorithm Scientist.

**Source data (at `/home/saigyukari/PoGAIV1/`):**
- `pog_map_graph.json` — currently only 10/~70 spaces (Western Front only); no dynamic fields
- `pog_cards_db.json` — 110 cards present but all `phase: "Unknown"`; some `event_text` truncated

---

## 1. RESOLVED: Tensor Encoding for Trench Level & OOS Status

### Decision: 3-Plane Hybrid ("Approach A-prime") — PENDING HUMAN APPROVAL

After a full 2-round debate between Data Architect and Environment Builder, the team converged on this consensus:

| Plane | Name | Encoding | Shape | dtype |
|---|---|---|---|---|
| 1 | `trench_ordinal` | `trench_level / 3.0` | `(N_spaces,)` | float32 |
| 2 | `trench_l3_gate` | `(trench_level >= 3).astype(float32)` | `(N_spaces,)` | float32 |
| 3 | `oos_plane` | `oos_status.astype(float32)` | `(N_spaces,)` | float32 |

**Total: 3 planes. Combined shape: `(3, N_spaces)` within the larger `(N_planes, N_spaces)` observation tensor.**

**JAX construction (fully jit-safe, static shapes):**
```python
trench_ordinal = (trench_levels / 3.0).astype(jnp.float32)[None, :]        # (1, 70)
trench_l3_gate = (trench_levels >= 3).astype(jnp.float32)[None, :]          # (1, 70)
oos_plane      = oos_status.astype(jnp.float32)[None, :]                    # (1, 70)
# trench_levels: jnp.ndarray shape (70,) dtype int8, values 0-3
# oos_status:    jnp.ndarray shape (70,) dtype bool
```

**JSON fields to add to every space in `pog_map_graph.json`:**
```json
"trench_level": 0,
"oos_status": false
```

### Why This Beats the Pure Options

**vs. Approach B (scalar only, 2 planes):**
The Level-3 trench is a hard action-legality gate in PoG — assaulting a Level-3 trench requires a specific card (e.g., Allenby/Assault). Without an explicit gate plane, the network must learn an internal threshold detector at `f(1.0)` during early training, which can cause illegal action mask violations before it converges. The explicit binary gate plane eliminates this risk.

**vs. Approach A (one-hot, 5 planes):**
The ordinal DRM signal (trench levels 0–2 contribute linearly to `defender_DRM += trench_level`) is correctly captured by the scalar plane with free cross-level gradient generalization. Eliminates 2 redundant planes (Level-0 and Level-1 have no mechanical discontinuity between them).

### Key Game Mechanics That Drove This Decision

| Trench Level | DRM effect | Loss rule | Assault legality |
|---|---|---|---|
| 0 | +0 | Normal | Normal |
| 1 | +1 | Normal | Normal |
| 2 | +1 | Absorbs 1st step loss | Normal |
| 3 | +2 | Absorbs 1st step loss | Requires Assault card |

The 2→3 boundary is a **phase transition** (qualitative rule change + extra DRM). The scalar plane handles 0-2 linearity; the gate plane handles the 3-threshold.

---

## 2. RL Scientist Full Design Document

*Drafted by RL Algorithm Scientist (Task #6/#7 research phase). Not yet written to `src/rl/design_doc.md` — pending directory creation and human approval to write.*

### 2.1 Data Findings from Existing Files

**`pog_map_graph.json` (10 nodes, expanding to ~70):**
- Node types: Clear, Fort (Paris, Verdun, Belfort, Liege, Metz, Strasbourg)
- Nations present so far: FR, BE, GE — full map adds RU, AH, TU, IT, BU theatres
- Average degree ~4–5 edges per node → sparse graph, ideal for GAT

**`pog_cards_db.json` (110 unique card IDs: AP-01..AP-65 + CP-01..CP-65):**
- Ops values: 2–5; SR values: 2–5
- ~40% of cards are `is_combat_card: true` → nested combat sub-phase required
- Phase gating: Unknown / Limited War / Total War (legal mask must encode war status)
- Duplicate entries detected: AP-56, AP-57, AP-60, AP-61, AP-63, CP-56, CP-57, CP-60, CP-64 — likely two physical copies in the deck. Vocabulary de-duplicates to 110 unique IDs.

---

### 2.2 Network Architecture

**Recommended `N_planes = 32`**, covering:

| Feature | Planes |
|---|---|
| Space type (Clear/Fort/Mountain/Sea) | 1 |
| VP value per space | 1 |
| Controlling faction (AP/CP/neutral) | 1 |
| Trench level (ordinal scalar) | 1 |
| Trench L3 gate (binary) | 1 |
| Fort destroyed flag | 1 |
| AP OOS flag | 1 |
| CP OOS flag | 1 |
| AP unit stack by type×nationality (8 types) | 8 |
| CP unit stack by type×nationality (8 types) | 8 |
| Hand-size scalars (AP, CP) | 2 |
| Hand aggregate stats | 2 |
| VP marker position | 1 |
| War status AP/CP | 2 |
| Turn number (normalized) | 1 |
| Active player | 1 |

Card hands are encoded as a **separate `(7, 16)` embedding** injected as a global context token — NOT broadcast spatially — to avoid polluting the per-space feature planes with hand information.

**Total observation shape: `(32, 70)` for spatial planes + `(7, 16)` global context.**

---

### 2.3 Backbone: Hybrid GATv2 + Global Context

**Architecture:** 4 × GATv2Conv layers (4-head, hidden dim 64→256 after concat), with global context (card embeddings + game flags, dim 256) injected into every node at every layer via a linear projection.

- **Normalization:** Pre-LayerNorm residuals throughout. No BatchNorm (incompatible with variable-batch MCTS rollouts in JAX).
- **Adjacency matrix:** Static, precomputed from `pog_map_graph.json`. Enables efficient `jit` reuse — the graph topology never changes during play.
- **JAX library choice (open question):** `jraph` (DeepMind's JAX graph library) vs. custom GATv2 in Flax. Recommendation: custom Flax for full control over static shapes; `jraph` adds abstractions that can fight JAX's static-shape requirements.

**Why GAT over pure Transformer:**
- PoG map is a sparse irregular graph (~70 nodes, avg degree ~4–5). Full self-attention would be O(70²) = 4,900 attention weights with ~4,500 being structurally zero (non-adjacent spaces). GAT's masked attention is the natural fit.
- Global context (card hand, VP track, war status) is injected as a single dense token — handles the non-spatial features without bloating the spatial attention.

---

### 2.4 Output Heads

**Policy head:** Flat logit vector of **5,341 elements**:

| Slot range | Action type | Count |
|---|---|---|
| `[0]` | PASS | 1 |
| `[1:111]` | PLAY_AS_EVENT (by card_id) | 110 |
| `[111:441]` | PLAY_AS_OPS (card_id × op_type ∈ {MOVE, ATTACK, SR}) | 330 |
| `[441:5341]` | MOVE_UNIT / PLACE_UNIT (from_space × to_space, 70×70) | 4,900 |

*Extension: Event-triggered placement actions (e.g., Yudenitch, Allenby) add 70 PLACE_EVENT slots → 5,411 total.*

**Legal action mask:** Always `jnp.bool_[5341]` — static shape, fully `jit`-compatible.
**Masked softmax:** `jnp.where(legal_mask, logits, -jnp.inf)`

**Value head:** `Linear(256→128) → ReLU → Linear(128→1) → tanh`

---

### 2.5 Flat vs. Factored Action Head (OPEN DECISION — Needs Human Approval)

**Option A — Flat (5,341 logits):**
- Simpler to implement, static shape trivially
- MCTS sampling is a single masked softmax
- Con: most logits are illegal most of the time; sparse supervision signal

**Option B — Factored / Autoregressive (4 sequential sub-heads):**
- Sub-head 1: card selection (110 logits)
- Sub-head 2: action type (3 logits: Event/Ops/SR)
- Sub-head 3: unit/space source (70 logits)
- Sub-head 4: target space (70 logits)
- Pro: lower branching factor, denser supervision
- Con: requires non-standard autoregressive sampling during MCTS; complex `jit` boundary management; conditional masking between sub-heads is stateful

**RL Scientist recommendation:** Start with **Flat** for correctness, migrate to Factored if training proves unstable due to sparse policy supervision.

---

### 2.6 Behavioral Cloning Pre-training Pipeline

**Storage format for human expert game records:** JSONL, one game per line.
Each step entry:
```json
{
  "game_id": "...",
  "turn": 3,
  "action_round": 2,
  "player": "AP",
  "obs_tensor": [[...float32 array, serialized...]],
  "legal_mask": [...bool array...],
  "action_taken": 247,
  "outcome": 1
}
```

**Loss function:**
```
L = L_policy + λ_v × L_value
L_policy = CrossEntropy(policy_logits[legal_mask], expert_action_one_hot)
L_value  = MSE(value_head_output, discounted_outcome_target)
```

**3-Phase Anti-Collapse BC Curriculum:**

| Phase | Epochs | What's trained | λ_v | Value target |
|---|---|---|---|---|
| 1 | 1–10 | Backbone + policy head only (value frozen) | 0.0 | N/A |
| 2 | 11–30 | Backbone + policy + value head | 0.1 | `outcome × 0.99^(T−t)` |
| 3 | 31+ | Full AlphaZero MCTS self-play targets | 1.0 | MCTS value estimates |

**Why 3 phases:** Naively training the value head from sparse game outcomes during BC causes it to collapse to predicting the mean (≈0.0). Freezing it initially lets the policy head develop meaningful features first, then the value head bootstraps off those features in Phase 2.

---

### 2.7 JAX-Specific Implementation Notes

| Component | JAX strategy |
|---|---|
| Network forward pass | `jax.jit` |
| Masked softmax + action sampling | `jax.jit` |
| UCB scoring in MCTS | `jax.jit` |
| MCTS simulation loop | `jax.lax.fori_loop` with fixed `N_SIMULATIONS` |
| MCTS tree storage | Pre-allocated static arrays: `Q[4096, 5341]`, `N[4096, 5341]`, etc. No dynamic Python structures under `jit` |
| PRNG | `jax.random.split` at every simulation step |
| Dirichlet noise on root | `alpha=0.3`, `epsilon=0.25` |
| Python-native PogEnv | Wrap with `jax.pure_callback` to preserve `jit` scope |

---

## 3. Task Board — Current Status

| # | Owner | Subject | Status | Blocked By |
|---|---|---|---|---|
| 1 | DA + ENV | DEBATE: Trench & OOS tensor plane representation | **PENDING HUMAN APPROVAL** | — |
| 2 | DATA-ARCH | Complete pog_map_graph.json to full ~70 spaces | pending | awaiting approval of #1 |
| 3 | DATA-ARCH | Refine pog_cards_db.json and Python dataclasses | pending | awaiting approval of #1 |
| **8** | HITL | Human review: completed data layer | pending | blocked by #2, #3 |
| 4 | ENV-BUILD | PogEnv state machine skeleton (PettingZoo/Gym) | pending | blocked by #1, #2, #3, #8 |
| 5 | ENV-BUILD | pytest: CRT + ZOC + OOS unit tests | pending | blocked by #4 |
| **9** | HITL | Human review: step() draft + test results | pending | blocked by #5 |
| 6 | RL-SCI | JAX/Flax AlphaZero network architecture | pending | blocked by #9 |
| 7 | RL-SCI | Action space design + BC pipeline | pending | blocked by #9 |

**Dependency chain:**
```
#1 (debate) → #2, #3 → #8 (HITL) → #4 → #5 → #9 (HITL) → #6, #7
```

---

## 4. File Lock Protocol

| Agent | Owns |
|---|---|
| Data Architect | `pog_map_graph.json`, `pog_cards_db.json`, `src/data/` |
| Env Builder | `src/env/`, `tests/` |
| RL Scientist | `src/rl/` |

No agent writes to another agent's owned files. No code is written until Task #1 is approved by human lead.

---

## 5. Open Decisions Requiring Human Approval

1. **Tensor encoding (Task #1):** Approve the 3-plane hybrid (Section 1 above), or choose pure scalar (2 planes) or pure one-hot (5 planes).
2. **Flat vs. factored action head:** Flat 5,341-logit head recommended to start (Section 2.5).
3. **JAX graph library:** Custom Flax GATv2 vs. `jraph` (Section 2.3).
4. **RL Scientist design doc:** Approve writing to `src/rl/design_doc.md` (directory does not yet exist at `/home/saigyukari/PoGAIV1/src/rl/`).

---

## 6. Next Steps (After Environment Migration)

1. Human approves tensor encoding → Task #1 marked complete → Data Architect begins Tasks #2 and #3
2. Human reviews completed data layer (Task #8 HITL checkpoint)
3. Env Builder builds PogEnv skeleton (Task #4) → writes pytest suite (Task #5)
4. Human reviews step() and test results (Task #9 HITL checkpoint)
5. RL Scientist writes full design doc and begins network implementation (Tasks #6, #7)
