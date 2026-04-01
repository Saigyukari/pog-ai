# Paths of Glory RL Engine — Implementation Summary
*Last updated: 2026-04-01 | Branch: quantum_dyn_2 | Lead Architect: Claude Sonnet 4.6*

---

## 0. What This Project Builds

A blazing-fast, JAX-ready Reinforcement Learning environment for the CDG board game **"Paths of Glory" (PoG)**. The system learns to play both sides (Allied Powers / Central Powers) via an AlphaZero-style pipeline: a Graph Attention Network evaluates board positions, a factored policy head selects actions, and behavioral cloning on human expert games bootstraps the agent before self-play.

---

## 1. Directory Structure

```
PoGAIV1/
├── pog_map_graph.json          # Full ~72-space board graph (Data Architect)
├── pog_cards_db.json           # 110-card deck database (Data Architect)
├── pog_architecture_context.md # Finalized architecture decisions
├── IMPLEMENTATION_SUMMARY.md   # ← This file
├── PoG-Deluxe Rules 2022 Final.pdf
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── pog_engine.py       # Core dataclasses + tensor builders (Data Architect)
│   ├── env/
│   │   ├── __init__.py
│   │   └── pog_env.py          # PettingZoo AEC environment (Env Builder)
│   └── rl/
│       ├── __init__.py
│       ├── network.py          # JAX/Flax GATv2 + factored policy (RL Scientist)
│       ├── bc_pipeline.py      # Behavioral cloning pipeline (RL Scientist)
│       └── design_doc.md       # Full network design document (RL Scientist)
│
└── tests/
    ├── __init__.py
    ├── test_crt.py             # Combat Results Table unit tests
    ├── test_zoc.py             # Zone of Control unit tests
    └── test_oos.py             # Out-of-Supply unit tests
```

---

## 2. Finalized Architecture Decisions

All of the following were debated and locked in — **do not re-open these**:

| Decision | Choice | Rationale |
|---|---|---|
| Map encoding — trench/OOS | **3-plane hybrid** | Ordinal scalar (0–2 linear DRM) + L3 gate (hard assault legality) + OOS bool |
| Graph neural network | **Custom Flax GATv2Conv** | Sparse irregular graph (~72 nodes, avg degree ~4); O(N×degree) vs O(N²) attention |
| Action head | **4-sub-head factored** | Lower branching factor, denser gradient signal vs flat 5,341-logit head |
| Normalization | **Pre-LayerNorm residuals** | No BatchNorm (breaks under variable-batch MCTS rollouts) |
| JAX graph library | **Pure Flax** (no jraph) | Full static-shape control; jraph abstractions fight JAX's `jit` requirements |
| BC value training | **3-phase curriculum** | Freeze value head first; prevents collapse to mean from sparse outcome signal |

---

## 3. Observation Space — Full 32-Plane Spec

**Shape:** `(32, 72)` float32 spatial planes + `(7, 16)` float32 global card context

| Plane | Name | Encoding | Notes |
|---|---|---|---|
| 0 | `terrain_type` | `terrain_int / 4.0` | 0=Clear, 1=Fort, 2=Mountain, 3=Sea, 4=Desert |
| 1 | `vp_value` | `vp / 5.0` | 0–5 normalized |
| 2 | `controlling_faction` | 0.0=AP, 0.5=neutral, 1.0=CP | |
| 3 | `trench_ordinal` | `trench_level / 3.0` | 0–3 scalar; captures linear DRM for levels 0–2 |
| 4 | `trench_l3_gate` | `(trench_level >= 3).astype(float)` | Hard gate: Assault card required |
| 5 | `fort_destroyed` | bool → float | Fortress destroyed flag |
| 6 | `ap_oos` | bool → float | AP unit at this space is OOS |
| 7 | `cp_oos` | bool → float | CP unit at this space is OOS |
| 8–15 | `ap_units_by_type` | `count / 3.0` (clamped 0–1) | 8 unit types: INF, CAV, ART, CORP, ARMY, FLEET, SUB, AIR |
| 16–23 | `cp_units_by_type` | `count / 3.0` (clamped 0–1) | Same 8 types |
| 24 | `ap_hand_size` | `len(hand) / 10.0` | Broadcast as scalar plane |
| 25 | `cp_hand_size` | `len(hand) / 10.0` | |
| 26 | `ap_mean_ops` | `mean(ops) / 5.0` | AP hand aggregate quality |
| 27 | `cp_mean_ops` | `mean(ops) / 5.0` | |
| 28 | `vp_track` | `(vp + 20) / 40.0` | Maps [−20, +20] → [0, 1] |
| 29 | `ap_war_status` | 0=Limited War, 1=Total War | |
| 30 | `cp_war_status` | 0=Limited War, 1=Total War | |
| 31 | `active_player` | 0=AP, 1=CP | |

**Card context `(7, 16)`:**
Each slot encodes one card in the active player's hand: `[ops/5, sr/5, is_combat, phase_gate/2, faction, 0…]`. Padded with zeros for empty slots. Flattened to `(112,)` → `Linear(256)` as global context input to GATv2.

---

## 4. Action Space — Factored 4-Sub-Head

**Total flat action space:** 5,341 actions

| Flat range | Type | Count | Sub-head mapping |
|---|---|---|---|
| `[0]` | PASS | 1 | — |
| `[1–110]` | PLAY_AS_EVENT | 110 | Sub-head 1 (card_idx = flat − 1) |
| `[111–440]` | PLAY_AS_OPS | 330 | Sub-head 1 (card) + Sub-head 2 (type: 0=MOVE, 1=ATK, 2=SR) |
| `[441–5340]` | MOVE_UNIT | 4,900 | Sub-head 3 (src = offset//72) + Sub-head 4 (tgt = offset%72) |

**4 policy sub-heads (all Linear from 256-dim graph embedding):**
- Sub-head 1: card selection → `(batch, 110)` logits
- Sub-head 2: action type → `(batch, 3)` logits
- Sub-head 3: source space → `(batch, 72)` logits
- Sub-head 4: target space → `(batch, 72)` logits

**Legal action mask:** `(5,341,)` bool — always static shape, fully `jit`-compatible.
Enforcement: `jnp.where(legal_mask, logits, -jnp.inf)` before softmax.

**Why factored over flat:**
The flat head has ~5,200 illegal logits at any given step (sparse supervision). The factored head decomposes each decision so every sub-head sees a much denser correct/incorrect signal. MCTS sampling requires 4 sequential masked softmaxes instead of 1.

---

## 5. Network Architecture — PoGNet

```
spatial_obs (B, 32, 72)          card_context (B, 7, 16)
      │                                  │
      │ transpose → (B, 72, 32)          │ flatten → (B, 112)
      │                                  │
      ▼                                  ▼
  Linear(32→256)                    Linear(112→256)
  (B, 72, 256)                      LayerNorm → ReLU
      │                              global_ctx (B, 256)
      │◄──────────────────────────────────┘
      │          (injected at every layer)
      ▼
  ┌─────────────────────────────────────┐
  │  × 4  GATv2 Layer                   │
  │  Pre-LayerNorm(h)                   │
  │  h += Linear(global_ctx)[:, None,:] │  ← context injection
  │  h += GATv2Conv(h, adj)             │  ← 4-head, out=256
  │  (residual)                         │
  └─────────────────────────────────────┘
      │
      ▼
  mean_pool(axis=1) → graph_emb (B, 256)
      │
      ├──→ Linear(256→110)            card_logits
      ├──→ Linear(256→3)              action_type_logits
      ├──→ Linear(256→72)             src_logits
      ├──→ Linear(256→72)             tgt_logits
      │
      └──→ Linear(256→128) → ReLU → Linear(128→1) → tanh
                                                     value (B, 1)
```

**GATv2 attention formula:**
```
e_ij = W_a^T · LeakyReLU(W_l·h_i + W_r·h_j)
α_ij = softmax_j( e_ij · mask_ij )   [mask non-edges with −1e9]
h_i' = Σ_j α_ij · W_r·h_j
```

---

## 6. Data Layer — pog_engine.py

**Key dataclasses (all numpy-friendly scalar fields):**

```
MapSpace  — idx, str_id, name, terrain_type(int), vp_value(int), nation_id(int),
            is_fortress(bool), connection_idxs(List[int]),
            trench_level(int), oos_status(bool), controlling_faction(int)

Unit      — unit_id, str_id, nation_id, faction(int), unit_type(int),
            strength(int), max_strength(int), location(int), is_eliminated(bool)

Card      — card_idx(int), str_id, faction(int), ops_value(int), sr_value(int),
            is_combat_card(bool), phase_gate(int), event_text(str)

GameState — turn, action_round, active_player, war_status_ap/cp, vp_track,
            spaces(List[MapSpace]), units(List[Unit]),
            ap_hand/cp_hand/ap_deck/cp_deck/ap_discard/cp_discard (List[int])
```

**Entry points:**
- `GameState.from_json(map_path, cards_path)` — loads and initializes full game state
- `build_observation_tensor(state)` → `(32, 72)` float32
- `build_card_context(state, faction)` → `(7, 16)` float32
- `compute_action_mask(state)` → `(5341,)` bool

---

## 7. Environment — pog_env.py

**Class:** `PogEnv` (PettingZoo AEC-style)

**Key methods:**

| Method | Signature | Notes |
|---|---|---|
| `reset()` | `→ Dict[str, obs]` | Loads JSON, deals AP=6/CP=7 opening hands, shuffles decks |
| `step(action)` | `→ (obs, reward, done, trunc, info)` | Decodes flat action, executes game logic, advances turn |
| `observe(agent)` | `→ {'spatial': (32,72), 'card_context': (7,16)}` | Builds full observation dict |
| `action_mask(agent)` | `→ (5341,) bool` | Phase gate + trench L3 gate + ZOC enforcement |
| `_resolve_crt(...)` | `→ dict` | CRT table lookup with DRM + trench absorption |
| `_check_zoc(space, enemy)` | `→ bool` | Projects ZOC from all enemy combat units |
| `_check_oos(space, faction)` | `→ bool` | BFS supply trace to supply source |
| `render()` | `→ str` | ASCII board state |

**Action decoding:**
```
action == 0              → PASS
1  ≤ action < 111        → PLAY_AS_EVENT  (card_idx = action − 1)
111 ≤ action < 441       → PLAY_AS_OPS   (card_idx = (action−111)//3, op_type = (action−111)%3)
441 ≤ action < 5341      → MOVE_UNIT     (src = (action−441)//72, tgt = (action−441)%72)
```

**Trench CRT rules encoded:**

| Trench Level | DRM | L3 Gate | Step Absorption |
|---|---|---|---|
| 0 | +0 | — | — |
| 1 | +1 | — | — |
| 2 | +1 | — | Absorbs 1st step loss |
| 3 | +2 | Assault card required | Absorbs 1st step loss |

---

## 8. BC Pre-training Curriculum

**3-Phase anti-collapse design:**

| Phase | Epochs | Trained heads | λ_v | Value target |
|---|---|---|---|---|
| 1 | 1–10 | Policy only (value frozen) | 0.0 | N/A |
| 2 | 11–30 | Policy + value | 0.1 | `outcome × 0.99^(T−t)` |
| 3 | 31+ | Full (AlphaZero MCTS targets) | 1.0 | MCTS value estimates |

**Loss (Phase 2+):**
```
L = L_policy + λ_v × L_value
L_policy = mean CrossEntropy(sub-head logits, decomposed expert action)
L_value  = MSE(value_head_output, discounted_outcome)
```

**Expert game record format (JSONL):**
```json
{
  "game_id": "...", "turn": 3, "action_round": 2, "player": "AP",
  "obs_tensor": [[...float32...]],
  "card_context": [[...float32...]],
  "legal_mask": [...bool...],
  "action_taken": 247,
  "outcome": 1
}
```

---

## 9. JAX Implementation Strategy

| Component | JAX Strategy |
|---|---|
| Network forward | `jax.jit` |
| Masked softmax + sampling | `jax.jit` |
| UCB scoring in MCTS | `jax.jit` |
| MCTS simulation loop | `jax.lax.fori_loop` (fixed `N_SIMULATIONS`) |
| MCTS tree storage | Pre-allocated static arrays: `Q[4096, 5341]`, `N[4096, 5341]` |
| PRNG | `jax.random.split` at every simulation step |
| Dirichlet noise on root | `alpha=0.3`, `epsilon=0.25` |
| Python env in JAX | `jax.pure_callback` to preserve `jit` scope |
| Adjacency matrix | Static precomputed arg — never changes during play |
| No BatchNorm | Pre-LayerNorm only — stable under variable-batch MCTS |

---

## 10. Task Board — Current Status

| # | Owner | Task | Status |
|---|---|---|---|
| 1 | Data Architect | Complete `pog_map_graph.json` → 72 spaces | **in_progress** |
| 2 | Data Architect | Fix `pog_cards_db.json` + write `src/data/pog_engine.py` | **in_progress** |
| 3 | Env Builder | `src/env/pog_env.py` PettingZoo state machine | **in_progress** |
| 4 | Env Builder | pytest: CRT + ZOC + OOS | **in_progress** |
| 5 | RL Scientist | `src/rl/network.py` GATv2 + factored policy | **in_progress** |
| 6 | RL Scientist | `src/rl/bc_pipeline.py` + `design_doc.md` | **blocked by #5** |

**Dependency chain:**
```
#1, #2  (Data) → #3 (Env skeleton) → #4 (pytest)
#5 (Network)   → #6 (BC pipeline + design doc)
```

---

## 11. Known Limitations & Next Steps

### Current scope limitations
- **Event card resolution**: Individual card events (`_play_event`) are stubbed — each of the ~110 events needs its own handler
- **SR pathfinding**: Strategic Redeployment range check is simplified (marked legal if unit exists); full rail-path BFS needed
- **Unit placement**: Opening setup (initial unit placement per PoG rules) is not yet implemented
- **Attrition phase**: End-of-turn attrition for OOS units is stubbed
- **Fortification**: Building/destroying trenches (Entrench cards) not yet wired

### After framework completion
1. **Run pytest suite** — `pytest tests/ -v` to verify CRT/ZOC/OOS correctness
2. **Integration test** — `python -m src.rl.network` smoke test for network forward pass
3. **Smoke-run env** — 10-step random rollout to verify reset/step/observe cycle
4. **Collect expert games** — Format VASSAL game logs into JSONL for BC pipeline
5. **BC pre-training** — Phase 1-3 curriculum on expert data
6. **MCTS self-play** — AlphaZero training loop with pre-allocated JAX tree

---

## 12. File Checksums / Completion Markers

*Updated automatically when agents write files.*

| File | Status | Notes |
|---|---|---|
| `pog_map_graph.json` | pending agent | 72 spaces, bidirectional adjacency |
| `pog_cards_db.json` | pending agent | All "Unknown" phases corrected |
| `src/data/pog_engine.py` | pending agent | Dataclasses + tensor builders |
| `src/env/pog_env.py` | pending agent | Full PettingZoo AEC env |
| `tests/test_crt.py` | pending agent | |
| `tests/test_zoc.py` | pending agent | |
| `tests/test_oos.py` | pending agent | |
| `src/rl/network.py` | pending agent | GATv2 + 4-sub-head |
| `src/rl/bc_pipeline.py` | pending agent | 3-phase curriculum |
| `src/rl/design_doc.md` | pending agent | Full design doc |
