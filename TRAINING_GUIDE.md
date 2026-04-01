# Paths of Glory AI — Training Guide
*Hardware: RTX 5060 8 GB · CUDA 12.9 · JAX 0.6.2*

---

## Overview — 3-Phase Training Pipeline

```
Human records  ──►  Phase 1: Behavioral Cloning  ──►  Phase 2: Self-Play (AlphaZero)  ──►  Evaluation
(your games)         (learn from experts)               (surpass humans)                    (Elo / win rate)
```

---

## Step 0 — Fix GPU Support (Do This First)

JAX currently falls back to CPU because `jaxlib-cuda` is not installed.

```bash
# Uninstall CPU-only jaxlib, install CUDA 12 version
pip uninstall jaxlib -y
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Verify:**
```bash
python3 -c "import jax; print(jax.devices())"
# Should print: [CudaDevice(id=0)]
```

With the RTX 5060 (8 GB VRAM) you can run:
- Batch size 64–128 for BC training
- 200–400 MCTS simulations per move during self-play
- ~1,000–2,000 self-play games per day

---

## Step 1 — Prepare Your Human Game Records

### What format are your records in?

**If VASSAL save files (.vsav):**
- Open VASSAL, load your game, use *File → Export Chat Log* to save the action log as `.txt`
- Each line will be a chat/action entry like: `AP: Plays British Reinforcements as OPS`

**If hand-recorded notation:**
Use this plain-text format (one action per line):
```
AP PLAYS AP-14 AS OPS MOVE FROM AMIENS TO PARIS
CP PLAYS CP-07 AS EVENT
AP PLAYS AP-18 AS OPS ATTACK FROM SEDAN TO LIEGE
AP PASS
CP PLAYS CP-02 AS OPS SR FROM BERLIN TO WARSAW
```

**If you have JSON exports:**
```json
[
  {"player":"AP","card":"AP-14","type":"OPS_MOVE","from":"AMIENS","to":"PARIS"},
  {"player":"CP","card":"CP-01","type":"EVENT"}
]
```

### Name your files with the outcome
```
game001_APwin.txt
game002_CPwin.txt
game003_draw.json
```

### Convert to JSONL
```bash
python3 -m src.data.convert_records \
  --input  games/raw/ \
  --output data/expert_games.jsonl \
  --map    pog_map_graph.json \
  --cards  pog_cards_db.json
```

**Quality checks before training:**
```bash
# Count records
wc -l data/expert_games.jsonl

# Check for illegal actions (should be 0 or near-0)
python3 -c "
import json
records = [json.loads(l) for l in open('data/expert_games.jsonl')]
illegal = sum(1 for r in records if not r['legal_mask'][r['action_taken']])
print(f'{illegal}/{len(records)} illegal actions ({100*illegal/len(records):.1f}%)')
"
```

**Minimum recommended records:** 500 complete games ≈ ~50,000–100,000 (state, action) pairs.

---

## Step 2 — Behavioral Cloning (BC) Pre-Training

BC teaches the network to **imitate human play** before self-play begins. This prevents the network from starting self-play with completely random behaviour.

### Run BC training

```bash
python3 train_bc.py \
  --data   data/expert_games.jsonl \
  --map    pog_map_graph.json \
  --cards  pog_cards_db.json \
  --epochs 50 \
  --batch  64 \
  --lr     1e-4 \
  --save   checkpoints/bc_final.pkl
```

### What happens in 3 phases

| Phase | Epochs | What's trained | λ_v | Purpose |
|---|---|---|---|---|
| 1 | 1–10 | Policy only (value frozen) | 0.0 | Build meaningful card/space features |
| 2 | 11–30 | Policy + value | 0.1 | Bootstrap value from game outcomes |
| 3 | 31–50 | Full (MCTS-ready targets) | 1.0 | Align with AlphaZero loss format |

### Expected loss curve

```
Epoch   1 [Phase 1]  train_loss=2.800   ← random init
Epoch  10 [Phase 1]  train_loss=1.200   ← learning card selection
Epoch  20 [Phase 2]  train_loss=0.800   ← value head converging
Epoch  50 [Phase 3]  train_loss=0.500   ← plateau (human-level imitation)
```

If loss stays above 2.0 after epoch 20: your data is too small or labels have errors.

### How to do BC better

| Technique | Impact | Notes |
|---|---|---|
| **More games** | High | 1,000+ games significantly improves policy quality |
| **Data augmentation** | Medium | Mirror board symmetries where applicable |
| **Curriculum ordering** | Medium | Train on later-game positions first (richer signals) |
| **Label smoothing** | Low | `ε=0.1` prevents overconfident illegal-action collapse |
| **Learning rate schedule** | Medium | Cosine decay from 1e-4 → 1e-6 over 50 epochs |

---

## Step 3 — Self-Play (AlphaZero)

After BC, the network plays **against itself** using MCTS to generate new training data that exceeds human-level play.

### Run self-play

```bash
python3 train_selfplay.py \
  --bc_checkpoint  checkpoints/bc_final.pkl \
  --map            pog_map_graph.json \
  --cards          pog_cards_db.json \
  --iterations     100 \
  --games_per_iter 50 \
  --mcts_sims      400 \
  --save_dir       checkpoints/selfplay/
```

### Self-play loop (one iteration)

```
1. Generate 50 games using current network + MCTS (400 sims/move)
2. Add generated (obs, policy, value) triples to replay buffer
3. Sample 512 random positions from buffer
4. Train network for 1 epoch on sampled data
5. Evaluate new network vs previous checkpoint (100-game match)
6. If win rate > 55%: promote new checkpoint
7. Repeat
```

### MCTS configuration

| Parameter | Value | Notes |
|---|---|---|
| `n_simulations` | 400 | More = stronger but slower. 200 minimum |
| `c_puct` | 1.5 | Exploration constant. Increase if play is too conservative |
| `dirichlet_alpha` | 0.3 | Root noise. Lower = more focused search |
| `dirichlet_eps` | 0.25 | Noise weight at root |
| `temperature` | 1.0 for first 20 moves, then 0 | Diversity early, deterministic late |

### How to make self-play better

| Technique | Impact | Notes |
|---|---|---|
| **Replay buffer** | Critical | Keep last 500K positions; sample uniformly |
| **Network ELO tracking** | High | Only promote if new net is clearly better |
| **Parallel self-play** | High | Use `multiprocessing` for game generation |
| **Temperature schedule** | Medium | High temp early forces exploration of rare positions |
| **Resign threshold** | Medium | Resign when value < −0.9 to save time on lost games |

---

## Step 4 — Hardware Requirements

### Minimum (what you have now)

| Component | Your Spec | Capability |
|---|---|---|
| GPU | RTX 5060 8 GB | BC training ✅, Self-play ✅ (slower) |
| VRAM | 8 GB | Batch 64 for BC, 32 for self-play |
| CUDA | 12.9 | Supported by JAX |
| RAM | (check with `free -h`) | Need ≥16 GB for replay buffer |

**Estimated training time on RTX 5060:**

| Task | Time estimate |
|---|---|
| BC Phase 1–3 (50 epochs, 50K records) | ~2–4 hours |
| Self-play: 1 iteration (50 games × 400 MCTS sims) | ~3–6 hours |
| 100 self-play iterations (reach superhuman) | ~2–4 weeks |

### Recommended upgrades

| Hardware | Why | Gain |
|---|---|---|
| **RTX 4090 / RTX 5090** (24 GB VRAM) | 3× bigger batches, 3× faster BC | 3–5× speedup |
| **A100 80 GB** (cloud: ~$3/hr) | Train BC in 30 min; self-play overnight | 15× speedup |
| **TPU v3-8** (Google TPU Research Cloud — free for research) | JAX's native hardware | 20–30× speedup |
| **16+ CPU cores** | Parallel game generation (env runs on CPU) | 4–8× self-play speedup |

### Cloud options (cost/performance)

| Provider | Instance | VRAM | Cost | Best for |
|---|---|---|---|---|
| Vast.ai | RTX 4090 × 1 | 24 GB | ~$0.40/hr | BC training |
| RunPod | A100 80 GB | 80 GB | ~$2.50/hr | Full self-play |
| Google Colab Pro+ | A100 | 40 GB | $50/month | Experimentation |
| TPU Research Cloud | TPU v3-8 | — | Free (apply) | Production training |

---

## Step 5 — Evaluation & Testing

### Test 1: Sanity check (run this now)

```bash
cd /home/saigyukari/PoGAIV1
python3 -c "
from src.env.pog_env import PogEnv
env = PogEnv('pog_map_graph.json', 'pog_cards_db.json')
obs = env.reset(seed=42)
mask = env.action_mask('AP')
legal = [i for i,m in enumerate(mask) if m]
print(f'Reset OK. {len(legal)} legal actions for AP on turn 1.')
obs, rew, done, trunc, info = env.step(legal[0])
print(f'Step OK. VP={env.vp_track}, turn={env.turn}')
"
```

### Test 2: Random rollout (measures env correctness)

```bash
python3 -c "
from src.env.pog_env import PogEnv
import random, numpy as np
env = PogEnv('pog_map_graph.json', 'pog_cards_db.json')
env.reset(seed=0)
for step in range(100):
    player = 'AP' if env.active_player == 0 else 'CP'
    mask   = env.action_mask(player)
    legal  = np.where(mask)[0]
    action = random.choice(legal.tolist())
    _, _, done, _, _ = env.step(action)
    if any(done.values()):
        print(f'Game ended at step {step}')
        break
print(f'100 random steps OK. VP={env.vp_track}')
"
```

### Test 3: Network forward pass

```bash
python3 -m src.rl.network
# Should print: Smoke test PASSED.
```

### Test 4: After BC training — policy quality

```python
# Measure Top-1 accuracy on validation set
# (what % of expert moves does the network predict exactly?)
# Target: > 30% Top-1, > 60% Top-5

from src.rl.bc_pipeline import load_expert_games, make_bc_batches
# ... load model, iterate val set, compute argmax match rate
```

**Interpreting BC accuracy:**

| Top-1 Accuracy | Meaning |
|---|---|
| < 10% | Random / training failed |
| 10–25% | Learning basic patterns |
| 25–40% | Strong human imitation |
| > 40% | Excellent (PoG is highly tactical — 40%+ is near-human) |

### Test 5: After self-play — Elo rating

Run a tournament between model checkpoints:

```bash
python3 eval/tournament.py \
  --checkpoint_dir checkpoints/selfplay/ \
  --n_games        100 \
  --mcts_sims      200
```

**Elo milestones:**

| Milestone | Approximate Elo gap | What it means |
|---|---|---|
| BC model vs random | +400 | Successfully learned legal moves |
| Self-play iter 10 vs BC | +100–200 | Starting to find non-human strategies |
| Self-play iter 50 vs BC | +400–600 | Clearly superhuman on tactical level |
| Self-play iter 100 vs BC | +800+ | Strategic mastery (if env is complete) |

---

## File Map — Training Scripts Needed

```
PoGAIV1/
├── train_bc.py                ← run BC training          [TODO: create]
├── train_selfplay.py          ← run self-play loop       [TODO: create]
├── eval/
│   └── tournament.py          ← Elo evaluation           [TODO: create]
├── src/
│   ├── data/
│   │   └── convert_records.py ← data pipeline            ✅ DONE
│   ├── env/
│   │   └── pog_env.py         ← game environment         ✅ DONE
│   └── rl/
│       ├── network.py         ← GATv2 + factored policy  ✅ DONE
│       ├── bc_pipeline.py     ← BC loss + training loop  ✅ DONE
│       └── mcts.py            ← MCTS search              ✅ DONE
└── data/
    └── expert_games.jsonl     ← your converted records   [TODO: you convert]
```

---

## Recommended Order of Operations

```
TODAY
  1.  pip install jax[cuda12]          ← unlock GPU
  2.  pytest tests/ -v                 ← confirm 24 tests pass
  3.  python3 -m src.rl.network        ← confirm network runs

THIS WEEK
  4.  Convert your human records       ← convert_records.py
  5.  python3 train_bc.py (50 epochs) ← ~2-4 hours on RTX 5060
  6.  Check BC accuracy on val set    ← target > 25% Top-1

THIS MONTH
  7.  Run 20 self-play iterations     ← ~1 week wall-clock
  8.  Evaluate vs BC baseline         ← Elo tournament
  9.  Complete event card handlers    ← src/env/pog_env.py _play_event()
  10. Full 100-iteration self-play    ← ~3-4 weeks
```

---

## Known Gaps To Fix Before Full Training

These are stubs in the current env that must be implemented for complete self-play:

| Gap | File | Impact |
|---|---|---|
| Event card resolution (`_play_event`) | `src/env/pog_env.py` | High — ~40% of cards are events |
| SR rail pathfinding | `src/env/pog_env.py` | Medium — affects strategic movement |
| Opening setup (unit placement) | `src/env/pog_env.py` | High — game starts wrong |
| Attrition (OOS step losses) | `src/env/pog_env.py` | Medium — end-of-turn supply penalty |
| War status advancement | `src/env/pog_env.py` | High — Total War gate never triggers |
| Replacement points (RPs) | `src/env/pog_env.py` | High — units never reinforce |

**Practical advice:** Start BC training now with the current env — the network will learn map patterns and card preferences even with incomplete rules. Then layer in the remaining rules as you complete them, and fine-tune the BC checkpoint rather than starting from scratch.
