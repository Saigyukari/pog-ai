# claude2gpt

## Review of Phase 3 + 4 work — PASSED

Reviewed `train_selfplay.py`, `src/rl/mcts.py` (jax_mcts_search), and `eval/tournament.py`.

### What is correct ✅

- `jax_mcts_search` uses `jax.lax.fori_loop` (not Python loop) → vmappable ✓
- `_rollout_value` does depth-limited forward search with `jax.lax.fori_loop` ✓
- `train_selfplay.py` auto-detects single vs multi-GPU, sets n_actors/buffer accordingly ✓
- `get_search_params(total_games)` staged schedule matches design_doc.md §2a ✓
- `replay.push(..., policy=policies)` correctly passes MCTS visit-count distributions ✓
- `alphazero_loss` unpacks batch as `(obs, card_ctx, legal_mask, target_policy, action, outcome, done)` —
  matches `replay_buffer.sample()` 7-item return order ✓
- `eval/tournament.py` alternates AP/CP sides, anchors random at Elo=0, saves CSV ✓

### One fix needed ⚠️

**File:** `README.md`

The README references `setup_env.sh` which does not exist. Also the example BC command
uses `--batch-size 32` (5060 config) rather than the cluster config.

Fix both:
```markdown
# Replace the Quick Start section with:

## Quick Start

### Cluster (H200 × 2)
python train_bc.py --data data/training/expert_games.jsonl --epochs 10 --batch-size 4096

### Local (RTX 5060)
python train_bc.py --data data/training/expert_games.jsonl --epochs 10 --batch-size 32
```

Remove the `setup_env.sh` reference entirely (no such file exists).

---

## Next task for GPT: Task 5.1 — Write `play.py`

**Why this task:** BC training is running on the cluster. All training infrastructure
is done. The user's final goal is to play against the AI on their RTX 5060. This task
has zero dependency on the BC checkpoint being ready — it works with any `.pkl` file,
including a random-initialized one (the interface just won't play well until a real
checkpoint is ready).

**File to create:** `play.py` (project root)

### What it must do

1. Load a checkpoint: `python play.py --checkpoint checkpoints/bc/epoch_010.pkl`
2. Ask the user which side they want to play (AP or CP)
3. Loop:
   - Print a compact ASCII board state showing:
     - VP track (current value)
     - Turn / Action Round
     - Cards in hand (numbered list)
     - Legal actions (numbered list, grouped: PASS / EVENTS / OPS / MOVES)
   - If it's the human's turn: read a number from stdin, execute the action
   - If it's the AI's turn: run MCTS, print the chosen action, execute it
4. Print the game result at the end

### CLI flags

```
--checkpoint PATH   required; .pkl checkpoint file
--side {AP,CP}      which side the human plays (default: ask at startup)
--mcts-sims N       AI MCTS simulations per move (default 64; fast on 5060)
--depth N           AI search depth (default 4)
--map PATH          board graph (default pog_map_graph.json)
--cards PATH        card db (default pog_cards_db.json)
--seed N            RNG seed (default 0)
```

### Key implementation notes

- Use `src/env/pog_env.py` (CPU PettingZoo env) — NOT jax_env.py. The CPU env has
  familiar Python objects, easy to print; jax_env is for training not display.
- Use the existing `MCTS` class from `src/rl/mcts.py` (the Python loop version).
- For the board display, a minimal representation is enough — no need for a full
  ASCII map. A table of spaces with their control/unit status is fine.
- Legal action display: group by type (PASS / EVENT card_name / OPS card_name /
  MOVE src→tgt) so the human can read them. Number them 0..N for input.
- Handle invalid input gracefully (re-prompt on bad number).
- When pog_env.py has no starting units (current state), the game is still playable
  for card events only — do not block on this.

### Acceptance criteria

- `python play.py --checkpoint <any_pkl>` launches without error
- Human can select actions by number and the game progresses
- AI responds within 5 seconds per move on RTX 5060 with default settings
- Game ends and prints result (AP wins / CP wins / Draw)

---

## Reminder: do not start train_selfplay.py until BC checkpoint arrives

The cluster is running BC training now. Once `checkpoints/bc/epoch_010.pkl` is
available, the self-play command will be:

### Cluster (H200 × 2):
```bash
python train_selfplay.py \
  --bc-checkpoint checkpoints/bc/epoch_010.pkl \
  --n-actors 256 \
  --iterations 100 \
  --checkpoint-dir checkpoints/selfplay
```

### Local (RTX 5060):
```bash
python train_selfplay.py \
  --bc-checkpoint checkpoints/bc/epoch_010.pkl \
  --n-actors 16 \
  --buffer-capacity 50000 \
  --batch-size 256 \
  --iterations 100 \
  --checkpoint-dir checkpoints/selfplay
```
