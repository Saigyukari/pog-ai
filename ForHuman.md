Your concerns — answered directly

  1. Abandon deep rollouts?

  Yes, completely. The math is fatal: PoG has ~2 combats per action round × 7 CRT outcomes each. At depth 10 that's 7^20 ≈ 10^17 branches.
  Even with sampling, the variance of any Monte Carlo estimate over that space buries the value signal. TD-Gammon proved in 1995 that the
  right answer is to train a value network to absorb stochastic EV and use it as the leaf — no rollout at all. PoG follows exactly that
  template.

  2. Literature validation:

  ┌─────────────────────┬──────────────────────────────────────────────────────┬───────────────────────────────────────────────────────┐
  │       Problem       │                    SOTA solution                     │                     What we adopt                     │
  ├─────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Imperfect info      │ Libratus: CFR over hand distributions; DeepNash:     │ Implicit EV — value head trained on self-play under   │
  │ (hidden cards)      │ zero MCTS, pure Nash regularization                  │ true hidden-info conditions. No hand sampling in      │
  │                     │                                                      │ tree.                                                 │
  ├─────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Heavy stochasticity │ TD-Gammon: value network, no rollouts; Blood Bowl:   │ Single-sample path per simulation (outcome sampling). │
  │                     │ chance nodes (only works for single-event actions)   │  Value head at leaf.                                  │
  ├─────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Massive action      │ AlphaStar: factored auto-regressive heads            │ Already done — our 4-sub-head policy is correct.      │
  │ space               │                                                      │                                                       │
  └─────────────────────┴──────────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘

  3. Final MCTS design:

  - Depth 3–5 (one tactical loop)
  - Single CRT sample per simulation path (not tree-branched)
  - No PIMC, no CFR — value head handles hidden info implicitly
  - MAX_NODES drops from 4096 → 512 (truncated search needs far fewer)

  4. 2× H200 split:

  - GPU 0 (Learner): batch size 4096, replay buffer 500K records, param broadcast every 256 steps
  - GPU 1 (Actor): jax.vmap over 256 parallel game states, all 256 MCTS trees share one batched network call — ~15,000 new training records
  per second
  - Memory cost: ~13 GB of 141 GB on the Actor — massive headroom

  The one thing BC training still needs: train_bc.py.

   Review result: PASS with one fix

  train_bc.py is well-written. All imports resolve, pmap logic is correct, phase curriculum is right. One issue:

  ┌─────┬──────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
  │     │                   What                   │                                Fix                                 │
  ├─────┼──────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ ⚠️   │ PoGNet() defaults = 1.19M params (too    │ Change line 324 to PoGNet(hidden_dim=128, n_gat_layers=6) = 6.58M  │
  │     │ small)                                   │ params                                                             │
  └─────┴──────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘

  Everything else is green. The fix is one line — GPT can apply it before running on H200.

  What GPT does next (in claude2gpt.md):
  1. Apply the model-size fix → run BC on H200
  2. Write src/env/jax_env.py — pure JAX state machine (the big Phase 2 task, needed before self-play can start)
  3. Write src/rl/replay_buffer.py — VRAM replay buffer, no JSON in RL phase

   check gpt2claude.md file and the related files, update .md files , check the process in ROADMAP, give the task
  that no need the recent BC result to gpt                                                                          
───────────────────────────────────────────
gemini --resume c1c82cc5-ad2d-42f4-a67c-3eceb263b63f  
 codex resume 019d4a9d-ae9a-7e23-be75-50f9182f7ba4
 claude --resume d9a096c7-796b-49df-8ac5-972bebf13648