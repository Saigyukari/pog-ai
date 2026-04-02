#!/usr/bin/env python3
"""
eval/tournament.py — Round-robin checkpoint evaluation on the CPU env.

Loads one or more checkpoint `.pkl` files, runs pairwise matches on `PogEnv`,
computes simple Elo estimates, prints a table, and writes `eval/results.csv`.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.pog_env import FACTION_AP, PogEnv
from src.rl.mcts import MCTS
from src.rl.network import PoGNet, load_adjacency_matrix

N_ACTIONS = 5341


def _greedy_action(env: PogEnv, agent: "AgentSpec", legal: np.ndarray, infer_fn) -> int:
    """Single JIT'd forward pass — no MCTS, no deepcopy. ~50x faster than MCTS."""
    obs = env._build_spatial_obs()
    card_ctx = env._build_card_context(env.active_player)
    heads, _ = infer_fn(agent.params, jnp.array(obs), jnp.array(card_ctx))
    card_l, atype_l, src_l, tgt_l = [np.array(h[0]) for h in heads]

    scores = np.full(N_ACTIONS, -1e9, dtype=np.float32)
    scores[0] = 0.0
    scores[1:111] = card_l + atype_l[0]
    ops = (card_l[:, None] + atype_l[None, :]).reshape(-1)
    scores[111:441] = ops[:330]
    moves = (src_l[:, None] + tgt_l[None, :]).reshape(-1)
    scores[441:] = moves[: N_ACTIONS - 441]
    scores[~legal] = -1e9
    return int(np.argmax(scores))


@dataclass
class AgentSpec:
    name: str
    kind: str
    params: object | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint round-robin evaluation.")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint .pkl files or directories containing them.")
    parser.add_argument("--games", type=int, default=200, help="Games per pairing.")
    parser.add_argument("--mcts-sims", type=int, default=0, help="MCTS simulations per move. 0=greedy (fastest).")
    parser.add_argument("--map", dest="map_json", default="pog_map_graph.json", help="Board graph path.")
    parser.add_argument("--cards", dest="cards_json", default="pog_cards_db.json", help="Card db path.")
    parser.add_argument("--results-csv", default="eval/results.csv", help="CSV output path.")
    parser.add_argument("--include-random", action="store_true", help="Include a random-policy anchor with Elo fixed at 0.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    return parser.parse_args()


def iter_checkpoint_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.pkl")))
        elif path.suffix == ".pkl" and path.is_file():
            files.append(path)
    deduped = []
    seen = set()
    for path in files:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def load_agent(path: Path) -> AgentSpec:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"{path} is not a supported checkpoint payload.")
    return AgentSpec(name=path.stem, kind="model", params=payload["params"])


def choose_action(env: PogEnv, agent: AgentSpec, model: PoGNet, adj: jnp.ndarray,
                  mcts_sims: int, rng_seed: int, infer_fn=None) -> int:
    active_name = env._agent_name(env.active_player)
    legal = env.action_mask(active_name)
    legal_idx = np.flatnonzero(legal)
    if len(legal_idx) == 0:
        return 0

    if agent.kind == "random":
        return int(np.random.default_rng(rng_seed).choice(legal_idx))

    if mcts_sims == 0 and infer_fn is not None:
        return _greedy_action(env, agent, legal, infer_fn)

    mcts = MCTS(model=model, params=agent.params, adj=adj, n_simulations=mcts_sims, depth_limit=4, temperature=0.0)
    policy = mcts.search(env, jax.random.PRNGKey(rng_seed))
    action = int(np.argmax(policy))
    if not legal[action]:
        action = int(legal_idx[0])
    return action


def play_game(ap_agent: AgentSpec, cp_agent: AgentSpec, model: PoGNet, adj: jnp.ndarray,
              args: argparse.Namespace, seed: int, infer_fn=None) -> int:
    env = PogEnv(args.map_json, args.cards_json)
    env.reset(seed=seed)

    for ply in range(256):
        active_agent = ap_agent if env.active_player == FACTION_AP else cp_agent
        action = choose_action(env, active_agent, model, adj, args.mcts_sims, seed + ply, infer_fn)
        _, reward, done, _, _ = env.step(action)
        if any(done.values()):
            if reward["AP"] > reward["CP"]:
                return 1
            if reward["CP"] > reward["AP"]:
                return -1
            return 0

    if env.vp_track > 0:
        return 1
    if env.vp_track < 0:
        return -1
    return 0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(ratings: dict[str, float], name_a: str, name_b: str, score_a: float, k: float = 20.0) -> None:
    if name_a == "random" or name_b == "random":
        movable = name_b if name_a == "random" else name_a
        opponent = name_a if name_a == "random" else name_b
        expected = expected_score(ratings[movable], ratings[opponent])
        actual = 1.0 - score_a if name_a == "random" else score_a
        ratings[movable] += k * (actual - expected)
        ratings["random"] = 0.0
        return
    ea = expected_score(ratings[name_a], ratings[name_b])
    eb = expected_score(ratings[name_b], ratings[name_a])
    ratings[name_a] += k * (score_a - ea)
    ratings[name_b] += k * ((1.0 - score_a) - eb)


def main() -> int:
    args = parse_args()
    checkpoint_files = iter_checkpoint_files(args.checkpoints)
    if len(checkpoint_files) < 2 and not args.include_random:
        raise ValueError("Provide at least two checkpoint files, or use --include-random.")

    agents = [load_agent(path) for path in checkpoint_files]
    if args.include_random:
        agents.append(AgentSpec(name="random", kind="random"))

    model = PoGNet(hidden_dim=128, n_gat_layers=6)
    adj = load_adjacency_matrix(args.map_json)

    # Build a single JIT'd inference function shared across all games.
    # Compiled once on first call; subsequent calls hit the cache.
    @jax.jit
    def infer_fn(params, obs, card_ctx):
        heads, val = model.apply(params, obs[None], card_ctx[None], adj)
        return heads, val

    ratings = {agent.name: 0.0 for agent in agents}
    if "random" in ratings:
        ratings["random"] = 0.0

    rows = []
    seed = args.seed
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent_a = agents[i]
            agent_b = agents[j]
            wins_a = 0
            wins_b = 0
            draws = 0

            for game_idx in range(args.games):
                swap = game_idx % 2 == 1
                ap_agent = agent_b if swap else agent_a
                cp_agent = agent_a if swap else agent_b
                result = play_game(ap_agent, cp_agent, model, adj, args, seed + game_idx, infer_fn)
                score_for_a = 0.5
                if result == 1:
                    winner = ap_agent.name
                    score_for_a = 1.0 if winner == agent_a.name else 0.0
                elif result == -1:
                    winner = cp_agent.name
                    score_for_a = 1.0 if winner == agent_a.name else 0.0
                else:
                    draws += 1

                if score_for_a == 1.0:
                    wins_a += 1
                elif score_for_a == 0.0:
                    wins_b += 1

                update_elo(ratings, agent_a.name, agent_b.name, score_for_a)

            rows.append(
                {
                    "agent_a": agent_a.name,
                    "agent_b": agent_b.name,
                    "games": args.games,
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "draws": draws,
                    "score_a": (wins_a + 0.5 * draws) / args.games,
                    "elo_a": ratings[agent_a.name],
                    "elo_b": ratings[agent_b.name],
                }
            )

    ordered = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    print("Name                Elo")
    for name, elo in ordered:
        print(f"{name:18s} {elo:7.1f}")

    out_path = Path(args.results_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["agent_a", "agent_b", "games", "wins_a", "wins_b", "draws", "score_a", "elo_a", "elo_b"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
