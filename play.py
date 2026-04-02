#!/usr/bin/env python3
"""
play.py — Play Paths of Glory against the current AI checkpoint.

Uses the CPU env and the existing Python MCTS path for interactive play.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from src.env.pog_env import ACT_MOVE_START, ACT_OPS_START, FACTION_AP, FACTION_CP, PogEnv
from src.rl.mcts import MCTS
from src.rl.network import PoGNet, load_adjacency_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play against the PoG AI.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pkl file.")
    parser.add_argument("--side", choices=["AP", "CP"], help="Human side. If omitted, ask interactively.")
    parser.add_argument("--mcts-sims", type=int, default=64, help="AI MCTS simulations per move.")
    parser.add_argument("--depth", type=int, default=4, help="AI search depth.")
    parser.add_argument("--map", dest="map_json", default="pog_map_graph.json", help="Board graph path.")
    parser.add_argument("--cards", dest="cards_json", default="pog_cards_db.json", help="Card db path.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    return parser.parse_args()


def load_params(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"{path} is not a supported checkpoint payload.")
    return payload["params"]


def ask_side() -> str:
    while True:
        value = input("Choose your side [AP/CP]: ").strip().upper()
        if value in {"AP", "CP"}:
            return value
        print("Enter AP or CP.")


def faction_name(faction: int) -> str:
    return "AP" if faction == FACTION_AP else "CP"


def describe_action(env: PogEnv, action: int) -> str:
    if action == 0:
        return "PASS"
    if action < ACT_OPS_START:
        card_idx = action - 1
        card = env._cards_db[card_idx] if 0 <= card_idx < len(env._cards_db) else {"str_id": f"CARD_{card_idx}"}
        return f"EVENT {card['str_id']}"
    if action < ACT_MOVE_START:
        offset = action - ACT_OPS_START
        card_idx = offset // 3
        op_type = offset % 3
        op_name = ["OPS_MOVE", "OPS_ATTACK", "OPS_SR"][op_type]
        card = env._cards_db[card_idx] if 0 <= card_idx < len(env._cards_db) else {"str_id": f"CARD_{card_idx}"}
        return f"{op_name} {card['str_id']}"
    offset = action - ACT_MOVE_START
    src = offset // 72
    tgt = offset % 72
    src_name = env._spaces_db[src]["id"] if src < len(env._spaces_db) else str(src)
    tgt_name = env._spaces_db[tgt]["id"] if tgt < len(env._spaces_db) else str(tgt)
    return f"MOVE {src_name} -> {tgt_name}"


def grouped_actions(env: PogEnv, legal_actions: list[int]) -> list[tuple[str, list[int]]]:
    groups = {"PASS": [], "EVENTS": [], "OPS": [], "MOVES": []}
    for action in legal_actions:
        if action == 0:
            groups["PASS"].append(action)
        elif action < ACT_OPS_START:
            groups["EVENTS"].append(action)
        elif action < ACT_MOVE_START:
            groups["OPS"].append(action)
        else:
            groups["MOVES"].append(action)
    return [(name, groups[name]) for name in ("PASS", "EVENTS", "OPS", "MOVES") if groups[name]]


def print_board(env: PogEnv) -> None:
    print()
    print(f"Turn {env.turn}  Action Round {env.action_round}  Active {faction_name(env.active_player)}  VP {env.vp_track:+d}")
    print("Board:")
    rows = []
    for idx in range(env._n_spaces):
        units_here = [u for u in env._units if u["location"] == idx and not u["is_eliminated"]]
        trench = int(env._trench_levels[idx])
        ctrl = ["AP", "CP", "--"][int(env._space_control[idx])]
        if units_here or trench > 0:
            rows.append(f"  [{idx:02d}] {env._spaces_db[idx]['id']:<20} ctrl={ctrl} trench={trench} units={len(units_here)}")
    if rows:
        for row in rows[:24]:
            print(row)
        if len(rows) > 24:
            print(f"  ... {len(rows) - 24} more occupied spaces")
    else:
        print("  (no units on board in current env baseline)")


def print_hand(env: PogEnv, side: str) -> None:
    hand = env._ap_hand if side == "AP" else env._cp_hand
    print(f"{side} hand:")
    for i, card_idx in enumerate(hand, start=1):
        if 0 <= card_idx < len(env._cards_db):
            card = env._cards_db[card_idx]
            print(
                f"  [{i}] {card['str_id']}"
                f"  ops={card['ops']} sr={card['sr']}"
                f"  combat={int(card['is_combat_card'])}"
            )


def print_legal_actions(env: PogEnv, legal_actions: list[int]) -> dict[int, int]:
    numbered: dict[int, int] = {}
    display_idx = 0
    print("Legal actions:")
    for group_name, actions in grouped_actions(env, legal_actions):
        print(f"  {group_name}:")
        for action in actions:
            print(f"    [{display_idx}] {describe_action(env, action)}")
            numbered[display_idx] = action
            display_idx += 1
    return numbered


def choose_human_action(env: PogEnv, side: str) -> int | None:
    legal_mask = env.action_mask(side)
    legal_actions = [i for i, flag in enumerate(legal_mask) if flag]
    action_map = print_legal_actions(env, legal_actions)
    while True:
        try:
            value = input("Choose action number: ").strip()
        except EOFError:
            return None
        try:
            idx = int(value)
        except ValueError:
            print("Enter a valid number.")
            continue
        if idx in action_map:
            return action_map[idx]
        print("Action number out of range.")


def choose_ai_action(env: PogEnv, model: PoGNet, params, adj, args: argparse.Namespace) -> int:
    mcts = MCTS(
        model=model,
        params=params,
        adj=adj,
        n_simulations=args.mcts_sims,
        depth_limit=args.depth,
        temperature=0.0,
    )
    policy = mcts.search(env, jax.random.PRNGKey(args.seed + env.turn * 31 + env.action_round))
    action = int(np.argmax(policy))
    if not env.action_mask(faction_name(env.active_player))[action]:
        legal_actions = np.flatnonzero(env.action_mask(faction_name(env.active_player)))
        action = int(legal_actions[0]) if len(legal_actions) else 0
    print(f"AI chooses: {describe_action(env, action)}")
    return action


def print_result(env: PogEnv) -> None:
    if env.vp_track > 0:
        print("Game result: AP wins")
    elif env.vp_track < 0:
        print("Game result: CP wins")
    else:
        print("Game result: Draw")


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    human_side = args.side or ask_side()
    ai_side = "CP" if human_side == "AP" else "AP"
    params = load_params(checkpoint_path)
    adj = load_adjacency_matrix(args.map_json)
    model = PoGNet(hidden_dim=128, n_gat_layers=6)

    env = PogEnv(args.map_json, args.cards_json)
    env.reset(seed=args.seed)

    print(f"Human: {human_side}  AI: {ai_side}")

    for _ in range(256):
        active_side = faction_name(env.active_player)
        print_board(env)
        print_hand(env, active_side)

        if active_side == human_side:
            action = choose_human_action(env, active_side)
            if action is None:
                print("Input ended. Exiting game.")
                return 0
        else:
            action = choose_ai_action(env, model, params, adj, args)

        _, reward, done, _, _ = env.step(action)
        if any(done.values()):
            print_board(env)
            print_result(env)
            return 0

    print("Game ended by move cap.")
    print_result(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
