"""
convert_records.py — Convert human game records to JSONL for BC pre-training.

Supports two input formats:
  1. VASSAL save files (.vsav) — exported as action logs via VASSAL's chat/log
  2. Plain-text notation (hand-recorded) — one action per line

Output: JSONL file, one step per line, format:
  {game_id, turn, action_round, player, obs_tensor, card_context,
   legal_mask, action_taken, outcome}

Usage:
  python -m src.data.convert_records \
      --input  games/raw/                # dir of .txt or .json game logs
      --output data/expert_games.jsonl  # output JSONL
      --map    pog_map_graph.json
      --cards  pog_cards_db.json
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# ── project imports ──────────────────────────────────────────────────
from src.env.pog_env import PogEnv
from src.data.pog_engine import (
    N_ACTIONS, N_SPACES, FACTION_AP, FACTION_CP,
    ACT_EVENT_START, ACT_OPS_START, ACT_MOVE_START,
)

# Alias
ACT_EVENT_START = 1
ACT_OPS_START   = 111
ACT_MOVE_START  = 441


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Parse raw game logs into a universal intermediate format
# ─────────────────────────────────────────────────────────────────────

class RawMove:
    """One parsed move from a human game log."""
    __slots__ = ["player", "card_id", "action_type", "src", "tgt", "raw_text"]

    def __init__(self, player: str, card_id: str, action_type: str,
                 src: str = "", tgt: str = "", raw_text: str = ""):
        self.player      = player       # "AP" or "CP"
        self.card_id     = card_id      # "AP-14" etc.
        self.action_type = action_type  # "EVENT" | "MOVE" | "ATTACK" | "SR" | "PASS"
        self.src         = src          # source space ID
        self.tgt         = tgt          # target space ID
        self.raw_text    = raw_text


def parse_text_log(filepath: str) -> List[RawMove]:
    """
    Parse a plain-text game log.

    Expected line format (flexible):
        AP PLAYS AP-14 AS OPS MOVE FROM AMIENS TO PARIS
        CP PLAYS CP-07 AS EVENT
        AP PLAYS AP-06 AS OPS ATTACK FROM SEDAN TO LIEGE
        AP PASS
        CP PLAYS CP-02 AS OPS SR FROM BERLIN TO WARSAW

    Lines not matching are skipped with a warning.
    """
    moves: List[RawMove] = []
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip().upper()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            player = parts[0] if parts[0] in ("AP", "CP") else None
            if player is None:
                continue

            if len(parts) >= 2 and parts[1] == "PASS":
                moves.append(RawMove(player, "", "PASS", raw_text=line))
                continue

            # Expect: PLAYER PLAYS CARD_ID AS TYPE [FROM SRC [TO TGT]]
            if len(parts) < 5 or parts[1] != "PLAYS":
                print(f"  [warn] line {lineno}: unrecognised format: {line!r}")
                continue

            card_id     = parts[2]   # e.g. "AP-14"
            action_type = parts[4]   # "EVENT" | "MOVE" | "ATTACK" | "SR"

            src = tgt = ""
            try:
                if "FROM" in parts:
                    fi = parts.index("FROM")
                    src = parts[fi + 1]
                if "TO" in parts:
                    ti = parts.index("TO")
                    tgt = parts[ti + 1]
            except IndexError:
                pass

            moves.append(RawMove(player, card_id, action_type, src, tgt, line))

    return moves


def parse_json_log(filepath: str) -> List[RawMove]:
    """
    Parse a structured JSON game log (e.g. from a VASSAL export script).

    Expected format:
    [
      {"player": "AP", "card": "AP-14", "type": "OPS_MOVE",
       "from": "AMIENS", "to": "PARIS"},
      ...
    ]
    """
    with open(filepath) as f:
        data = json.load(f)

    moves = []
    for entry in data:
        moves.append(RawMove(
            player      = entry.get("player", "AP").upper(),
            card_id     = entry.get("card", ""),
            action_type = entry.get("type", "PASS").upper().replace("OPS_", ""),
            src         = entry.get("from", "").upper().replace(" ", "_"),
            tgt         = entry.get("to",   "").upper().replace(" ", "_"),
            raw_text     = json.dumps(entry),
        ))
    return moves


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Map RawMove to flat action index
# ─────────────────────────────────────────────────────────────────────

def raw_move_to_flat_action(move: RawMove, env: PogEnv,
                             card_str_to_idx: Dict[str, int]) -> Optional[int]:
    """
    Convert a RawMove to a flat action integer [0, N_ACTIONS).
    Returns None if the move cannot be mapped (skipped with a warning).
    """
    if move.action_type == "PASS" or not move.card_id:
        return 0  # PASS

    card_idx = card_str_to_idx.get(move.card_id)
    if card_idx is None:
        print(f"  [warn] unknown card_id '{move.card_id}' — skipping")
        return None

    if move.action_type == "EVENT":
        flat = ACT_EVENT_START + card_idx
        if flat >= ACT_OPS_START:
            print(f"  [warn] card_idx {card_idx} out of EVENT range — skipping")
            return None
        return flat

    op_map = {"MOVE": 0, "ATTACK": 1, "SR": 2}
    op_type = op_map.get(move.action_type)

    if op_type is not None:
        # PLAY_AS_OPS
        flat = ACT_OPS_START + card_idx * 3 + op_type
        if flat >= ACT_MOVE_START:
            print(f"  [warn] OPS flat {flat} out of range — skipping")
            return None
        return flat

    # Could be a MOVE_UNIT action (src/tgt space names given)
    if move.src and move.tgt:
        src_idx = env._space_id_to_idx.get(move.src)
        tgt_idx = env._space_id_to_idx.get(move.tgt)
        if src_idx is None:
            print(f"  [warn] unknown src space '{move.src}' — skipping")
            return None
        if tgt_idx is None:
            print(f"  [warn] unknown tgt space '{move.tgt}' — skipping")
            return None
        return ACT_MOVE_START + src_idx * N_SPACES + tgt_idx

    print(f"  [warn] unresolvable move: {move.raw_text!r}")
    return None


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Replay game and emit JSONL records
# ─────────────────────────────────────────────────────────────────────

def replay_game(moves:         List[RawMove],
                env:           PogEnv,
                card_str_to_idx: Dict[str, int],
                game_id:       str,
                outcome_ap:    int) -> List[Dict]:
    """
    Replay a parsed game move-by-move through the env.
    At each step, snapshot: obs, legal_mask, action_taken, outcome.

    outcome_ap: +1 if AP won, -1 if CP won, 0 if draw.
    """
    records = []
    obs = env.reset(seed=0)

    for step_idx, move in enumerate(moves):
        player = env._agent_name(env.active_player)

        # Build obs tensors for current state
        spatial   = env._build_spatial_obs()
        card_ctx  = env._build_card_context(env.active_player)
        legal_mask = env.action_mask(player)

        # Map move to flat action
        flat_action = raw_move_to_flat_action(move, env, card_str_to_idx)
        if flat_action is None:
            flat_action = 0  # fall back to PASS

        # Validate legality (warn but don't crash)
        if not legal_mask[flat_action]:
            print(f"  [warn] game {game_id} step {step_idx}: "
                  f"action {flat_action} is illegal — using PASS")
            flat_action = 0

        records.append({
            "game_id":      game_id,
            "step":         step_idx,
            "turn":         env.turn,
            "action_round": env.action_round,
            "player":       player,
            "obs_tensor":   spatial.tolist(),
            "card_context": card_ctx.tolist(),
            "legal_mask":   legal_mask.tolist(),
            "action_taken": flat_action,
            "outcome":      outcome_ap if env.active_player == FACTION_AP else -outcome_ap,
        })

        # Step the env
        try:
            obs, _, done, _, _ = env.step(flat_action)
            if any(done.values()):
                break
        except Exception as e:
            print(f"  [warn] env.step raised {e} at step {step_idx} — stopping game")
            break

    return records


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Main conversion driver
# ─────────────────────────────────────────────────────────────────────

def convert_directory(input_dir:  str,
                      output_path: str,
                      map_json:   str = "pog_map_graph.json",
                      cards_json: str = "pog_cards_db.json") -> int:
    """
    Convert all game files in input_dir to JSONL at output_path.
    Returns total number of (state, action) records written.

    Game files should be named: <game_id>_<outcome>.txt or .json
      outcome suffix: _APwin, _CPwin, _draw
      e.g.  game001_APwin.txt
    """
    env = PogEnv(map_json, cards_json)
    env._load_data()

    # Build card string-id → integer-index map
    card_str_to_idx = {c["str_id"]: i for i, c in enumerate(env._cards_db)}

    total_records = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_f:
        for fname in sorted(os.listdir(input_dir)):
            fpath = os.path.join(input_dir, fname)
            if not os.path.isfile(fpath):
                continue

            ext = fname.lower().rsplit(".", 1)[-1]
            if ext not in ("txt", "json"):
                continue

            # Infer outcome from filename
            stem    = fname.rsplit(".", 1)[0].lower()
            outcome = 1 if "apwin" in stem else (-1 if "cpwin" in stem else 0)
            game_id = stem

            print(f"Processing {fname}  (outcome={'AP' if outcome==1 else 'CP' if outcome==-1 else 'draw'})")

            try:
                if ext == "json":
                    moves = parse_json_log(fpath)
                else:
                    moves = parse_text_log(fpath)
            except Exception as e:
                print(f"  [error] failed to parse {fname}: {e}")
                continue

            records = replay_game(moves, env, card_str_to_idx, game_id, outcome)
            for rec in records:
                out_f.write(json.dumps(rec) + "\n")

            total_records += len(records)
            print(f"  → {len(records)} steps written")

    print(f"\nTotal: {total_records} records → {output_path}")
    return total_records


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PoG game logs to JSONL")
    parser.add_argument("--input",  required=True, help="Directory of raw game logs")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--map",    default="pog_map_graph.json")
    parser.add_argument("--cards",  default="pog_cards_db.json")
    args = parser.parse_args()

    n = convert_directory(args.input, args.output, args.map, args.cards)
    print(f"Done. {n} training steps.")
