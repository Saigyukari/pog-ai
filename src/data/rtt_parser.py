"""
rtt_parser.py — Parse Rally the Troops (RTT) server JSON game records
into training JSONL compatible with bc_pipeline.make_bc_batches().

═══════════════════════════════════════════════════════════════════════
QUICK START FOR LLM AGENTS — single-file extraction in 4 lines:

    from src.data.rtt_parser import extract_training_records

    records = extract_training_records("data/176409.json")
    # records: List[Dict], each element is one ready-to-train step

    # Inspect a record
    import numpy as np
    r = records[0]
    print(r["player"])                          # "AP" or "CP"
    print(r["action_taken"])                    # int in [0, 5341)
    print(np.array(r["obs_tensor"]).shape)      # (32, 72)
    print(np.array(r["card_context"]).shape)    # (7, 16)
    print(np.array(r["legal_mask"]).shape)      # (5341,)
    print(r["outcome"])                         # +1 AP win / -1 CP win / 0 draw

BATCH EXTRACTION (directory of games → single JSONL file):

    from src.data.rtt_parser import convert_rtt_directory

    n = convert_rtt_directory(
        input_dir      = "data/rtt_games/",
        output_path    = "data/expert_games.jsonl",
        space_map_json = "data/rtt_space_map.json",   # optional but recommended
    )
    print(f"{n} training records written")
═══════════════════════════════════════════════════════════════════════

RTT JSON structure (top-level keys):
  replay:  [[role, action, *args], ...]   full action log (~3000 entries)
  players: [{"role": "Allied Powers"/"Central Powers", "name": ...}, ...]
  state:   final game state — .turn, .vp, .location (piece_idx → RTT space_id)

RTT card ID → our str_id:
  IDs  1–65  →  "AP-01" .. "AP-65"
  IDs 66–130 →  "CP-01" .. "CP-65"

RTT space IDs (1–283 are board spaces; 284+ are off-board boxes):
  Mapped via data/rtt_space_map.json (67 of our 72 spaces covered).
  See RTT_DATA_PIPELINE.md §5 for the full table.

Output record schema (matches bc_pipeline.make_bc_batches exactly):
  game_id      str         game identifier (filename stem)
  step         int         sequential step index within the game
  turn         int         game turn number (1–20)
  action_round int         action round within the turn
  player       str         "AP" or "CP"
  obs_tensor   (32,72)     float32 board observation
  card_context (7,16)      float32 hand/card context
  legal_mask   (5341,)     bool    legal actions in this position
  action_taken int [0,5341) flat action index the human expert chose
  outcome      int +1/-1/0 result from THIS player's perspective

CLI:
  python -m src.data.rtt_parser \\
      --input     data/rtt_games/          # dir of *.json RTT exports
      --output    data/expert_games.jsonl
      --map       pog_map_graph.json
      --cards     pog_cards_db.json
      [--space_map data/rtt_space_map.json]
"""

from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.env.pog_env import PogEnv

# ── Action layout constants (must match pog_engine.py / bc_pipeline.py) ──────
N_SPACES  = 72
N_CARDS   = 110
N_ACTIONS = 5341
ACT_EVENT_START = 1     # [1:111]   EVENT
ACT_OPS_START   = 111   # [111:441] OPS
ACT_MOVE_START  = 441   # [441:5341] MOVE_UNIT

# ── RTT-specific constants ────────────────────────────────────────────────────
RTT_ROLE_AP = "Allied Powers"
RTT_ROLE_CP = "Central Powers"

# Special RTT location values
RTT_LOC_OFF_BOARD   = 0    # piece not yet placed or removed
RTT_LOC_ELIMINATED  = 360  # confirmed eliminated box
RTT_LOC_REINFORCING = 284  # reinforcement pool-ish values
RTT_LOC_SPECIAL_MIN = 284  # locations >= this are off-board zones


# ─────────────────────────────────────────────────────────────────────────────
# Card ID mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def rtt_card_to_strid(rtt_id: int) -> str:
    """
    Convert RTT integer card ID to our canonical str_id.
      1-65  → AP-01..AP-65
      66-130 → CP-01..CP-65
    """
    if 1 <= rtt_id <= 65:
        return f"AP-{rtt_id:02d}"
    elif 66 <= rtt_id <= 130:
        return f"CP-{rtt_id - 65:02d}"
    else:
        raise ValueError(f"RTT card ID {rtt_id} out of known range 1-130")


def build_card_lookup(cards_db: List[Dict]) -> Dict[str, int]:
    """
    Build str_id → card_idx map from cards_db list.
    Uses first occurrence for any duplicate IDs.
    Only includes indices < N_CARDS (110) — cards beyond that slot are ignored.
    """
    lookup: Dict[str, int] = {}
    for idx, card in enumerate(cards_db):
        if idx >= N_CARDS:
            break
        sid = card.get("id") or card.get("str_id", "")
        if sid and sid not in lookup:
            lookup[sid] = idx
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Replay grouping
# ─────────────────────────────────────────────────────────────────────────────

def group_replay_by_action(replay: List) -> List[List]:
    """
    Split replay entries into action groups.
    Each group is a list of entries ending with an 'end_action' entry,
    OR the trailing tail (entries after the last end_action).

    Skips: setup entries, undo chains (entry immediately before an undo is
    dropped), and entries with action='.setup'.
    """
    groups: List[List] = []
    current: List = []
    # Build list with undo handling: if next entry is 'undo', pop the last entry
    cleaned: List = []
    i = 0
    while i < len(replay):
        entry = replay[i]
        action = entry[1] if len(entry) > 1 else None
        if action == ".setup":
            i += 1
            continue
        if action == "undo":
            # Roll back: remove last cleaned entry if any
            if cleaned:
                cleaned.pop()
            i += 1
            continue
        cleaned.append(entry)
        i += 1

    for entry in cleaned:
        action = entry[1] if len(entry) > 1 else None
        current.append(entry)
        if action == "end_action":
            groups.append(current)
            current = []

    if current:  # trailing entries (e.g. .resign)
        groups.append(current)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Action group → flat action
# ─────────────────────────────────────────────────────────────────────────────

def _extract_primary_action(group: List,
                             card_lookup: Dict[str, int]) -> Tuple[int, str]:
    """
    From one action group, extract:
      (flat_action_int, role_string)

    Priority: play_event > play_ops > play_rps > play_sr > next (PASS).
    A 'next' entry before a card play is ignored — card plays take precedence.

    Returns (0, role) for PASS/unrecognised.
    """
    role = None

    # First pass: find role from any entry
    for entry in group:
        if isinstance(entry, list) and entry[0] in (RTT_ROLE_AP, RTT_ROLE_CP):
            role = entry[0]
            break
    if role is None:
        return 0, RTT_ROLE_AP  # fallback

    # Pre-compute group action set for context (used for op_type detection)
    group_action_set = {e[1] for e in group if isinstance(e, list) and len(e) > 1}
    has_next = "next" in group_action_set

    # Scan for card-play entries in priority order
    def _try_card(rtt_id: int, op_type_override: Optional[int] = None) -> Optional[int]:
        """Map rtt_id → flat action. op_type_override: 0=MOVE,1=ATTACK,2=SR or None=detect."""
        try:
            strid = rtt_card_to_strid(rtt_id)
        except ValueError:
            return None
        card_idx = card_lookup.get(strid)
        if card_idx is None:
            return None
        return card_idx

    for entry in group:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        action = entry[1]

        if action == "play_event" and len(entry) >= 3:
            card_idx = _try_card(entry[2])
            if card_idx is not None:
                flat = ACT_EVENT_START + card_idx
                if flat < ACT_OPS_START:
                    return flat, role
            return 0, role  # unmappable → PASS

        if action == "play_ops" and len(entry) >= 3:
            card_idx = _try_card(entry[2])
            if card_idx is None:
                return 0, role
            if "activate_attack" in group_action_set:
                op_type = 1
            elif "activate_move" in group_action_set:
                op_type = 0
            else:
                op_type = 0
            flat = ACT_OPS_START + card_idx * 3 + op_type
            if flat < ACT_MOVE_START:
                return flat, role
            return 0, role

        if action == "play_rps" and len(entry) >= 3:
            card_idx = _try_card(entry[2])
            if card_idx is not None:
                flat = ACT_OPS_START + card_idx * 3 + 0
                if flat < ACT_MOVE_START:
                    return flat, role
            return 0, role

        if action == "play_sr" and len(entry) >= 3:
            card_idx = _try_card(entry[2])
            if card_idx is not None:
                flat = ACT_OPS_START + card_idx * 3 + 2
                if flat < ACT_MOVE_START:
                    return flat, role
            return 0, role

    # No card play found — PASS (covers 'next'-only and reaction groups)
    return 0, role


def _extract_move_unit_records(group: List,
                                piece_locations: List[int],
                                space_map: Dict[int, int]) -> List[Tuple[int, int, int]]:
    """
    Extract individual MOVE_UNIT (piece P → space S) sub-actions from an OPS group.
    Returns list of (src_space_idx, tgt_space_idx, flat_action_int).

    piece_locations: list[piece_idx] → current RTT space_id
    space_map: RTT space_id (int) → our space index (0-based)
    """
    if not space_map:
        return []

    moves = []
    current_piece: Optional[int] = None

    for entry in group:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        action = entry[1]

        if action == "piece" and len(entry) >= 3:
            current_piece = entry[2]

        elif action == "space" and len(entry) >= 3 and current_piece is not None:
            rtt_src = piece_locations[current_piece] if current_piece < len(piece_locations) else None
            rtt_tgt = entry[2]

            src_idx = space_map.get(rtt_src) if rtt_src else None
            tgt_idx = space_map.get(rtt_tgt)

            if src_idx is not None and tgt_idx is not None:
                flat = ACT_MOVE_START + src_idx * N_SPACES + tgt_idx
                if flat < N_ACTIONS:
                    moves.append((src_idx, tgt_idx, flat))

            # Update tracked location
            if current_piece is not None and current_piece < len(piece_locations):
                piece_locations[current_piece] = rtt_tgt

            # A piece action typically covers one destination; reset after first move
            current_piece = None

        elif action == "stop":
            current_piece = None

    return moves


def _update_piece_locations(group: List, piece_locations: List[int]) -> None:
    """
    Update piece_locations in-place based on piece→space entries in the group.
    Called after processing to keep location state consistent.
    """
    current_piece: Optional[int] = None
    for entry in group:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        action = entry[1]
        if action == "piece" and len(entry) >= 3:
            current_piece = entry[2]
        elif action == "space" and len(entry) >= 3 and current_piece is not None:
            if current_piece < len(piece_locations):
                piece_locations[current_piece] = entry[2]
            current_piece = None
        elif action in ("stop", "done"):
            current_piece = None


# ─────────────────────────────────────────────────────────────────────────────
# Outcome extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_outcome(replay: List) -> int:
    """
    Scan replay for a .resign entry to determine winner.
    Returns +1 if AP won, -1 if CP won, 0 for draw/unknown.

    RTT resign format: [loser_role, '.resign', winner_role]
    """
    for entry in reversed(replay):
        if isinstance(entry, list) and len(entry) >= 3 and entry[1] == ".resign":
            winner = entry[2]
            if winner == RTT_ROLE_AP:
                return 1
            elif winner == RTT_ROLE_CP:
                return -1
    return 0  # no resign found → draw or incomplete


# ─────────────────────────────────────────────────────────────────────────────
# Main game conversion
# ─────────────────────────────────────────────────────────────────────────────

def parse_rtt_game(game_path: str,
                   env: PogEnv,
                   card_lookup: Dict[str, int],
                   space_map: Dict[int, int],
                   game_id: str) -> List[Dict]:
    """
    Parse one RTT JSON game file → list of JSONL-ready record dicts.

    Each record has the fields expected by bc_pipeline.make_bc_batches():
      obs_tensor   (32, 72) float32  → env._build_spatial_obs()
      card_context (7, 16)  float32  → env._build_card_context(player_idx)
      legal_mask   (5341,)  bool     → env.action_mask(agent_name)
      action_taken int               → flat action index
      outcome      int               → +1 AP wins, -1 CP wins, 0 draw
    """
    with open(game_path) as f:
        data = json.load(f)

    replay  = data["replay"]
    outcome = extract_outcome(replay)

    # Initialise piece location tracker from final state
    # (RTT state.location reflects end state; for training we step from start)
    piece_locations: List[int] = list(data["state"]["location"])

    # Group replay into action sequences
    groups = group_replay_by_action(replay)

    # Reset env to fresh game
    env.reset(seed=0)

    records: List[Dict] = []
    step_idx = 0

    for group in groups:
        flat_action, role = _extract_primary_action(group, card_lookup)

        # Skip administrative / sub-action-only groups
        group_action_types = {e[1] for e in group if isinstance(e, list) and len(e) > 1}
        strategic_actions = {"play_event", "play_ops", "play_rps", "play_sr", "next"}
        if not (group_action_types & strategic_actions):
            _update_piece_locations(group, piece_locations)
            continue

        # Determine which player is active in our env
        player_name = env._agent_name(env.active_player)
        env_role    = RTT_ROLE_AP if env.active_player == 0 else RTT_ROLE_CP

        # Snapshot observation BEFORE stepping
        spatial   = env._build_spatial_obs()
        card_ctx  = env._build_card_context(env.active_player)
        legal     = env.action_mask(player_name)

        # Sanity-range check only (don't override based on env's approx legal mask;
        # the RTT action was legal in the actual game even if our env's state diverges)
        if flat_action >= N_ACTIONS or flat_action < 0:
            flat_action = 0

        # Outcome is from the active player's perspective
        faction_outcome = outcome if env.active_player == 0 else -outcome

        records.append({
            "game_id":      game_id,
            "step":         step_idx,
            "turn":         int(env.turn),
            "action_round": int(env.action_round),
            "player":       player_name,
            "obs_tensor":   spatial.tolist(),
            "card_context": card_ctx.tolist(),
            "legal_mask":   legal.tolist(),
            "action_taken": int(flat_action),
            "outcome":      int(faction_outcome),
        })

        # Also emit individual MOVE_UNIT records for OPS_MOVE groups
        # Only process genuine movement orders (activate_move present), not events/attacks
        if space_map and "activate_move" in group_action_types:
            move_records = _extract_move_unit_records(group, piece_locations, space_map)
            for src_idx, tgt_idx, move_flat in move_records:
                if src_idx == tgt_idx:
                    continue  # skip self-moves (stacking/placement artifacts)
                if not (0 <= move_flat < N_ACTIONS):
                    continue
                step_idx += 1
                records.append({
                    "game_id":      game_id,
                    "step":         step_idx,
                    "turn":         int(env.turn),
                    "action_round": int(env.action_round),
                    "player":       player_name,
                    "obs_tensor":   spatial.tolist(),
                    "card_context": card_ctx.tolist(),
                    "legal_mask":   legal.tolist(),
                    "action_taken": int(move_flat),
                    "outcome":      int(faction_outcome),
                })

        # Step env forward
        try:
            _, _, done, _, _ = env.step(flat_action)
            if any(done.values()):
                break
        except Exception as e:
            # Env step failed (incomplete rules); continue to next group
            pass

        _update_piece_locations(group, piece_locations)
        step_idx += 1

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Single-file convenience entry point  (primary API for LLM agents)
# ─────────────────────────────────────────────────────────────────────────────

def extract_training_records(
    game_path:      str,
    map_json:       str = "pog_map_graph.json",
    cards_json:     str = "pog_cards_db.json",
    space_map_json: Optional[str] = "data/rtt_space_map.json",
) -> List[Dict]:
    """
    Extract all training records from a single RTT game JSON file.

    This is the simplest entry point — give it one game file, get back a
    list of ready-to-use training dicts that feed directly into
    bc_pipeline.make_bc_batches().

    Parameters
    ----------
    game_path : str
        Path to a single RTT game export, e.g. "data/176409.json".
    map_json : str
        Path to pog_map_graph.json (default: "pog_map_graph.json").
    cards_json : str
        Path to pog_cards_db.json (default: "pog_cards_db.json").
    space_map_json : str | None
        Path to data/rtt_space_map.json for MOVE_UNIT extraction.
        Pass None to skip piece-movement records (card plays only).

    Returns
    -------
    List[Dict]
        Each dict has these keys, ready for bc_pipeline:
          "game_id"      : str
          "step"         : int
          "turn"         : int
          "action_round" : int
          "player"       : "AP" or "CP"
          "obs_tensor"   : list[list[float]]  shape (32, 72)
          "card_context" : list[list[float]]  shape (7, 16)
          "legal_mask"   : list[bool]         length 5341
          "action_taken" : int                in [0, 5341)
          "outcome"      : int                +1 / -1 / 0

    Example
    -------
    >>> from src.data.rtt_parser import extract_training_records
    >>> records = extract_training_records("data/176409.json")
    >>> len(records)
    195
    >>> records[0]["player"]
    'CP'
    >>> records[0]["action_taken"]   # flat index of the card played
    72
    >>> records[0]["outcome"]        # CP perspective: CP lost → -1
    -1
    """
    env = PogEnv(map_json, cards_json)
    env._load_data()

    card_lookup = build_card_lookup(env._cards_db)

    space_map: Dict[int, int] = {}
    if space_map_json and os.path.isfile(space_map_json):
        with open(space_map_json) as f:
            raw_sm = json.load(f)
        sid_to_idx = env._space_id_to_idx
        for rtt_str, our_strid in raw_sm.items():
            idx = sid_to_idx.get(our_strid)
            if idx is not None:
                space_map[int(rtt_str)] = idx

    game_id = os.path.splitext(os.path.basename(game_path))[0]
    return parse_rtt_game(game_path, env, card_lookup, space_map, game_id)


# ─────────────────────────────────────────────────────────────────────────────
# Directory conversion driver
# ─────────────────────────────────────────────────────────────────────────────

def convert_rtt_directory(input_dir:  str,
                          output_path: str,
                          map_json:   str = "pog_map_graph.json",
                          cards_json: str = "pog_cards_db.json",
                          space_map_json: Optional[str] = None) -> int:
    """
    Convert all *.json RTT game files in input_dir to JSONL at output_path.

    space_map_json (optional): path to a JSON file mapping RTT integer space IDs
    to our canonical space str_ids, e.g.:
      {"1": "PARIS", "2": "AMIENS", ...}
    Without this file, MOVE_UNIT sub-actions are skipped; card-play records
    are still emitted (the most strategically valuable training signal).

    Returns total number of records written.
    """
    env = PogEnv(map_json, cards_json)
    env._load_data()

    # Build card lookup
    cards_db = env._cards_db
    card_lookup = build_card_lookup(cards_db)
    print(f"Loaded {len(card_lookup)} card mappings (N_CARDS={N_CARDS})")

    # Build space map (RTT int → our space idx)
    space_map: Dict[int, int] = {}
    if space_map_json and os.path.isfile(space_map_json):
        with open(space_map_json) as f:
            raw_sm = json.load(f)
        # raw_sm: {"1": "PARIS", ...}  RTT_id_str → our space str_id
        sid_to_idx = env._space_id_to_idx  # str_id → int index (populated by _load_data)
        for rtt_str, our_strid in raw_sm.items():
            idx = sid_to_idx.get(our_strid)
            if idx is not None:
                space_map[int(rtt_str)] = idx
        print(f"Loaded {len(space_map)} space mappings from {space_map_json}")
    else:
        print("No space_map provided — MOVE_UNIT sub-actions will be skipped")

    total = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_f:
        for fname in sorted(os.listdir(input_dir)):
            if not fname.lower().endswith(".json"):
                continue
            fpath = os.path.join(input_dir, fname)
            game_id = fname.rsplit(".", 1)[0]

            print(f"Processing {fname} ...")
            try:
                records = parse_rtt_game(fpath, env, card_lookup, space_map, game_id)
            except Exception as e:
                print(f"  [error] {fname}: {e}")
                continue

            for rec in records:
                out_f.write(json.dumps(rec) + "\n")
            total += len(records)
            print(f"  → {len(records)} records written")

    print(f"\nTotal: {total} records → {output_path}")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RTT game JSONs to training JSONL")
    parser.add_argument("--input",     required=True,  help="Directory of RTT *.json game files")
    parser.add_argument("--output",    required=True,  help="Output JSONL file path")
    parser.add_argument("--map",       default="pog_map_graph.json")
    parser.add_argument("--cards",     default="pog_cards_db.json")
    parser.add_argument("--space_map", default=None,
                        help="Optional JSON: {rtt_space_id: our_space_str_id}")
    args = parser.parse_args()

    n = convert_rtt_directory(
        args.input, args.output, args.map, args.cards, args.space_map)
    print(f"Done. {n} training records.")
