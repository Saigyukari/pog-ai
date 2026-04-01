"""
starting_positions.py — Initial unit metadata and 1914 historical deployment.

This module parses the definitive piece list from `data/data.js` and exposes
fixed-size arrays that can be reused by both the JAX env and the Python env.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.data.pog_engine import FACTION_AP, FACTION_CP, N_SPACES, UNIT_ARMY, UNIT_CORP, UNIT_INF

ROOT = Path(__file__).resolve().parents[2]
DATA_JS_PATH = ROOT / "data" / "data.js"
SPACE_MAP_PATH = ROOT / "data" / "rtt_space_map.json"
MAP_GRAPH_PATH = ROOT / "pog_map_graph.json"

MAX_UNITS = 194
OFFBOARD = 255


def _extract_pieces_block(text: str) -> str:
    start = text.find("pieces = [")
    if start < 0:
        raise ValueError("Could not find `pieces = [` in data.js")
    start = text.find("[", start)
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]
    raise ValueError("Could not find matching closing bracket for `pieces` block in data.js")


def _load_piece_entries() -> list[dict]:
    raw = DATA_JS_PATH.read_text()
    block = _extract_pieces_block(raw)
    block = re.sub(r",(\s*[}\]])", r"\1", block)
    pieces = json.loads(block)
    if len(pieces) != MAX_UNITS:
        raise ValueError(f"Expected {MAX_UNITS} piece entries, found {len(pieces)}")
    return pieces


def _load_space_index() -> tuple[dict[str, str], dict[str, int]]:
    rtt_map = json.loads(SPACE_MAP_PATH.read_text())
    map_graph = json.loads(MAP_GRAPH_PATH.read_text())
    spaces = map_graph["spaces"] if isinstance(map_graph, dict) else map_graph
    our_index = {space["id"]: i for i, space in enumerate(spaces[:N_SPACES])}
    return {str(k): v for k, v in rtt_map.items()}, our_index


def _space_from_rtt(rtt_space: int, rtt_to_our: dict[str, str], our_index: dict[str, int]) -> int:
    if rtt_space == OFFBOARD:
        return OFFBOARD
    our_id = rtt_to_our.get(str(rtt_space))
    if our_id is None:
        return OFFBOARD
    return our_index.get(our_id, OFFBOARD)


def _build_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pieces = _load_piece_entries()
    rtt_to_our, our_index = _load_space_index()

    faction = np.full((MAX_UNITS,), -1, dtype=np.int8)
    unit_type = np.full((MAX_UNITS,), UNIT_INF, dtype=np.int8)
    strength = np.zeros((MAX_UNITS,), dtype=np.int8)
    loc = np.full((MAX_UNITS,), OFFBOARD, dtype=np.uint8)

    type_map = {"army": UNIT_ARMY, "corps": UNIT_CORP}

    for idx, piece in enumerate(pieces):
        if not piece:
            continue
        faction[idx] = FACTION_AP if piece.get("faction") == "ap" else FACTION_CP
        unit_type[idx] = type_map.get(piece.get("type"), UNIT_INF)
        strength[idx] = int(piece.get("cf", 0))

    # Historical 1914 scenario placements for army counters.
    rtt_locations = {
        1: 42, 2: 43, 3: 44, 4: 45, 5: 46, 6: 47, 7: 38, 8: 39, 9: 40, 10: 41,
        11: 52, 12: 53, 13: 54, 14: OFFBOARD, 15: OFFBOARD,
        16: 80, 17: 81, 18: 82, 19: 83, 20: 84, 21: 85, 22: 86, 23: 87, 24: OFFBOARD,
        25: OFFBOARD, 26: OFFBOARD, 27: OFFBOARD, 28: OFFBOARD, 29: OFFBOARD,
        30: 140, 31: 141,
        32: 20, 33: 21, 34: 30, 35: 34, 36: 35, 37: 36, 38: 28, 39: 15,
        40: OFFBOARD, 41: OFFBOARD,
        42: OFFBOARD, 43: OFFBOARD, 44: OFFBOARD, 45: OFFBOARD, 46: OFFBOARD,
        47: 53, 48: OFFBOARD, 49: OFFBOARD, 50: OFFBOARD, 51: OFFBOARD,
        52: 90, 53: 91, 54: 92, 55: 93, 56: 94, 57: 95, 58: 96, 59: 97,
        60: 98, 61: 99, 62: 100, 63: 101, 64: OFFBOARD,
        65: 56, 66: 110, 67: 111,
    }
    for idx, rtt_space in rtt_locations.items():
        loc[idx] = _space_from_rtt(rtt_space, rtt_to_our, our_index)

    # Corps and reinforcement pieces remain off-board at start by design.
    return faction, unit_type, strength, loc


_FACTION, _TYPE, _STRENGTH, _LOC = _build_arrays()

UNIT_FACTION_INIT = jnp.asarray(_FACTION, dtype=jnp.int8)
UNIT_TYPE_INIT = jnp.asarray(_TYPE, dtype=jnp.int8)
UNIT_STRENGTH_INIT = jnp.asarray(_STRENGTH, dtype=jnp.int8)
INITIAL_UNIT_LOC = jnp.asarray(_LOC, dtype=jnp.uint8)
