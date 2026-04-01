"""
pog_engine.py — Core data structures for Paths of Glory RL engine.

All dataclass fields use numpy-friendly scalar types (int, float, bool).
No nested dicts in GameState at runtime — all collections are fixed-length lists/arrays.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import json
import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TERRAIN_CLEAR    = 0
TERRAIN_FORT     = 1
TERRAIN_MOUNTAIN = 2
TERRAIN_SEA      = 3
TERRAIN_DESERT   = 4

NATION_FR = 0; NATION_GE = 1; NATION_BE = 2; NATION_RU = 3
NATION_AH = 4; NATION_TU = 5; NATION_IT = 6; NATION_BU = 7
NATION_SE = 8; NATION_RO = 9; NATION_NE = 10; NATION_GR = 11
NATION_BR = 12

NATION_STR_MAP = {
    "FR": 0, "GE": 1, "BE": 2, "RU": 3, "AH": 4, "TU": 5, "IT": 6,
    "BU": 7, "SE": 8, "RO": 9, "NE": 10, "GR": 11, "BR": 12,
}

FACTION_AP      = 0
FACTION_CP      = 1
FACTION_NEUTRAL = 2

UNIT_INF   = 0; UNIT_CAV  = 1; UNIT_ART  = 2; UNIT_CORP = 3
UNIT_ARMY  = 4; UNIT_FLEET = 5; UNIT_SUB = 6; UNIT_AIR  = 7
N_UNIT_TYPES = 8

PHASE_LIMITED = 0
PHASE_TOTAL   = 1
PHASE_EITHER  = 2

TERRAIN_STR_MAP = {"Clear": 0, "Fort": 1, "Mountain": 2, "Sea": 3, "Desert": 4}
PHASE_STR_MAP   = {"Limited War": 0, "Total War": 1, "Either": 2, "Unknown": 2}

N_SPACES  = 72
N_PLANES  = 32
N_ACTIONS = 5341   # 1 PASS + 110 EVENT + 330 OPS + 4900 MOVE

AP_NATIONS = {NATION_FR, NATION_BE, NATION_RU, NATION_IT, NATION_GR, NATION_BR, NATION_RO, NATION_SE}
CP_NATIONS = {NATION_GE, NATION_AH, NATION_TU, NATION_BU}


# ─────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────

@dataclass
class MapSpace:
    idx: int                       # integer index 0..N_SPACES-1
    str_id: str                    # "PARIS" etc.
    name: str
    terrain_type: int              # TERRAIN_* constant
    vp_value: int
    nation_id: int                 # NATION_* constant
    is_fortress: bool
    connection_idxs: List[int]     # indices into spaces array (NOT string IDs)
    # Dynamic fields reset each game
    trench_level: int = 0          # 0–3
    oos_status: bool = False
    controlling_faction: int = FACTION_NEUTRAL
    fort_destroyed: bool = False


@dataclass
class Unit:
    unit_id: int
    str_id: str                    # e.g. "GE_INF_01"
    nation_id: int
    faction: int                   # FACTION_AP or FACTION_CP
    unit_type: int                 # UNIT_* constant
    strength: int                  # current steps 1–3
    max_strength: int
    location: int                  # space index
    is_eliminated: bool = False
    has_moved: bool = False


@dataclass
class Card:
    card_idx: int                  # 0–109 unique integer index
    str_id: str                    # "AP-01"
    faction: int                   # FACTION_AP or FACTION_CP
    ops_value: int                 # 2–5
    sr_value: int                  # 2–5
    is_combat_card: bool
    phase_gate: int                # PHASE_* constant
    event_text: str
    remove_after_event: bool = True


@dataclass
class GameState:
    # ── Turn tracking ──────────────────────────────────────────────────
    turn: int = 1                  # 1–20
    action_round: int = 1          # 1–7
    active_player: int = FACTION_AP
    war_status_ap: int = PHASE_LIMITED
    war_status_cp: int = PHASE_LIMITED
    vp_track: int = 0              # negative=AP winning, positive=CP winning
    mobilization_turn: int = 0

    # ── Board ──────────────────────────────────────────────────────────
    spaces: List[MapSpace] = field(default_factory=list)
    units: List[Unit] = field(default_factory=list)

    # ── Hands & decks (card indices) ───────────────────────────────────
    ap_hand: List[int] = field(default_factory=list)
    cp_hand: List[int] = field(default_factory=list)
    ap_deck: List[int] = field(default_factory=list)
    cp_deck: List[int] = field(default_factory=list)
    ap_discard: List[int] = field(default_factory=list)
    cp_discard: List[int] = field(default_factory=list)

    # ── Card registry ──────────────────────────────────────────────────
    cards: List[Card] = field(default_factory=list)

    # ── Space index lookup ─────────────────────────────────────────────
    space_id_to_idx: dict = field(default_factory=dict)

    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, map_path: str, cards_path: str) -> "GameState":
        """Load initial game state from JSON files."""
        with open(map_path) as f:
            map_data = json.load(f)
        with open(cards_path) as f:
            cards_raw = json.load(f)

        # Handle both old format (list of dicts) and new format
        if isinstance(map_data, list):
            spaces_raw = []
            for item in map_data:
                for name, info in item.items():
                    spaces_raw.append({"id": name.upper().replace(" ", "_"), **info})
        else:
            spaces_raw = map_data.get("spaces", [])

        id_to_idx = {s["id"]: i for i, s in enumerate(spaces_raw)}

        spaces = []
        for i, s in enumerate(spaces_raw):
            nation_id = NATION_STR_MAP.get(s.get("nation", "NE"), NATION_NE)
            # Set initial control based on nation
            if nation_id in AP_NATIONS:
                ctrl = FACTION_AP
            elif nation_id in CP_NATIONS:
                ctrl = FACTION_CP
            else:
                ctrl = FACTION_NEUTRAL

            spaces.append(MapSpace(
                idx=i,
                str_id=s["id"],
                name=s.get("name", s["id"]),
                terrain_type=TERRAIN_STR_MAP.get(s.get("terrain", "Clear"), 0),
                vp_value=s.get("vp_value", 0),
                nation_id=nation_id,
                is_fortress=s.get("is_fortress", False),
                connection_idxs=[id_to_idx[c] for c in s.get("connections", []) if c in id_to_idx],
                trench_level=s.get("trench_level", 0),
                oos_status=s.get("oos_status", False),
                controlling_faction=ctrl,
            ))

        # Build card objects — deduplicate by str_id
        seen_idx: dict = {}
        cards: List[Card] = []
        ap_deck_indices: List[int] = []
        cp_deck_indices: List[int] = []
        card_counter = 0

        for entry in cards_raw:
            str_id = entry["id"]
            faction_str = entry.get("faction", "AP")
            faction = FACTION_AP if faction_str == "AP" else FACTION_CP
            phase_gate = PHASE_STR_MAP.get(entry.get("phase", "Either"), PHASE_EITHER)

            if str_id not in seen_idx:
                seen_idx[str_id] = card_counter
                cards.append(Card(
                    card_idx=card_counter,
                    str_id=str_id,
                    faction=faction,
                    ops_value=entry.get("ops", 2),
                    sr_value=entry.get("sr", 2),
                    is_combat_card=entry.get("is_combat_card", False),
                    phase_gate=phase_gate,
                    event_text=entry.get("event_text", ""),
                    remove_after_event=entry.get("remove_after_event", True),
                ))
                card_counter += 1

            # Add to deck (including physical duplicates)
            idx = seen_idx[str_id]
            if faction == FACTION_AP:
                ap_deck_indices.append(idx)
            else:
                cp_deck_indices.append(idx)

        return cls(
            spaces=spaces,
            cards=cards,
            ap_deck=ap_deck_indices,
            cp_deck=cp_deck_indices,
            space_id_to_idx=id_to_idx,
        )


# ─────────────────────────────────────────────
# Observation builders
# ─────────────────────────────────────────────

def build_observation_tensor(state: GameState) -> np.ndarray:
    """
    Build (N_PLANES, N_SPACES) float32 observation tensor.

    Plane layout (32 planes × 72 spaces):
      0  terrain_type         /4.0
      1  vp_value             /5.0
      2  controlling_faction  0=AP, 0.5=neutral, 1.0=CP
      3  trench_ordinal       trench_level/3.0
      4  trench_l3_gate       (trench_level>=3).astype(float)
      5  fort_destroyed       bool→float
      6  AP OOS flag          bool→float
      7  CP OOS flag          bool→float
      8-15   AP unit stacks by type (8 types, count/3 clamped)
      16-23  CP unit stacks by type
      24 AP hand size          /10.0
      25 CP hand size          /10.0
      26 AP mean ops           /5.0
      27 CP mean ops           /5.0
      28 VP track              (vp+20)/40.0
      29 AP war status         0=Limited, 1=Total
      30 CP war status
      31 active player         0=AP, 1=CP
    """
    n = len(state.spaces)
    obs = np.zeros((N_PLANES, n), dtype=np.float32)

    for i, sp in enumerate(state.spaces):
        obs[0, i] = sp.terrain_type / 4.0
        obs[1, i] = sp.vp_value / 5.0
        obs[2, i] = 0.0 if sp.controlling_faction == FACTION_AP else (
                    0.5 if sp.controlling_faction == FACTION_NEUTRAL else 1.0)
        obs[3, i] = sp.trench_level / 3.0
        obs[4, i] = float(sp.trench_level >= 3)
        obs[5, i] = float(sp.fort_destroyed)
        obs[6, i] = float(sp.oos_status)

    # Unit stacks
    unit_ap = np.zeros((N_UNIT_TYPES, n), dtype=np.float32)
    unit_cp = np.zeros((N_UNIT_TYPES, n), dtype=np.float32)
    for u in state.units:
        if u.is_eliminated or u.location >= n:
            continue
        if u.faction == FACTION_AP:
            unit_ap[u.unit_type, u.location] += 1
        else:
            unit_cp[u.unit_type, u.location] += 1
    obs[8:16]  = np.clip(unit_ap / 3.0, 0, 1)
    obs[16:24] = np.clip(unit_cp / 3.0, 0, 1)

    # Hand stats (broadcast as scalar plane)
    ap_ops = [state.cards[c].ops_value for c in state.ap_hand if c < len(state.cards)]
    cp_ops = [state.cards[c].ops_value for c in state.cp_hand if c < len(state.cards)]

    obs[24, :] = len(state.ap_hand) / 10.0
    obs[25, :] = len(state.cp_hand) / 10.0
    obs[26, :] = (float(np.mean(ap_ops)) / 5.0) if ap_ops else 0.0
    obs[27, :] = (float(np.mean(cp_ops)) / 5.0) if cp_ops else 0.0
    obs[28, :] = (state.vp_track + 20) / 40.0
    obs[29, :] = float(state.war_status_ap == PHASE_TOTAL)
    obs[30, :] = float(state.war_status_cp == PHASE_TOTAL)
    obs[31, :] = float(state.active_player == FACTION_CP)

    # Pad or trim to N_SPACES
    if n < N_SPACES:
        pad = np.zeros((N_PLANES, N_SPACES - n), dtype=np.float32)
        obs = np.concatenate([obs, pad], axis=1)
    elif n > N_SPACES:
        obs = obs[:, :N_SPACES]

    return obs


def build_card_context(state: GameState, faction: int,
                       n_slots: int = 7, embed_dim: int = 16) -> np.ndarray:
    """
    Build (n_slots, embed_dim) float32 card context for one player's hand.
    Encoding per slot: [ops/5, sr/5, is_combat, phase_gate/2, is_cp, 0…]
    """
    hand = state.ap_hand if faction == FACTION_AP else state.cp_hand
    ctx = np.zeros((n_slots, embed_dim), dtype=np.float32)
    for slot, card_idx in enumerate(hand[:n_slots]):
        if card_idx < len(state.cards):
            c = state.cards[card_idx]
            ctx[slot, 0] = c.ops_value / 5.0
            ctx[slot, 1] = c.sr_value / 5.0
            ctx[slot, 2] = float(c.is_combat_card)
            ctx[slot, 3] = c.phase_gate / 2.0
            ctx[slot, 4] = float(c.faction == FACTION_CP)
    return ctx


# ─────────────────────────────────────────────
# Action mask
# ─────────────────────────────────────────────

def compute_action_mask(state: GameState) -> np.ndarray:
    """
    Compute (N_ACTIONS=5341,) bool array of legal actions.

    Layout:
      [0]        PASS
      [1:111]    PLAY_AS_EVENT  (card_idx 0–109)
      [111:441]  PLAY_AS_OPS    (card_idx * 3 + op_type); op_type: 0=MOVE,1=ATTACK,2=SR
      [441:5341] MOVE_UNIT      441 + src*N_SPACES + tgt
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[0] = True  # PASS always legal

    faction = state.active_player
    war_st = state.war_status_ap if faction == FACTION_AP else state.war_status_cp
    hand = state.ap_hand if faction == FACTION_AP else state.cp_hand

    friendly_locs = {u.location for u in state.units if u.faction == faction and not u.is_eliminated}
    enemy_locs    = {u.location for u in state.units if u.faction != faction and not u.is_eliminated}
    space_adj = {sp.idx: set(sp.connection_idxs) for sp in state.spaces}

    for card_idx in hand:
        if card_idx >= len(state.cards):
            continue
        card = state.cards[card_idx]

        # Phase gate
        if card.phase_gate == PHASE_LIMITED and war_st == PHASE_TOTAL:
            continue
        if card.phase_gate == PHASE_TOTAL and war_st == PHASE_LIMITED:
            continue

        # EVENT
        ev = 1 + card_idx
        if ev < 111:
            mask[ev] = True

        # OPS MOVE
        s_move = 111 + card_idx * 3
        if s_move < 441 and friendly_locs:
            mask[s_move] = True

        # OPS ATTACK
        s_atk = 111 + card_idx * 3 + 1
        if s_atk < 441:
            for src in friendly_locs:
                if space_adj.get(src, set()) & enemy_locs:
                    mask[s_atk] = True
                    break

        # OPS SR
        s_sr = 111 + card_idx * 3 + 2
        if s_sr < 441 and friendly_locs:
            mask[s_sr] = True

    # MOVE_UNIT
    has_assault = any(state.cards[c].is_combat_card for c in hand if c < len(state.cards))
    for src in friendly_locs:
        for tgt in space_adj.get(src, set()):
            if tgt in enemy_locs and src < len(state.spaces):
                tgt_space = state.spaces[tgt] if tgt < len(state.spaces) else None
                if tgt_space and tgt_space.trench_level >= 3 and not has_assault:
                    continue
            flat = 441 + src * N_SPACES + tgt
            if flat < N_ACTIONS:
                mask[flat] = True

    return mask
