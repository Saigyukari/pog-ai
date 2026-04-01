"""
jax_env.py — Pure JAX Paths of Glory environment core.

This mirrors the current simplified Python env in src/env/pog_env.py closely
enough for Phase 2 integration work:
  - no Python objects in runtime state
  - fixed-shape arrays only
  - jax.jit/jax.vmap compatible reset/step/obs/mask helpers

The current project env baseline is intentionally minimal: reset deals opening
hands, but does not yet populate starting units or resolve card-specific events.
This module preserves that behavior so it stays aligned with the existing
library rather than inventing unverified game rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from src.data.pog_engine import (
    AP_NATIONS,
    CP_NATIONS,
    FACTION_AP,
    FACTION_CP,
    FACTION_NEUTRAL,
    N_ACTIONS,
    N_PLANES,
    N_SPACES,
    N_UNIT_TYPES,
    PHASE_LIMITED,
    PHASE_TOTAL,
    PHASE_STR_MAP,
    TERRAIN_STR_MAP,
    UNIT_ARMY,
    UNIT_ART,
    UNIT_CAV,
    UNIT_CORP,
    UNIT_INF,
)
from src.data.starting_positions import INITIAL_UNIT_LOC, UNIT_FACTION_INIT, UNIT_STRENGTH_INIT, UNIT_TYPE_INIT

ACT_PASS = 0
ACT_EVENT_START = 1
ACT_OPS_START = 111
ACT_MOVE_START = 441

MAX_UNITS = 194
HAND_SIZE = 7
EMPTY_CARD = 255
OFFBOARD = 255
FACTION_CARD_COUNT = 65

_ROOT = Path(__file__).resolve().parents[2]
_MAP_PATH = _ROOT / "pog_map_graph.json"
_CARDS_PATH = _ROOT / "pog_cards_db.json"


def _load_static_tables() -> dict[str, np.ndarray]:
    with _MAP_PATH.open() as handle:
        map_data = json.load(handle)
    with _CARDS_PATH.open() as handle:
        cards_raw = json.load(handle)

    spaces_raw = map_data if isinstance(map_data, list) else map_data["spaces"]
    if isinstance(spaces_raw, list) and spaces_raw and isinstance(spaces_raw[0], dict) and "id" in spaces_raw[0]:
        pass
    else:
        normalized = []
        for item in spaces_raw:
            for name, info in item.items():
                normalized.append({"id": name.upper().replace(" ", "_"), **info})
        spaces_raw = normalized

    space_id_to_idx = {space["id"]: i for i, space in enumerate(spaces_raw)}
    adj = np.zeros((N_SPACES, N_SPACES), dtype=np.bool_)
    terrain = np.zeros((N_SPACES,), dtype=np.int8)
    vp = np.zeros((N_SPACES,), dtype=np.int8)
    nation = np.zeros((N_SPACES,), dtype=np.int8)
    fortress = np.zeros((N_SPACES,), dtype=np.bool_)
    control = np.full((N_SPACES,), FACTION_NEUTRAL, dtype=np.int8)

    nation_str_map = {
        "FR": 0, "GE": 1, "BE": 2, "RU": 3, "AH": 4, "TU": 5, "IT": 6,
        "BU": 7, "SE": 8, "RO": 9, "NE": 10, "GR": 11, "BR": 12,
    }

    for i, space in enumerate(spaces_raw[:N_SPACES]):
        nat = nation_str_map.get(space.get("nation", "NE"), 10)
        terrain[i] = TERRAIN_STR_MAP.get(space.get("terrain", "Clear"), 0)
        vp[i] = space.get("vp_value", 0)
        nation[i] = nat
        fortress[i] = bool(space.get("is_fortress", False))
        if nat in AP_NATIONS:
            control[i] = FACTION_AP
        elif nat in CP_NATIONS:
            control[i] = FACTION_CP
        for conn in space.get("connections", []):
            j = space_id_to_idx.get(conn.upper().replace(" ", "_"))
            if j is not None and j < N_SPACES:
                adj[i, j] = True
                adj[j, i] = True

    ap_source_mask = np.array(
        [
            (space.get("nation", "") in {"FR", "BE"}) and control[i] == FACTION_AP
            for i, space in enumerate(spaces_raw[:N_SPACES])
        ],
        dtype=np.bool_,
    )
    cp_source_mask = np.array(
        [
            (space.get("nation", "") in {"GE", "AH"}) and control[i] == FACTION_CP
            for i, space in enumerate(spaces_raw[:N_SPACES])
        ],
        dtype=np.bool_,
    )

    seen_cards: dict[str, int] = {}
    card_faction = []
    card_ops = []
    card_sr = []
    card_is_combat = []
    card_phase_gate = []
    ap_physical_deck = []
    cp_physical_deck = []
    ap_unique_local = np.full((130,), -1, dtype=np.int16)
    cp_unique_local = np.full((130,), -1, dtype=np.int16)
    ap_unique_count = 0
    cp_unique_count = 0

    for entry in cards_raw:
        sid = entry["id"]
        faction = FACTION_AP if entry.get("faction", "AP") == "AP" else FACTION_CP
        if sid not in seen_cards:
            global_idx = len(seen_cards)
            seen_cards[sid] = global_idx
            card_faction.append(faction)
            card_ops.append(entry.get("ops", 2))
            card_sr.append(entry.get("sr", 2))
            card_is_combat.append(bool(entry.get("is_combat_card", False)))
            card_phase_gate.append(PHASE_STR_MAP.get(entry.get("phase", "Either"), 2))
            if faction == FACTION_AP:
                ap_unique_local[global_idx] = ap_unique_count
                ap_unique_count += 1
            else:
                cp_unique_local[global_idx] = cp_unique_count
                cp_unique_count += 1

        global_idx = seen_cards[sid]
        if faction == FACTION_AP:
            ap_physical_deck.append(global_idx)
        else:
            cp_physical_deck.append(global_idx)

    return {
        "adj": adj,
        "terrain": terrain,
        "vp": vp,
        "nation": nation,
        "fortress": fortress,
        "control": control,
        "ap_source_mask": ap_source_mask,
        "cp_source_mask": cp_source_mask,
        "card_faction": np.array(card_faction, dtype=np.int8),
        "card_ops": np.array(card_ops, dtype=np.float32),
        "card_sr": np.array(card_sr, dtype=np.float32),
        "card_is_combat": np.array(card_is_combat, dtype=np.bool_),
        "card_phase_gate": np.array(card_phase_gate, dtype=np.int8),
        "ap_physical_deck": np.array(ap_physical_deck, dtype=np.int16),
        "cp_physical_deck": np.array(cp_physical_deck, dtype=np.int16),
        "ap_unique_local": ap_unique_local,
        "cp_unique_local": cp_unique_local,
    }


_STATIC = _load_static_tables()
ADJ = jnp.asarray(_STATIC["adj"], dtype=jnp.bool_)
SPACE_TERRAIN = jnp.asarray(_STATIC["terrain"], dtype=jnp.float32)
SPACE_VP = jnp.asarray(_STATIC["vp"], dtype=jnp.float32)
SPACE_CONTROL = jnp.asarray(_STATIC["control"], dtype=jnp.int8)
SPACE_FORTRESS = jnp.asarray(_STATIC["fortress"], dtype=jnp.bool_)
AP_SOURCE_MASK = jnp.asarray(_STATIC["ap_source_mask"], dtype=jnp.bool_)
CP_SOURCE_MASK = jnp.asarray(_STATIC["cp_source_mask"], dtype=jnp.bool_)
CARD_OPS = jnp.asarray(_STATIC["card_ops"], dtype=jnp.float32)
CARD_SR = jnp.asarray(_STATIC["card_sr"], dtype=jnp.float32)
CARD_FACTION = jnp.asarray(_STATIC["card_faction"], dtype=jnp.int8)
CARD_IS_COMBAT = jnp.asarray(_STATIC["card_is_combat"], dtype=jnp.bool_)
CARD_PHASE_GATE = jnp.asarray(_STATIC["card_phase_gate"], dtype=jnp.int8)
AP_PHYSICAL_DECK = jnp.asarray(_STATIC["ap_physical_deck"], dtype=jnp.int16)
CP_PHYSICAL_DECK = jnp.asarray(_STATIC["cp_physical_deck"], dtype=jnp.int16)
AP_UNIQUE_LOCAL = jnp.asarray(_STATIC["ap_unique_local"], dtype=jnp.int16)
CP_UNIQUE_LOCAL = jnp.asarray(_STATIC["cp_unique_local"], dtype=jnp.int16)

UNIT_FACTION = UNIT_FACTION_INIT
UNIT_TYPE = UNIT_TYPE_INIT


class CRTResult(NamedTuple):
    attacker_losses: jnp.ndarray
    defender_losses: jnp.ndarray
    attacker_retreats: jnp.ndarray
    defender_retreats: jnp.ndarray
    defender_eliminated: jnp.ndarray


class JaxGameState(NamedTuple):
    unit_loc: jnp.ndarray
    unit_strength: jnp.ndarray
    trench_level: jnp.ndarray
    oos_mask: jnp.ndarray
    control: jnp.ndarray
    ap_hand: jnp.ndarray
    cp_hand: jnp.ndarray
    ap_discard: jnp.ndarray
    cp_discard: jnp.ndarray
    turn: jnp.ndarray
    action_round: jnp.ndarray
    active_player: jnp.ndarray
    war_status: jnp.ndarray
    vp: jnp.ndarray
    rng_key: jnp.ndarray


def _remove_card_from_hand(hand: jnp.ndarray, card_idx: jnp.ndarray) -> jnp.ndarray:
    match = hand == card_idx
    remove_idx = jnp.argmax(match.astype(jnp.int32))
    has_match = jnp.any(match)
    hand = jnp.where(has_match & (jnp.arange(HAND_SIZE) == remove_idx), EMPTY_CARD, hand)
    order = jnp.argsort(hand == EMPTY_CARD, stable=True)
    return hand[order]


def _mark_discard(discard: jnp.ndarray, local_idx: jnp.ndarray) -> jnp.ndarray:
    valid = (local_idx >= 0) & (local_idx < discard.shape[0])
    return jax.lax.cond(
        valid,
        lambda d: d.at[local_idx].set(True),
        lambda d: d,
        discard,
    )


def _reachable(control: jnp.ndarray, sources: jnp.ndarray) -> jnp.ndarray:
    def body(_, reach):
        nbr = jnp.any(ADJ & reach[None, :], axis=1)
        return reach | (nbr & control)

    initial = sources & control
    return jax.lax.fori_loop(0, N_SPACES, body, initial)


@jax.jit
def jax_oos(state: JaxGameState, player: int) -> jnp.ndarray:
    control = state.control == player
    sources = jax.lax.cond(player == FACTION_AP, lambda: AP_SOURCE_MASK, lambda: CP_SOURCE_MASK)
    reachable = _reachable(control, sources)
    return ~reachable


@jax.jit
def jax_zoc(state: JaxGameState, player: int) -> jnp.ndarray:
    combat_mask = (
        (UNIT_FACTION == player)
        & (state.unit_loc < N_SPACES)
        & (state.unit_strength > 0)
        & (
            (UNIT_TYPE == UNIT_INF)
            | (UNIT_TYPE == UNIT_CAV)
            | (UNIT_TYPE == UNIT_ART)
            | (UNIT_TYPE == UNIT_CORP)
            | (UNIT_TYPE == UNIT_ARMY)
        )
    )
    origins = jax.nn.one_hot(jnp.where(combat_mask, state.unit_loc, 0), N_SPACES, dtype=jnp.bool_)
    origins = jnp.where(combat_mask[:, None], origins, False)
    return jnp.any(origins @ ADJ, axis=0)


@jax.jit
def jax_crt(
    rng: jnp.ndarray,
    atk_str: jnp.ndarray,
    def_str: jnp.ndarray,
    drm: jnp.ndarray,
    trench_level: jnp.ndarray,
) -> CRTResult:
    trench_drm = jnp.minimum(trench_level, 2)
    eff_def = jnp.maximum(1.0, def_str.astype(jnp.float32) + trench_drm.astype(jnp.float32))
    ratio = jnp.maximum(0.0, atk_str.astype(jnp.float32) + drm.astype(jnp.float32)) / eff_def
    roll = jax.random.randint(rng, shape=(), minval=1, maxval=7)

    key = jnp.where(
        ratio < 0.5, 0,
        jnp.where(
            ratio < 1.0, jnp.where(roll >= 4, 1, 0),
            jnp.where(
                ratio < 1.5, jnp.where(roll >= 4, 2, 1),
                jnp.where(
                    ratio < 2.0, jnp.where(roll >= 3, 3, 2),
                    jnp.where(
                        ratio < 3.0, jnp.where(roll >= 3, 4, 3),
                        jnp.where(ratio < 4.0, jnp.where(roll >= 3, 5, 4), jnp.where(roll >= 4, 6, 5)),
                    ),
                ),
            ),
        ),
    )
    table = jnp.array(
        [
            [0, 0, 0, 0, 0],  # engaged
            [1, 0, 0, 0, 0],  # A1
            [1, 1, 0, 0, 0],  # EX
            [0, 1, 0, 0, 0],  # D1
            [0, 1, 0, 1, 0],  # D1R
            [0, 2, 0, 0, 0],  # D2
            [0, 0, 0, 0, 1],  # DE
        ],
        dtype=jnp.int32,
    )
    row = table[key]
    defender_losses = jnp.where((trench_level >= 3) & (row[1] > 0), row[1] - 1, row[1])
    return CRTResult(
        attacker_losses=row[0],
        defender_losses=defender_losses,
        attacker_retreats=row[2].astype(jnp.bool_),
        defender_retreats=row[3].astype(jnp.bool_),
        defender_eliminated=row[4].astype(jnp.bool_),
    )


def _sample_hand(deck: jnp.ndarray, perm: jnp.ndarray, n_cards: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    shuffled = deck[perm]
    hand = jnp.full((HAND_SIZE,), EMPTY_CARD, dtype=jnp.int16)
    hand = hand.at[:n_cards].set(shuffled[:n_cards])
    remaining = shuffled[n_cards:]
    return hand, remaining


@jax.jit
def jax_reset(rng_key: jnp.ndarray) -> JaxGameState:
    ap_key, cp_key = jax.random.split(rng_key)
    ap_perm = jax.random.permutation(ap_key, AP_PHYSICAL_DECK.shape[0])
    cp_perm = jax.random.permutation(cp_key, CP_PHYSICAL_DECK.shape[0])
    ap_hand, _ = _sample_hand(AP_PHYSICAL_DECK, ap_perm, 6)
    cp_hand, _ = _sample_hand(CP_PHYSICAL_DECK, cp_perm, 7)

    state = JaxGameState(
        unit_loc=INITIAL_UNIT_LOC,
        unit_strength=UNIT_STRENGTH_INIT,
        trench_level=jnp.zeros((N_SPACES,), dtype=jnp.int8),
        oos_mask=jnp.zeros((N_SPACES,), dtype=jnp.bool_),
        control=SPACE_CONTROL,
        ap_hand=ap_hand.astype(jnp.int16),
        cp_hand=cp_hand.astype(jnp.int16),
        ap_discard=jnp.zeros((FACTION_CARD_COUNT,), dtype=jnp.bool_),
        cp_discard=jnp.zeros((FACTION_CARD_COUNT,), dtype=jnp.bool_),
        turn=jnp.asarray(1, dtype=jnp.int8),
        action_round=jnp.asarray(1, dtype=jnp.int8),
        active_player=jnp.asarray(FACTION_AP, dtype=jnp.int8),
        war_status=jnp.asarray(PHASE_LIMITED, dtype=jnp.int8),
        vp=jnp.asarray(0, dtype=jnp.int8),
        rng_key=rng_key,
    )
    return state._replace(oos_mask=jax_oos(state, FACTION_AP) | jax_oos(state, FACTION_CP))


def _hand_for_player(state: JaxGameState, player: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.cond(player == FACTION_AP, lambda: state.ap_hand, lambda: state.cp_hand)


def _has_assault_card(hand: jnp.ndarray) -> jnp.ndarray:
    valid = hand < CARD_IS_COMBAT.shape[0]
    card_flags = jnp.where(valid, CARD_IS_COMBAT[jnp.clip(hand, 0, CARD_IS_COMBAT.shape[0] - 1)], False)
    return jnp.any(card_flags)


@jax.jit
def jax_legal_mask(state: JaxGameState) -> jnp.ndarray:
    player = state.active_player
    hand = _hand_for_player(state, player)
    legal = jnp.zeros((N_ACTIONS,), dtype=jnp.bool_).at[0].set(True)

    alive = (state.unit_loc < N_SPACES) & (state.unit_strength > 0)
    friendly_units = alive & (UNIT_FACTION == player)
    enemy_units = alive & (UNIT_FACTION == (1 - player))
    friendly_by_space = jnp.any(
        jax.nn.one_hot(jnp.where(friendly_units, state.unit_loc, 0), N_SPACES, dtype=jnp.bool_)
        & friendly_units[:, None],
        axis=0,
    )
    enemy_by_space = jnp.any(
        jax.nn.one_hot(jnp.where(enemy_units, state.unit_loc, 0), N_SPACES, dtype=jnp.bool_)
        & enemy_units[:, None],
        axis=0,
    )

    has_friendly = jnp.any(friendly_by_space)
    has_assault = _has_assault_card(hand)

    def apply_card(mask, card_idx):
        valid = card_idx < CARD_PHASE_GATE.shape[0]
        phase_gate = jnp.where(valid, CARD_PHASE_GATE[jnp.clip(card_idx, 0, CARD_PHASE_GATE.shape[0] - 1)], 2)
        phase_ok = valid & ~((phase_gate == PHASE_LIMITED) & (state.war_status == PHASE_TOTAL)) & ~(
            (phase_gate == PHASE_TOTAL) & (state.war_status == PHASE_LIMITED)
        )
        event_idx = ACT_EVENT_START + card_idx
        move_idx = ACT_OPS_START + card_idx * 3
        atk_idx = move_idx + 1
        sr_idx = move_idx + 2

        adjacency_to_enemy = jnp.any((ADJ & enemy_by_space[None, :]) & friendly_by_space[:, None])
        mask = jnp.where(phase_ok & (event_idx < ACT_OPS_START), mask.at[event_idx].set(True), mask)
        mask = jnp.where(phase_ok & has_friendly & (move_idx < ACT_MOVE_START), mask.at[move_idx].set(True), mask)
        mask = jnp.where(phase_ok & adjacency_to_enemy & (atk_idx < ACT_MOVE_START), mask.at[atk_idx].set(True), mask)
        mask = jnp.where(phase_ok & has_friendly & (sr_idx < ACT_MOVE_START), mask.at[sr_idx].set(True), mask)
        return mask, None

    legal, _ = jax.lax.scan(apply_card, legal, hand)

    src_ok = friendly_by_space[:, None]
    tgt_ok = ADJ
    attack_blocked = enemy_by_space[None, :] & (state.trench_level[None, :] >= 3) & (~has_assault)
    move_ok = src_ok & tgt_ok & ~(tgt_ok & attack_blocked)
    move_flat = move_ok.reshape(-1)[: N_ACTIONS - ACT_MOVE_START]
    legal = legal.at[ACT_MOVE_START:].set(move_flat)
    return legal


def _remove_active_card(state: JaxGameState, card_idx: jnp.ndarray) -> JaxGameState:
    is_ap = state.active_player == FACTION_AP
    ap_hand = jax.lax.cond(is_ap, lambda: _remove_card_from_hand(state.ap_hand, card_idx), lambda: state.ap_hand)
    cp_hand = jax.lax.cond(~is_ap, lambda: _remove_card_from_hand(state.cp_hand, card_idx), lambda: state.cp_hand)
    ap_discard = jax.lax.cond(
        is_ap,
        lambda: _mark_discard(state.ap_discard, AP_UNIQUE_LOCAL[jnp.clip(card_idx, 0, AP_UNIQUE_LOCAL.shape[0] - 1)]),
        lambda: state.ap_discard,
    )
    cp_discard = jax.lax.cond(
        ~is_ap,
        lambda: _mark_discard(state.cp_discard, CP_UNIQUE_LOCAL[jnp.clip(card_idx, 0, CP_UNIQUE_LOCAL.shape[0] - 1)]),
        lambda: state.cp_discard,
    )
    return state._replace(ap_hand=ap_hand, cp_hand=cp_hand, ap_discard=ap_discard, cp_discard=cp_discard)


def _advance_turn(state: JaxGameState) -> JaxGameState:
    next_action_round = state.action_round + jnp.asarray(1, dtype=jnp.int8)
    wrap = next_action_round > 7
    next_turn = state.turn + wrap.astype(jnp.int8)
    next_action_round = jnp.where(wrap, 1, next_action_round)
    next_player = (1 - state.active_player).astype(jnp.int8)
    return state._replace(turn=next_turn, action_round=next_action_round, active_player=next_player)


def _first_unit_at(state: JaxGameState, player: jnp.ndarray, src: jnp.ndarray) -> jnp.ndarray:
    matches = (UNIT_FACTION == player) & (state.unit_loc == src) & (state.unit_strength > 0)
    idx = jnp.argmax(matches.astype(jnp.int32))
    return jnp.where(jnp.any(matches), idx, -1)


def _combat_step(state: JaxGameState, src: jnp.ndarray, tgt: jnp.ndarray, rng_key: jnp.ndarray) -> JaxGameState:
    player = state.active_player
    atk_mask = (UNIT_FACTION == player) & (state.unit_loc == src) & (state.unit_strength > 0)
    def_mask = (UNIT_FACTION == (1 - player)) & (state.unit_loc == tgt) & (state.unit_strength > 0)
    atk_str = jnp.sum(atk_mask.astype(jnp.int32))
    def_str = jnp.sum(def_mask.astype(jnp.int32))
    crt = jax_crt(rng_key, atk_str, def_str, jnp.asarray(0, dtype=jnp.int32), state.trench_level[tgt])

    atk_order = jnp.where(atk_mask, jnp.arange(MAX_UNITS), MAX_UNITS)
    def_order = jnp.where(def_mask, jnp.arange(MAX_UNITS), MAX_UNITS)
    atk_sorted = jnp.sort(atk_order)
    def_sorted = jnp.sort(def_order)

    def eliminate_some(loc, idxs, count):
        def body(i, curr):
            idx = idxs[i]
            return jax.lax.cond(
                idx < MAX_UNITS,
                lambda arr: arr.at[idx].set(jnp.asarray(OFFBOARD, dtype=arr.dtype)),
                lambda arr: arr,
                curr,
            )

        return jax.lax.fori_loop(0, count, body, loc)

    unit_loc = state.unit_loc
    unit_strength = state.unit_strength
    unit_loc = eliminate_some(unit_loc, atk_sorted, crt.attacker_losses)
    unit_strength = jnp.where(unit_loc == OFFBOARD, 0, unit_strength)

    def_lost_all = crt.defender_eliminated
    unit_loc = jax.lax.cond(def_lost_all, lambda loc: eliminate_some(loc, def_sorted, jnp.sum(def_mask.astype(jnp.int32))), lambda loc: loc, unit_loc)
    unit_loc = jax.lax.cond(~def_lost_all, lambda loc: eliminate_some(loc, def_sorted, crt.defender_losses), lambda loc: loc, unit_loc)
    unit_strength = jnp.where(unit_loc == OFFBOARD, 0, unit_strength)
    control = jnp.where(def_lost_all, state.control.at[tgt].set(player), state.control)
    return state._replace(unit_loc=unit_loc, unit_strength=unit_strength, control=control)


def _move_step(state: JaxGameState, src: jnp.ndarray, tgt: jnp.ndarray, rng_key: jnp.ndarray) -> JaxGameState:
    player = state.active_player
    enemy_present = jnp.any((UNIT_FACTION == (1 - player)) & (state.unit_loc == tgt) & (state.unit_strength > 0))

    def do_combat(s):
        return _combat_step(s, src, tgt, rng_key)

    def do_move(s):
        unit_idx = _first_unit_at(s, player, src)
        unit_loc = jax.lax.cond(
            unit_idx >= 0,
            lambda loc: loc.at[unit_idx].set(tgt.astype(loc.dtype)),
            lambda loc: loc,
            s.unit_loc,
        )
        control = s.control.at[tgt].set(player)
        return s._replace(unit_loc=unit_loc, control=control)

    return jax.lax.cond(enemy_present, do_combat, do_move, state)


@jax.jit
def jax_step(state: JaxGameState, action: jnp.ndarray) -> tuple[JaxGameState, jnp.ndarray, jnp.ndarray]:
    key, step_key = jax.random.split(state.rng_key)

    def do_pass(s):
        return s

    def do_event(s):
        card_idx = action - ACT_EVENT_START
        return _remove_active_card(s, card_idx.astype(jnp.int16))

    def do_ops(s):
        card_idx = ((action - ACT_OPS_START) // 3).astype(jnp.int16)
        return _remove_active_card(s, card_idx)

    def do_move(s):
        offset = action - ACT_MOVE_START
        src = (offset // N_SPACES).astype(jnp.int32)
        tgt = (offset % N_SPACES).astype(jnp.int32)
        return _move_step(s, src, tgt, step_key)

    state = jax.lax.cond(
        action == ACT_PASS,
        do_pass,
        lambda s: jax.lax.cond(
            action < ACT_OPS_START,
            do_event,
            lambda s2: jax.lax.cond(action < ACT_MOVE_START, do_ops, do_move, s2),
            s,
        ),
        state,
    )
    state = _advance_turn(state)._replace(rng_key=key)
    oos = jax_oos(state, FACTION_AP) | jax_oos(state, FACTION_CP)
    state = state._replace(oos_mask=oos)

    done = (jnp.abs(state.vp) >= 20) | (state.turn > 20)
    reward = jnp.asarray(0.0, dtype=jnp.float32)
    return state, reward, done


@jax.jit
def jax_obs(state: JaxGameState, player: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    obs = jnp.zeros((N_PLANES, N_SPACES), dtype=jnp.float32)
    obs = obs.at[0].set(SPACE_TERRAIN / 4.0)
    obs = obs.at[1].set(SPACE_VP / 5.0)
    ctrl = state.control.astype(jnp.float32)
    obs = obs.at[2].set(jnp.where(ctrl == FACTION_AP, 0.0, jnp.where(ctrl == FACTION_NEUTRAL, 0.5, 1.0)))
    obs = obs.at[3].set(state.trench_level.astype(jnp.float32) / 3.0)
    obs = obs.at[4].set((state.trench_level >= 3).astype(jnp.float32))
    obs = obs.at[5].set(jnp.zeros((N_SPACES,), dtype=jnp.float32))
    obs = obs.at[6].set(jax_oos(state, FACTION_AP).astype(jnp.float32))
    obs = obs.at[7].set(jax_oos(state, FACTION_CP).astype(jnp.float32))

    alive = (state.unit_loc < N_SPACES) & (state.unit_strength > 0)
    loc_one_hot = jax.nn.one_hot(jnp.where(alive, state.unit_loc, 0), N_SPACES, dtype=jnp.float32)
    loc_one_hot = loc_one_hot * alive[:, None]

    def unit_plane(player_id: int, unit_type: int) -> jnp.ndarray:
        mask = alive & (UNIT_FACTION == player_id) & (UNIT_TYPE == unit_type)
        return jnp.clip(jnp.sum(loc_one_hot * mask[:, None], axis=0) / 3.0, 0.0, 1.0)

    for i in range(N_UNIT_TYPES):
        obs = obs.at[8 + i].set(unit_plane(FACTION_AP, i))
        obs = obs.at[16 + i].set(unit_plane(FACTION_CP, i))

    ap_valid = state.ap_hand < CARD_OPS.shape[0]
    cp_valid = state.cp_hand < CARD_OPS.shape[0]
    ap_ops = jnp.where(ap_valid, CARD_OPS[jnp.clip(state.ap_hand, 0, CARD_OPS.shape[0] - 1)], 0.0)
    cp_ops = jnp.where(cp_valid, CARD_OPS[jnp.clip(state.cp_hand, 0, CARD_OPS.shape[0] - 1)], 0.0)

    obs = obs.at[24].set(jnp.sum(ap_valid).astype(jnp.float32) / 10.0)
    obs = obs.at[25].set(jnp.sum(cp_valid).astype(jnp.float32) / 10.0)
    obs = obs.at[26].set(jnp.where(jnp.any(ap_valid), jnp.sum(ap_ops) / jnp.sum(ap_valid) / 5.0, 0.0))
    obs = obs.at[27].set(jnp.where(jnp.any(cp_valid), jnp.sum(cp_ops) / jnp.sum(cp_valid) / 5.0, 0.0))
    obs = obs.at[28].set((state.vp.astype(jnp.float32) + 20.0) / 40.0)
    obs = obs.at[29].set((state.war_status == PHASE_TOTAL).astype(jnp.float32))
    obs = obs.at[30].set((state.war_status == PHASE_TOTAL).astype(jnp.float32))
    obs = obs.at[31].set((state.active_player == FACTION_CP).astype(jnp.float32))

    hand = jax.lax.cond(player == FACTION_AP, lambda: state.ap_hand, lambda: state.cp_hand)
    valid = hand < CARD_OPS.shape[0]
    clipped = jnp.clip(hand, 0, CARD_OPS.shape[0] - 1)
    ctx = jnp.zeros((HAND_SIZE, 16), dtype=jnp.float32)
    ctx = ctx.at[:, 0].set(jnp.where(valid, CARD_OPS[clipped] / 5.0, 0.0))
    ctx = ctx.at[:, 1].set(jnp.where(valid, CARD_SR[clipped] / 5.0, 0.0))
    ctx = ctx.at[:, 2].set(jnp.where(valid, CARD_IS_COMBAT[clipped].astype(jnp.float32), 0.0))
    ctx = ctx.at[:, 3].set(jnp.where(valid, CARD_PHASE_GATE[clipped].astype(jnp.float32) / 2.0, 0.0))
    ctx = ctx.at[:, 4].set(jnp.where(valid, (CARD_FACTION[clipped] == FACTION_CP).astype(jnp.float32), 0.0))
    return obs, ctx
