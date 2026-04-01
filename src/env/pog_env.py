"""
pog_env.py — Paths of Glory RL environment (PettingZoo AEC-style).

Observation:
  'spatial':      np.ndarray (N_PLANES, N_SPACES) float32
  'card_context': np.ndarray (7, 16)              float32

Action: int in [0, N_ACTIONS-1]
Action mask: np.ndarray (N_ACTIONS,) bool
"""

import numpy as np
import json
import random
from collections import deque
from typing import Optional, Dict, List, Tuple
from src.data.starting_positions import INITIAL_UNIT_LOC, UNIT_FACTION_INIT, UNIT_STRENGTH_INIT, UNIT_TYPE_INIT

try:
    from src.data.pog_engine import (
        GameState, MapSpace, Unit, Card,
        build_observation_tensor, build_card_context, compute_action_mask,
        FACTION_AP, FACTION_CP, FACTION_NEUTRAL,
        PHASE_LIMITED, PHASE_TOTAL, PHASE_EITHER,
        N_SPACES, N_PLANES, N_ACTIONS,
        UNIT_INF, UNIT_CAV, UNIT_ART, UNIT_CORP, UNIT_ARMY,
        TERRAIN_STR_MAP, NATION_STR_MAP, PHASE_STR_MAP,
        AP_NATIONS, CP_NATIONS, N_UNIT_TYPES,
    )
except ImportError:
    # Fallback constants for standalone testing
    FACTION_AP = 0; FACTION_CP = 1; FACTION_NEUTRAL = 2
    PHASE_LIMITED = 0; PHASE_TOTAL = 1; PHASE_EITHER = 2
    N_SPACES = 72; N_PLANES = 32; N_ACTIONS = 5341
    UNIT_INF = 0; UNIT_CAV = 1; UNIT_ART = 2; UNIT_CORP = 3; UNIT_ARMY = 4
    TERRAIN_STR_MAP = {"Clear": 0, "Fort": 1, "Mountain": 2, "Sea": 3, "Desert": 4}
    NATION_STR_MAP = {"FR":0,"GE":1,"BE":2,"RU":3,"AH":4,"TU":5,"IT":6,"BU":7,"SE":8,"RO":9,"NE":10,"GR":11,"BR":12}
    PHASE_STR_MAP = {"Limited War":0,"Total War":1,"Either":2,"Unknown":2}
    AP_NATIONS = {0, 2, 3, 6, 11, 12, 9, 8}
    CP_NATIONS = {1, 4, 5, 7}
    N_UNIT_TYPES = 8

# Action index boundaries
ACT_PASS       = 0
ACT_EVENT_START = 1    # [1, 111)
ACT_OPS_START  = 111   # [111, 441)
ACT_MOVE_START = 441   # [441, 5341)


class PogEnv:
    """
    Paths of Glory RL environment.

    Usage:
        env = PogEnv("pog_map_graph.json", "pog_cards_db.json")
        obs = env.reset()
        mask = env.action_mask("AP")
        legal = np.where(mask)[0]
        obs, rew, done, trunc, info = env.step(legal[0])
    """

    metadata = {"name": "paths_of_glory_v0", "render_modes": ["human", "ansi"]}
    possible_agents = ["AP", "CP"]

    def __init__(self, map_json: str = "pog_map_graph.json",
                 cards_json: str = "pog_cards_db.json"):
        self.map_json   = map_json
        self.cards_json = cards_json
        self.agents     = list(self.possible_agents)

        # Runtime state
        self._n_spaces: int = 0
        self.turn: int = 1
        self.action_round: int = 1
        self.active_player: int = FACTION_AP
        self.war_status_ap: int = PHASE_LIMITED
        self.war_status_cp: int = PHASE_LIMITED
        self.vp_track: int = 0

        # Per-space arrays (set in reset)
        self._space_terrain:  Optional[np.ndarray] = None  # (N,) int8
        self._space_vp:       Optional[np.ndarray] = None  # (N,) int8
        self._space_nation:   Optional[np.ndarray] = None  # (N,) int8
        self._space_fortress: Optional[np.ndarray] = None  # (N,) bool
        self._space_control:  Optional[np.ndarray] = None  # (N,) int8
        self._trench_levels:  Optional[np.ndarray] = None  # (N,) int8
        self._oos_ap:         Optional[np.ndarray] = None  # (N,) bool
        self._oos_cp:         Optional[np.ndarray] = None  # (N,) bool
        self._fort_destroyed: Optional[np.ndarray] = None  # (N,) bool
        self._adj:            Optional[np.ndarray] = None  # (N, N) bool

        # Units as list of dicts
        self._units: List[dict] = []

        # Card DB and hands
        self._cards_db: List[dict] = []
        self._spaces_db: List[dict] = []
        self._space_id_to_idx: Dict[str, int] = {}
        self._ap_hand: List[int] = []
        self._cp_hand: List[int] = []
        self._ap_deck: List[int] = []
        self._cp_deck: List[int] = []
        self._ap_supply_sources: set = set()
        self._cp_supply_sources: set = set()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> Dict:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._load_data()

        self.turn          = 1
        self.action_round  = 1
        self.active_player = FACTION_AP
        self.war_status_ap = PHASE_LIMITED
        self.war_status_cp = PHASE_LIMITED
        self.vp_track      = 0

        n = self._n_spaces
        self._trench_levels  = np.zeros(n, dtype=np.int8)
        self._oos_ap         = np.zeros(n, dtype=bool)
        self._oos_cp         = np.zeros(n, dtype=bool)
        self._fort_destroyed = np.zeros(n, dtype=bool)
        self._units          = self._build_starting_units()

        self._update_oos()

        # Deal opening hands: AP=6, CP=7
        random.shuffle(self._ap_deck)
        random.shuffle(self._cp_deck)
        self._ap_hand = self._ap_deck[:6]
        self._ap_deck = self._ap_deck[6:]
        self._cp_hand = self._cp_deck[:7]
        self._cp_deck = self._cp_deck[7:]

        return {ag: self.observe(ag) for ag in self.agents}

    def _build_starting_units(self) -> List[dict]:
        units: List[dict] = []
        for idx in range(len(INITIAL_UNIT_LOC)):
            faction = int(UNIT_FACTION_INIT[idx])
            strength = int(UNIT_STRENGTH_INIT[idx])
            if idx == 0 or faction < 0 or strength <= 0:
                continue
            loc = int(INITIAL_UNIT_LOC[idx])
            units.append({
                "unit_id": idx,
                "faction": faction,
                "unit_type": int(UNIT_TYPE_INIT[idx]),
                "strength": strength,
                "location": loc,
                "is_eliminated": loc >= N_SPACES,
                "has_moved": False,
            })
        return units

    def step(self, action: int) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        agent = self._agent_name(self.active_player)
        assert self.action_mask(agent)[action], f"Illegal action {action} for {agent}"

        if action == ACT_PASS:
            pass

        elif ACT_EVENT_START <= action < ACT_OPS_START:
            self._play_event(action - ACT_EVENT_START)

        elif ACT_OPS_START <= action < ACT_MOVE_START:
            offset   = action - ACT_OPS_START
            card_idx = offset // 3
            op_type  = offset % 3
            self._play_ops(card_idx, op_type)

        else:
            offset    = action - ACT_MOVE_START
            src_space = offset // N_SPACES
            tgt_space = offset % N_SPACES
            self._execute_move(src_space, tgt_space)

        self._advance_turn()
        self._update_oos()

        done   = {ag: False for ag in self.agents}
        reward = {ag: 0.0   for ag in self.agents}
        trunc  = {ag: False for ag in self.agents}

        if abs(self.vp_track) >= 20 or self.turn > 20:
            for ag in self.agents:
                done[ag] = True
            winner = FACTION_AP if self.vp_track <= -10 else FACTION_CP
            reward["AP"] =  1.0 if winner == FACTION_AP else -1.0
            reward["CP"] = -reward["AP"]

        obs = {ag: self.observe(ag) for ag in self.agents}
        return obs, reward, done, trunc, {}

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        faction  = FACTION_AP if agent == "AP" else FACTION_CP
        spatial  = self._build_spatial_obs()
        card_ctx = self._build_card_context(faction)
        return {"spatial": spatial, "card_context": card_ctx}

    def action_mask(self, agent: str) -> np.ndarray:
        faction = FACTION_AP if agent == "AP" else FACTION_CP
        return self._compute_action_mask_for(faction)

    def render(self, mode: str = "human") -> Optional[str]:
        lines = [
            f"\n=== PoG T{self.turn} AR{self.action_round} "
            f"| Active: {'AP' if self.active_player==FACTION_AP else 'CP'} "
            f"| VP: {self.vp_track:+d} ==="
        ]
        for i in range(self._n_spaces):
            ctrl   = ["AP", "CP", "--"][int(self._space_control[i])]
            trench = int(self._trench_levels[i])
            units_here = [u for u in self._units if u["location"] == i and not u["is_eliminated"]]
            if units_here or trench > 0:
                lines.append(f"  [{i:02d}] {ctrl} trench={trench} units={len(units_here)}")
        lines.append(
            f"  AP hand: {len(self._ap_hand)} | CP hand: {len(self._cp_hand)} "
            f"| AP deck: {len(self._ap_deck)} | CP deck: {len(self._cp_deck)}"
        )
        result = "\n".join(lines)
        if mode == "human":
            print(result)
        return result

    def close(self):
        pass

    # ──────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────

    def _load_data(self):
        with open(self.map_json) as f:
            map_data = json.load(f)
        with open(self.cards_json) as f:
            cards_raw = json.load(f)

        if isinstance(map_data, list):
            spaces_raw = []
            for item in map_data:
                for name, info in item.items():
                    spaces_raw.append({"id": name.upper().replace(" ", "_"), **info})
        else:
            spaces_raw = map_data.get("spaces", [])

        self._n_spaces = len(spaces_raw)
        self._spaces_db = spaces_raw
        self._space_id_to_idx = {s["id"]: i for i, s in enumerate(spaces_raw)}

        n = self._n_spaces
        self._space_terrain  = np.array([TERRAIN_STR_MAP.get(s.get("terrain", "Clear"), 0) for s in spaces_raw], dtype=np.int8)
        self._space_vp       = np.array([s.get("vp_value", 0) for s in spaces_raw], dtype=np.int8)
        self._space_nation   = np.array([NATION_STR_MAP.get(s.get("nation", "NE"), 10) for s in spaces_raw], dtype=np.int8)
        self._space_fortress = np.array([s.get("is_fortress", False) for s in spaces_raw], dtype=bool)
        self._space_control  = np.full(n, FACTION_NEUTRAL, dtype=np.int8)

        for i, s in enumerate(spaces_raw):
            nat_id = NATION_STR_MAP.get(s.get("nation", "NE"), 10)
            if nat_id in AP_NATIONS:
                self._space_control[i] = FACTION_AP
            elif nat_id in CP_NATIONS:
                self._space_control[i] = FACTION_CP

        self._adj = self._build_adj_matrix(spaces_raw)

        self._ap_supply_sources = {
            i for i, s in enumerate(spaces_raw)
            if s.get("nation", "") in {"FR", "BE"} and self._space_control[i] == FACTION_AP
        }
        self._cp_supply_sources = {
            i for i, s in enumerate(spaces_raw)
            if s.get("nation", "") in {"GE", "AH"} and self._space_control[i] == FACTION_CP
        }

        # Build card DB
        seen: dict = {}
        self._cards_db = []
        self._ap_deck = []
        self._cp_deck = []

        for entry in cards_raw:
            sid = entry["id"]
            faction_str = entry.get("faction", "AP")
            faction_int = FACTION_AP if faction_str == "AP" else FACTION_CP
            phase_int   = PHASE_STR_MAP.get(entry.get("phase", "Either"), PHASE_EITHER)

            if sid not in seen:
                seen[sid] = len(self._cards_db)
                self._cards_db.append({
                    "str_id":        sid,
                    "faction":       faction_int,
                    "ops":           entry.get("ops", 2),
                    "sr":            entry.get("sr", 2),
                    "is_combat_card": entry.get("is_combat_card", False),
                    "phase_gate":    phase_int,
                    "event_text":    entry.get("event_text", ""),
                })

            idx = seen[sid]
            if faction_int == FACTION_AP:
                self._ap_deck.append(idx)
            else:
                self._cp_deck.append(idx)

    def _build_adj_matrix(self, spaces_raw: List[dict]) -> np.ndarray:
        n = len(spaces_raw)
        adj = np.zeros((n, n), dtype=bool)
        for i, s in enumerate(spaces_raw):
            for conn in s.get("connections", []):
                j = self._space_id_to_idx.get(conn.upper().replace(" ", "_"), -1)
                if j >= 0:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    # ──────────────────────────────────────────────────────────────────
    # Game logic
    # ──────────────────────────────────────────────────────────────────

    def _play_event(self, card_idx: int):
        hand = self._ap_hand if self.active_player == FACTION_AP else self._cp_hand
        if card_idx in hand:
            hand.remove(card_idx)
        # Individual event resolution not yet implemented

    def _play_ops(self, card_idx: int, op_type: int):
        hand = self._ap_hand if self.active_player == FACTION_AP else self._cp_hand
        if card_idx in hand:
            hand.remove(card_idx)
        # op_type: 0=MOVE, 1=ATTACK, 2=SR
        # Actual unit movement is via MOVE_UNIT actions

    def _execute_move(self, src: int, tgt: int):
        faction = self.active_player
        # Move first available friendly unit from src to tgt
        for u in self._units:
            if u["faction"] == faction and u["location"] == src and not u["is_eliminated"]:
                enemy_at_tgt = any(
                    eu["faction"] != faction and eu["location"] == tgt and not eu["is_eliminated"]
                    for eu in self._units
                )
                if enemy_at_tgt:
                    atk_str = sum(1 for au in self._units
                                  if au["faction"] == faction and au["location"] == src and not au["is_eliminated"])
                    def_str = sum(1 for du in self._units
                                  if du["faction"] != faction and du["location"] == tgt and not du["is_eliminated"])
                    trench  = int(self._trench_levels[tgt]) if self._trench_levels is not None else 0
                    result  = self._resolve_crt(atk_str, def_str, 0, 0, trench)

                    def_units = [du for du in self._units
                                 if du["faction"] != faction and du["location"] == tgt and not du["is_eliminated"]]
                    atk_units = [au for au in self._units
                                 if au["faction"] == faction and au["location"] == src and not au["is_eliminated"]]

                    for _ in range(result["attacker_losses"]):
                        if atk_units:
                            atk_units.pop()["is_eliminated"] = True

                    if result["defender_eliminated"]:
                        for du in def_units:
                            du["is_eliminated"] = True
                        self._space_control[tgt] = faction
                    else:
                        for _ in range(result["defender_losses"]):
                            if def_units:
                                def_units.pop()["is_eliminated"] = True
                        if result["defender_retreats"] and def_units:
                            for nb in np.where(self._adj[tgt])[0]:
                                if self._space_control[int(nb)] != (1 - faction):
                                    for du in def_units:
                                        du["location"] = int(nb)
                                    break
                else:
                    u["location"] = tgt
                    self._space_control[tgt] = faction
                break

    def _advance_turn(self):
        self.action_round += 1
        if self.action_round > 7:
            self.action_round = 1
            # Draw cards
            if self._ap_deck:
                self._ap_hand.append(self._ap_deck.pop(0))
            if self._cp_deck:
                self._cp_hand.append(self._cp_deck.pop(0))
            self.turn += 1
        self.active_player = 1 - self.active_player

    def _update_oos(self):
        if self._adj is None or self._n_spaces == 0:
            return
        n = self._n_spaces
        for i in range(n):
            self._oos_ap[i] = self._check_oos(i, FACTION_AP)
            self._oos_cp[i] = self._check_oos(i, FACTION_CP)

    # ──────────────────────────────────────────────────────────────────
    # Core game mechanics (publicly accessible for testing)
    # ──────────────────────────────────────────────────────────────────

    def _resolve_crt(self, attacker_str: int, defender_str: int,
                     atk_drm: int, def_drm: int, trench_level: int) -> dict:
        """
        Combat Results Table.

        Trench DRM capped at +2 (level 3 → +2, not +3).
        Level-3 trench absorbs first defender step loss.
        """
        CRT = {
            "engaged": dict(attacker_losses=0, defender_losses=0, attacker_retreats=False, defender_retreats=False, defender_eliminated=False),
            "A1":      dict(attacker_losses=1, defender_losses=0, attacker_retreats=False, defender_retreats=False, defender_eliminated=False),
            "EX":      dict(attacker_losses=1, defender_losses=1, attacker_retreats=False, defender_retreats=False, defender_eliminated=False),
            "D1":      dict(attacker_losses=0, defender_losses=1, attacker_retreats=False, defender_retreats=False, defender_eliminated=False),
            "D1R":     dict(attacker_losses=0, defender_losses=1, attacker_retreats=False, defender_retreats=True,  defender_eliminated=False),
            "D2":      dict(attacker_losses=0, defender_losses=2, attacker_retreats=False, defender_retreats=False, defender_eliminated=False),
            "DE":      dict(attacker_losses=0, defender_losses=0, attacker_retreats=False, defender_retreats=False, defender_eliminated=True),
        }
        trench_drm = min(trench_level, 2)
        eff_def    = max(1, defender_str + trench_drm + def_drm)
        ratio      = max(0.0, (attacker_str + atk_drm)) / eff_def
        roll       = random.randint(1, 6)

        if ratio < 0.5:
            key = "engaged"
        elif ratio < 1.0:
            key = "A1" if roll >= 4 else "engaged"
        elif ratio < 1.5:
            key = "EX" if roll >= 4 else "A1"
        elif ratio < 2.0:
            key = "D1" if roll >= 3 else "EX"
        elif ratio < 3.0:
            key = "D1R" if roll >= 3 else "D1"
        elif ratio < 4.0:
            key = "D2" if roll >= 3 else "D1R"
        else:
            key = "DE" if roll >= 4 else "D2"

        result = dict(CRT[key])
        # Level-3 trench absorbs first defender step loss
        if trench_level >= 3 and result["defender_losses"] > 0:
            result["defender_losses"] = max(0, result["defender_losses"] - 1)

        return result

    def _check_zoc(self, space_idx: int, enemy_faction: int) -> bool:
        """
        Return True if space_idx is in ZOC of enemy_faction.
        ZOC projected by all non-eliminated enemy combat units (INF/CAV/ART/CORP/ARMY)
        into all adjacent spaces.
        """
        if self._adj is None:
            return False
        combat_types = {UNIT_INF, UNIT_CAV, UNIT_ART, UNIT_CORP, UNIT_ARMY}
        for u in self._units:
            if u["is_eliminated"] or u["faction"] != enemy_faction:
                continue
            if u.get("unit_type", UNIT_INF) not in combat_types:
                continue
            uloc = u["location"]
            if 0 <= uloc < self._n_spaces and self._adj[uloc, space_idx]:
                return True
        return False

    def _check_oos(self, space_idx: int, faction: int) -> bool:
        """
        BFS supply check. Returns True if OUT OF SUPPLY.
        Supply path must trace through contiguous friendly-controlled spaces to a supply source.
        """
        if self._adj is None or self._n_spaces == 0:
            return False
        sources = self._ap_supply_sources if faction == FACTION_AP else self._cp_supply_sources
        if space_idx in sources:
            return False

        visited: set = set()
        queue = deque([space_idx])
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            if curr in sources:
                return False
            for nxt_arr in np.where(self._adj[curr])[0]:
                nxt = int(nxt_arr)
                if nxt not in visited and self._space_control[nxt] == faction:
                    queue.append(nxt)
        return True

    # ──────────────────────────────────────────────────────────────────
    # Observation & mask builders
    # ──────────────────────────────────────────────────────────────────

    def _compute_action_mask_for(self, faction: int) -> np.ndarray:
        mask    = np.zeros(N_ACTIONS, dtype=bool)
        mask[0] = True  # PASS always legal

        war_st = self.war_status_ap if faction == FACTION_AP else self.war_status_cp
        hand   = self._ap_hand if faction == FACTION_AP else self._cp_hand
        n      = self._n_spaces

        friendly_locs = {u["location"] for u in self._units if u["faction"] == faction and not u["is_eliminated"]}
        enemy_locs    = {u["location"] for u in self._units if u["faction"] != faction and not u["is_eliminated"]}

        has_assault = any(self._cards_db[c]["is_combat_card"] for c in hand if c < len(self._cards_db))

        for card_idx in hand:
            if card_idx >= len(self._cards_db):
                continue
            card = self._cards_db[card_idx]
            pg   = card["phase_gate"]

            if pg == PHASE_LIMITED and war_st == PHASE_TOTAL:
                continue
            if pg == PHASE_TOTAL  and war_st == PHASE_LIMITED:
                continue

            ev = ACT_EVENT_START + card_idx
            if ev < ACT_OPS_START:
                mask[ev] = True

            s_move = ACT_OPS_START + card_idx * 3
            if s_move < ACT_MOVE_START and friendly_locs:
                mask[s_move] = True

            s_atk = ACT_OPS_START + card_idx * 3 + 1
            if s_atk < ACT_MOVE_START:
                for src in friendly_locs:
                    if self._adj is not None:
                        adj_nbrs = set(np.where(self._adj[src])[0].tolist())
                        if adj_nbrs & enemy_locs:
                            mask[s_atk] = True
                            break

            s_sr = ACT_OPS_START + card_idx * 3 + 2
            if s_sr < ACT_MOVE_START and friendly_locs:
                mask[s_sr] = True

        if self._adj is not None:
            for src in friendly_locs:
                if src >= n:
                    continue
                for tgt in np.where(self._adj[src])[0]:
                    tgt = int(tgt)
                    if tgt >= n:
                        continue
                    if tgt in enemy_locs and self._trench_levels is not None:
                        if self._trench_levels[tgt] >= 3 and not has_assault:
                            continue
                    flat = ACT_MOVE_START + src * N_SPACES + tgt
                    if flat < N_ACTIONS:
                        mask[flat] = True

        return mask

    def _build_spatial_obs(self) -> np.ndarray:
        n   = self._n_spaces
        obs = np.zeros((N_PLANES, N_SPACES), dtype=np.float32)
        if n == 0:
            return obs

        m = min(n, N_SPACES)
        obs[0, :m] = self._space_terrain[:m]  / 4.0
        obs[1, :m] = self._space_vp[:m]       / 5.0
        ctrl = self._space_control[:m].astype(np.float32)
        obs[2, :m] = np.where(ctrl == FACTION_AP, 0.0, np.where(ctrl == FACTION_NEUTRAL, 0.5, 1.0))
        obs[3, :m] = self._trench_levels[:m]  / 3.0
        obs[4, :m] = (self._trench_levels[:m] >= 3).astype(np.float32)
        obs[5, :m] = self._fort_destroyed[:m].astype(np.float32)
        obs[6, :m] = self._oos_ap[:m].astype(np.float32)
        obs[7, :m] = self._oos_cp[:m].astype(np.float32)

        unit_ap = np.zeros((N_UNIT_TYPES, m), dtype=np.float32)
        unit_cp = np.zeros((N_UNIT_TYPES, m), dtype=np.float32)
        for u in self._units:
            if u["is_eliminated"] or u["location"] >= m:
                continue
            ut = u.get("unit_type", UNIT_INF)
            if u["faction"] == FACTION_AP:
                unit_ap[ut, u["location"]] += 1
            else:
                unit_cp[ut, u["location"]] += 1

        obs[8:16,  :m] = np.clip(unit_ap / 3.0, 0, 1)
        obs[16:24, :m] = np.clip(unit_cp / 3.0, 0, 1)

        ap_ops = [self._cards_db[c]["ops"] for c in self._ap_hand if c < len(self._cards_db)]
        cp_ops = [self._cards_db[c]["ops"] for c in self._cp_hand if c < len(self._cards_db)]

        obs[24, :] = len(self._ap_hand) / 10.0
        obs[25, :] = len(self._cp_hand) / 10.0
        obs[26, :] = (float(np.mean(ap_ops)) / 5.0) if ap_ops else 0.0
        obs[27, :] = (float(np.mean(cp_ops)) / 5.0) if cp_ops else 0.0
        obs[28, :] = (self.vp_track + 20) / 40.0
        obs[29, :] = float(self.war_status_ap == PHASE_TOTAL)
        obs[30, :] = float(self.war_status_cp == PHASE_TOTAL)
        obs[31, :] = float(self.active_player == FACTION_CP)
        return obs

    def _build_card_context(self, faction: int, n_slots: int = 7, embed_dim: int = 16) -> np.ndarray:
        hand = self._ap_hand if faction == FACTION_AP else self._cp_hand
        ctx  = np.zeros((n_slots, embed_dim), dtype=np.float32)
        for slot, card_idx in enumerate(hand[:n_slots]):
            if card_idx < len(self._cards_db):
                c = self._cards_db[card_idx]
                ctx[slot, 0] = c["ops"]  / 5.0
                ctx[slot, 1] = c["sr"]   / 5.0
                ctx[slot, 2] = float(c["is_combat_card"])
                ctx[slot, 3] = c["phase_gate"] / 2.0
                ctx[slot, 4] = float(c["faction"] == FACTION_CP)
        return ctx

    @staticmethod
    def _agent_name(faction: int) -> str:
        return "AP" if faction == FACTION_AP else "CP"
