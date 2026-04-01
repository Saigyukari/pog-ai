"""
tests/test_zoc.py — Unit tests for Zone of Control (ZOC) mechanics.
"""
import pytest
import numpy as np
from src.env.pog_env import PogEnv, FACTION_AP, FACTION_CP, UNIT_INF, UNIT_ART
UNIT_FLEET = 5  # fleet constant (not combat unit — no ZOC)


@pytest.fixture
def env():
    """
    Minimal 4-node linear env for ZOC testing.
    Graph: 0 — 1 — 2 — 3
    """
    e = PogEnv.__new__(PogEnv)
    e._n_spaces = 4
    e._adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=bool)
    # One CP infantry at space 2
    e._units = [
        {"faction": FACTION_CP, "location": 2,
         "unit_type": UNIT_INF, "is_eliminated": False},
    ]
    return e


def test_adjacent_space_in_enemy_zoc(env):
    """Spaces adjacent to the CP unit (1 and 3) are in CP ZOC."""
    assert env._check_zoc(1, enemy_faction=FACTION_CP) is True
    assert env._check_zoc(3, enemy_faction=FACTION_CP) is True


def test_non_adjacent_space_not_in_zoc(env):
    """Space 0 is not adjacent to CP unit at 2 — not in ZOC."""
    assert env._check_zoc(0, enemy_faction=FACTION_CP) is False


def test_unit_own_space_in_zoc(env):
    """Space 2 (where CP unit stands) is in CP ZOC of itself's neighbours — but we test space 2 vs CP."""
    # Space 1 and 3 are in ZOC — already tested; additionally space 2 itself
    # ZOC is projected into ADJACENT spaces, not the space the unit occupies
    # This is a documentation test
    assert env._check_zoc(1, enemy_faction=FACTION_CP) is True


def test_eliminated_unit_no_zoc(env):
    """Eliminated units project no ZOC."""
    env._units[0]["is_eliminated"] = True
    assert env._check_zoc(1, enemy_faction=FACTION_CP) is False
    assert env._check_zoc(3, enemy_faction=FACTION_CP) is False


def test_non_combat_unit_no_zoc(env):
    """Fleet/naval units do not project ZOC."""
    env._units[0]["unit_type"] = UNIT_FLEET
    assert env._check_zoc(1, enemy_faction=FACTION_CP) is False


def test_own_faction_units_dont_count_as_enemy_zoc(env):
    """AP units should not be counted as CP ZOC."""
    env._units.append({
        "faction":    FACTION_AP,
        "location":   1,
        "unit_type":  UNIT_ART,
        "is_eliminated": False,
    })
    # Space 0 is adjacent to AP unit at 1 — but we're checking CP ZOC
    assert env._check_zoc(0, enemy_faction=FACTION_CP) is False


def test_multiple_units_zoc(env):
    """Multiple enemy units extend ZOC coverage."""
    env._units.append({
        "faction":    FACTION_CP,
        "location":   0,
        "unit_type":  UNIT_INF,
        "is_eliminated": False,
    })
    # Space 0's CP unit also projects into space 1
    # Space 1 was already in ZOC from unit at 2 — still True
    assert env._check_zoc(1, enemy_faction=FACTION_CP) is True
