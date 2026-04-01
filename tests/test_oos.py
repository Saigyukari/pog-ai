"""
tests/test_oos.py — Unit tests for Out-of-Supply (OOS) checks.
"""
import pytest
import numpy as np
from src.env.pog_env import PogEnv, FACTION_AP, FACTION_CP


@pytest.fixture
def env():
    """
    Linear 5-node env for supply testing.
    Graph: 0 — 1 — 2 — 3 — 4
    Space 0 = AP supply source (FR, AP-controlled).
    Space 4 = CP supply source (GE, CP-controlled).
    """
    e = PogEnv.__new__(PogEnv)
    e._n_spaces = 5
    e._adj = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=bool)
    # AP controls 0,1,2; CP controls 3,4
    e._space_control = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    e._ap_supply_sources = {0}
    e._cp_supply_sources = {4}
    e._units = []
    return e


def test_supply_source_itself_in_supply(env):
    """The supply source space is always in supply."""
    assert env._check_oos(0, FACTION_AP) is False
    assert env._check_oos(4, FACTION_CP) is False


def test_connected_friendly_space_in_supply(env):
    """Space 2 (AP-controlled, path to source at 0 unblocked) → in supply."""
    assert env._check_oos(2, FACTION_AP) is False


def test_cp_connected_to_source_in_supply(env):
    """Space 3 (CP-controlled, adjacent to source at 4) → in supply."""
    assert env._check_oos(3, FACTION_CP) is False


def test_cut_off_space_is_oos(env):
    """AP space 1 cut off when CP controls everything between it and source 0."""
    env._space_control[0] = 1   # CP captures supply source
    # Space 1 is AP but no longer has a path to any AP supply source
    assert env._check_oos(1, FACTION_AP) is True


def test_cp_isolated_is_oos(env):
    """CP space 3 is OOS when AP captures source space 4."""
    env._space_control[4] = 0
    env._cp_supply_sources = set()   # source captured, no longer CP
    assert env._check_oos(3, FACTION_CP) is True


def test_supply_path_through_friendly_chain(env):
    """Supply traces through the full friendly chain 2→1→0."""
    # All AP-controlled, path exists
    assert env._check_oos(2, FACTION_AP) is False


def test_enemy_controlled_space_blocks_supply(env):
    """Enemy-controlled space in the path blocks supply."""
    env._space_control[1] = 1   # CP takes space 1, cutting AP supply to 2
    assert env._check_oos(2, FACTION_AP) is True


def test_neutral_space_does_not_extend_friendly_supply(env):
    """Neutral spaces (faction=2) do not extend supply lines."""
    env._space_control[1] = 2   # neutral
    # Space 2 can no longer trace supply through space 1 (neutral, not AP)
    assert env._check_oos(2, FACTION_AP) is True


def test_empty_supply_sources_is_oos(env):
    """If a faction has no supply sources, every space is OOS."""
    env._ap_supply_sources = set()
    assert env._check_oos(0, FACTION_AP) is True
    assert env._check_oos(1, FACTION_AP) is True
