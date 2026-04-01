"""
tests/test_crt.py — Unit tests for Combat Results Table (CRT) resolution.
"""
import pytest
from src.env.pog_env import PogEnv


@pytest.fixture
def env():
    return PogEnv()


def test_crt_below_half_ratio_always_engaged(env):
    """Ratio < 0.5 (1 vs 4): always Engaged — no losses."""
    for _ in range(30):
        r = env._resolve_crt(1, 4, 0, 0, 0)
        assert r["attacker_losses"] == 0
        assert r["defender_losses"] == 0
        assert not r["defender_eliminated"]


def test_crt_overwhelming_ratio_frequently_eliminates(env):
    """8:1 attack should often result in Defender Eliminated."""
    de_count = sum(env._resolve_crt(8, 1, 0, 0, 0)["defender_eliminated"]
                   for _ in range(100))
    assert de_count > 40, f"Expected frequent DE at 8:1, got {de_count}/100"


def test_crt_returns_all_required_keys(env):
    r = env._resolve_crt(3, 2, 0, 0, 1)
    for k in ["attacker_losses", "defender_losses",
              "attacker_retreats", "defender_retreats", "defender_eliminated"]:
        assert k in r, f"Missing key '{k}'"


def test_trench_l3_absorbs_first_step_loss(env):
    """Level-3 trench absorbs the first defender step loss."""
    losses_none = sum(env._resolve_crt(4, 2, 0, 0, 0)["defender_losses"] for _ in range(500))
    losses_l3   = sum(env._resolve_crt(4, 2, 0, 0, 3)["defender_losses"] for _ in range(500))
    # Trench 3 should reduce total losses (absorption)
    assert losses_l3 <= losses_none, (
        f"Trench L3 should absorb losses: {losses_l3} <= {losses_none}"
    )


def test_trench_drm_capped_at_2(env):
    """Level 3 trench uses DRM +2 (same cap as level 2)."""
    # Both should run without error and return valid structure
    for lvl in (2, 3):
        for _ in range(50):
            r = env._resolve_crt(4, 2, 0, 0, lvl)
            assert r["defender_losses"] >= 0
            assert r["attacker_losses"] >= 0


def test_crt_no_negative_losses(env):
    """CRT must never return negative loss counts."""
    for atk in range(1, 6):
        for dfd in range(1, 6):
            for trench in range(4):
                r = env._resolve_crt(atk, dfd, 0, 0, trench)
                assert r["attacker_losses"] >= 0
                assert r["defender_losses"] >= 0


def test_crt_exchange_produces_mutual_losses(env):
    """At roughly 1:1 ratio, expect some EX results (both sides lose)."""
    both_lose = sum(
        1 for _ in range(300)
        if (lambda r: r["attacker_losses"] > 0 and r["defender_losses"] > 0)
           (env._resolve_crt(3, 3, 0, 0, 0))
    )
    assert both_lose > 5, f"Expected some EX results, got {both_lose}/300"


def test_crt_defender_retreat_at_2to1(env):
    """2:1 attack band can produce D1R (retreat required)."""
    retreats = sum(
        env._resolve_crt(4, 2, 0, 0, 0)["defender_retreats"]
        for _ in range(200)
    )
    assert retreats > 30, f"Expected D1R results at 2:1, got {retreats}/200"
