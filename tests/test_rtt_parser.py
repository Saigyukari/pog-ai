"""
tests/test_rtt_parser.py — Unit tests for RTT game record parser.
"""
import json
import os
import pytest
import numpy as np

from src.data.rtt_parser import (
    rtt_card_to_strid,
    build_card_lookup,
    group_replay_by_action,
    _extract_primary_action,
    extract_outcome,
    N_ACTIONS, N_SPACES, ACT_EVENT_START, ACT_OPS_START, ACT_MOVE_START,
)

# ── Card mapping tests ─────────────────────────────────────────────────────

def test_rtt_card_ap_range():
    assert rtt_card_to_strid(1)  == "AP-01"
    assert rtt_card_to_strid(3)  == "AP-03"
    assert rtt_card_to_strid(65) == "AP-65"


def test_rtt_card_cp_range():
    assert rtt_card_to_strid(66)  == "CP-01"
    assert rtt_card_to_strid(77)  == "CP-12"
    assert rtt_card_to_strid(130) == "CP-65"


def test_rtt_card_out_of_range():
    with pytest.raises(ValueError):
        rtt_card_to_strid(0)
    with pytest.raises(ValueError):
        rtt_card_to_strid(131)


def test_card_lookup_from_db():
    # Use real cards DB
    db_path = os.path.join(os.path.dirname(__file__), "..", "pog_cards_db.json")
    with open(db_path) as f:
        cards_db = json.load(f)
    lookup = build_card_lookup(cards_db)
    assert len(lookup) <= 110          # capped at N_CARDS
    assert "AP-01" in lookup
    assert lookup["AP-01"] == 0        # first card → index 0
    assert "CP-01" in lookup


# ── Replay grouping tests ──────────────────────────────────────────────────

def _make_replay(*groups):
    """Build a minimal replay from action-group tuples."""
    entries = []
    for group in groups:
        for entry in group:
            entries.append(entry)
        entries.append(["Allied Powers", "end_action"])
    return entries


def test_group_basic_split():
    replay = _make_replay(
        [["Allied Powers", "play_event", 3]],
        [["Central Powers", "play_ops", 66]],
    )
    groups = group_replay_by_action(replay)
    assert len(groups) == 2


def test_group_undo_rolls_back():
    replay = [
        ["Allied Powers", "play_rps", 5],
        ["Allied Powers", "undo"],
        ["Allied Powers", "play_ops", 3],
        ["Allied Powers", "end_action"],
    ]
    groups = group_replay_by_action(replay)
    assert len(groups) == 1
    # The play_rps should have been undone; only play_ops should remain
    actions = [e[1] for e in groups[0] if isinstance(e, list) and len(e) > 1]
    assert "play_ops" in actions
    assert "play_rps" not in actions


def test_group_setup_skipped():
    replay = [
        [None, ".setup", [123, "Historical", {}]],
        ["Central Powers", "play_event", 66],
        ["Central Powers", "end_action"],
    ]
    groups = group_replay_by_action(replay)
    assert len(groups) == 1


# ── Primary action extraction tests ───────────────────────────────────────

@pytest.fixture
def card_lookup():
    db_path = os.path.join(os.path.dirname(__file__), "..", "pog_cards_db.json")
    with open(db_path) as f:
        cards_db = json.load(f)
    return build_card_lookup(cards_db)


def test_extract_play_event(card_lookup):
    group = [
        ["Central Powers", "play_event", 66],  # CP-01
        ["Central Powers", "end_action"],
    ]
    flat, role = _extract_primary_action(group, card_lookup)
    cp01_idx = card_lookup["CP-01"]
    assert flat == ACT_EVENT_START + cp01_idx
    assert role == "Central Powers"


def test_extract_play_ops_move(card_lookup):
    group = [
        ["Allied Powers", "play_ops", 3],  # AP-03
        ["Allied Powers", "activate_move", 32],
        ["Allied Powers", "piece", 10],
        ["Allied Powers", "space", 15],
        ["Allied Powers", "end_action"],
    ]
    flat, role = _extract_primary_action(group, card_lookup)
    ap03_idx = card_lookup["AP-03"]
    assert flat == ACT_OPS_START + ap03_idx * 3 + 0  # MOVE op_type=0
    assert role == "Allied Powers"


def test_extract_play_ops_attack(card_lookup):
    group = [
        ["Allied Powers", "play_ops", 8],  # AP-08
        ["Allied Powers", "activate_attack", 125],
        ["Allied Powers", "end_action"],
    ]
    flat, _ = _extract_primary_action(group, card_lookup)
    ap08_idx = card_lookup["AP-08"]
    assert flat == ACT_OPS_START + ap08_idx * 3 + 1  # ATTACK op_type=1


def test_extract_play_sr(card_lookup):
    group = [
        ["Allied Powers", "play_sr", 14],  # AP-14
        ["Allied Powers", "end_action"],
    ]
    flat, _ = _extract_primary_action(group, card_lookup)
    ap14_idx = card_lookup["AP-14"]
    assert flat == ACT_OPS_START + ap14_idx * 3 + 2  # SR op_type=2


def test_extract_next_is_pass(card_lookup):
    group = [
        ["Central Powers", "next"],
        ["Central Powers", "end_action"],
    ]
    flat, _ = _extract_primary_action(group, card_lookup)
    assert flat == 0


def test_next_before_event_yields_event(card_lookup):
    """next appearing before play_event must not shadow the event."""
    group = [
        ["Central Powers", "next"],
        ["Central Powers", "play_event", 66],
        ["Central Powers", "end_action"],
    ]
    flat, _ = _extract_primary_action(group, card_lookup)
    cp01_idx = card_lookup["CP-01"]
    assert flat == ACT_EVENT_START + cp01_idx


# ── Outcome extraction tests ───────────────────────────────────────────────

def test_outcome_ap_wins():
    replay = [["Central Powers", ".resign", "Allied Powers"]]
    assert extract_outcome(replay) == 1


def test_outcome_cp_wins():
    replay = [["Allied Powers", ".resign", "Central Powers"]]
    assert extract_outcome(replay) == -1


def test_outcome_no_resign():
    replay = [["Allied Powers", "play_ops", 3]]
    assert extract_outcome(replay) == 0


# ── Integration test against sample game ──────────────────────────────────

SAMPLE_GAME = os.path.join(os.path.dirname(__file__), "..", "data", "176409.json")

@pytest.mark.skipif(not os.path.exists(SAMPLE_GAME), reason="sample game not present")
def test_full_parse_sample_game():
    from src.data.rtt_parser import convert_rtt_directory
    import tempfile, shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(SAMPLE_GAME, os.path.join(tmpdir, "176409.json"))
        out = os.path.join(tmpdir, "out.jsonl")
        n = convert_rtt_directory(tmpdir, out)
        assert n >= 100, f"Expected ≥100 records, got {n}"
        with open(out) as f:
            records = [json.loads(l) for l in f]

    for r in records:
        assert np.array(r["obs_tensor"]).shape    == (32, 72)
        assert np.array(r["card_context"]).shape  == (7, 16)
        assert np.array(r["legal_mask"]).shape    == (5341,)
        assert 0 <= r["action_taken"] < N_ACTIONS
        assert r["outcome"] in (-1, 0, 1)

    # Outcome: AP wins game 176409
    assert records[0]["outcome"] in (1, -1)

    # Action distribution sanity: must have OPS and EVENT records
    from collections import Counter
    atype = Counter()
    for r in records:
        a = r["action_taken"]
        if   a == 0:   atype["PASS"]  += 1
        elif a < 111:  atype["EVENT"] += 1
        elif a < 441:  atype["OPS"]   += 1
        else:          atype["MOVE"]  += 1
    assert atype["OPS"]   > 50
    assert atype["EVENT"] > 10
