from src.env.pog_env import (
    ACT_EVENT_START,
    ACT_OPS_START,
    ACT_MOVE_START,
    FACTION_AP,
    OFFBOARD,
    PogEnv,
)


def _make_env():
    env = PogEnv("pog_map_graph.json", "pog_cards_db.json")
    env.reset(seed=0)
    return env


def test_ops_card_goes_to_played_pile_not_void():
    env = _make_env()
    mask = env.action_mask("AP")
    ops_actions = [i for i in range(ACT_OPS_START, ACT_MOVE_START) if mask[i]]
    assert ops_actions, "expected at least one legal AP ops action at reset"

    played_before = len(env._ap_played)
    hand_before = list(env._ap_hand)
    action = ops_actions[0]
    card_idx = (action - ACT_OPS_START) // 3
    assert card_idx in hand_before

    env.step(action)

    assert len(env._ap_played) == played_before + 1
    assert env._ap_played[-1] == card_idx


def test_reinforcement_event_brings_unit_on_map():
    env = _make_env()
    offboard_before = sum(1 for u in env._units if u["faction"] == FACTION_AP and u["location"] == OFFBOARD)
    reinf_idx = next(
        i for i, c in enumerate(env._cards_db)
        if c["faction"] == FACTION_AP and "reinforcement" in c["name"].lower()
    )
    env._ap_hand = [reinf_idx]
    env.active_player = FACTION_AP
    action = ACT_EVENT_START + reinf_idx
    assert env.action_mask("AP")[action]

    env.step(action)

    offboard_after = sum(1 for u in env._units if u["faction"] == FACTION_AP and u["location"] == OFFBOARD)
    assert offboard_after < offboard_before


def test_sr_ops_brings_unit_on_map():
    env = _make_env()
    offboard_before = sum(1 for u in env._units if u["faction"] == FACTION_AP and u["location"] == OFFBOARD)
    ap_card_idx = next(i for i, c in enumerate(env._cards_db) if c["faction"] == FACTION_AP)
    env._ap_hand = [ap_card_idx]
    env.active_player = FACTION_AP
    action = ACT_OPS_START + ap_card_idx * 3 + 2
    assert env.action_mask("AP")[action]

    env.step(action)

    offboard_after = sum(1 for u in env._units if u["faction"] == FACTION_AP and u["location"] == OFFBOARD)
    assert offboard_after < offboard_before
