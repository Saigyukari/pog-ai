from src.env.pog_env import PogEnv


def test_pog_env_refills_hands_on_turn_boundary():
    env = PogEnv("pog_map_graph.json", "pog_cards_db.json")
    env.reset(seed=0)

    env.action_round = 7
    env._ap_hand = []
    env._cp_hand = []
    ap_before = len(env._ap_deck)
    cp_before = len(env._cp_deck)

    env._advance_turn()

    assert env.turn == 2
    assert env.action_round == 1
    assert len(env._ap_hand) > 0
    assert len(env._cp_hand) > 0
    assert len(env._ap_deck) < ap_before
    assert len(env._cp_deck) < cp_before
