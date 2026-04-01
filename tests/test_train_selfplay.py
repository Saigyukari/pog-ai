from train_selfplay import get_search_params


def test_get_search_params_schedule():
    assert get_search_params(0) == (6, 256, 1.0)
    assert get_search_params(1999) == (6, 256, 1.0)
    assert get_search_params(2000) == (5, 192, 1.0)
    assert get_search_params(9999) == (5, 192, 1.0)
    assert get_search_params(10000) == (4, 128, 0.5)
