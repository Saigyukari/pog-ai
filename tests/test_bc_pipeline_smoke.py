import os

import pytest

from src.data.rtt_parser import extract_training_records
from src.rl.bc_pipeline import train_bc
from src.rl.network import PoGNet, load_adjacency_matrix


SAMPLE_GAME = os.path.join(os.path.dirname(__file__), "..", "data", "176409.json")


@pytest.mark.skipif(not os.path.exists(SAMPLE_GAME), reason="sample game not present")
def test_train_bc_smoke_runs():
    records = extract_training_records(SAMPLE_GAME)[:16]
    adj = load_adjacency_matrix("pog_map_graph.json")
    model = PoGNet()

    params = train_bc(
        model=model,
        train_records=records[:12],
        val_records=records[12:16],
        adj=adj,
        n_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        rng_seed=0,
    )

    assert isinstance(params, dict)
