"""
Cold-start self-play smoke test: run 1 iteration with n_actors=2, batch=16.
Passes if the script completes and writes a checkpoint.
"""

from pathlib import Path

from train_selfplay import main


def test_selfplay_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_selfplay.py",
            "--n-actors",
            "2",
            "--buffer-capacity",
            "512",
            "--min-buffer-size",
            "16",
            "--batch-size",
            "16",
            "--mcts-sims",
            "2",
            "--depth-limit",
            "1",
            "--learner-steps",
            "1",
            "--max-steps",
            "8",
            "--iterations",
            "1",
            "--checkpoint-dir",
            str(tmp_path),
        ],
    )
    assert main() == 0
    assert (tmp_path / "iter_001.pkl").is_file()
