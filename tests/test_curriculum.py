"""Tests for training/curriculum.py's step->shard schedule builder."""
import yaml
from training.curriculum import build_phase_schedule, shard_for_step


def test_phase_schedule_covers_all_steps(tmp_path):
    curriculum = {"phases": [
        {"name": "a", "mixture": "m1", "token_fraction": 0.5},
        {"name": "b", "mixture": "m2", "token_fraction": 0.5},
    ]}
    p = tmp_path / "curriculum.yaml"
    p.write_text(yaml.dump(curriculum))
    schedule = build_phase_schedule(str(p), {"m1": "shard1.npy", "m2": "shard2.npy"}, total_steps=100)
    assert schedule[0] == (0, 50, "shard1.npy", "a")
    assert schedule[-1][1] == 100  # always reaches total_steps exactly
    for step in [0, 25, 49, 50, 75, 99]:
        shard, name = shard_for_step(schedule, step)
        assert shard in ("shard1.npy", "shard2.npy")


def test_phase_schedule_handles_rounding(tmp_path):
    curriculum = {"phases": [
        {"name": "a", "mixture": "m1", "token_fraction": 0.33},
        {"name": "b", "mixture": "m2", "token_fraction": 0.33},
        {"name": "c", "mixture": "m1", "token_fraction": 0.34},
    ]}
    p = tmp_path / "c2.yaml"
    p.write_text(yaml.dump(curriculum))
    schedule = build_phase_schedule(str(p), {"m1": "s1", "m2": "s2"}, total_steps=99)
    assert schedule[-1][1] == 99
    covered = sum(e - s for s, e, _, _ in schedule)
    assert covered == 99
