"""Real curriculum-learning integration (section 11): builds a step->shard
schedule from configs/curriculum.yaml so train.py can switch the active
training shard mid-run according to configured phases."""
from pathlib import Path

import yaml


def build_phase_schedule(curriculum_path: str, mixture_shards: dict, total_steps: int):
    """Returns a list of (start_step, end_step, shard_path, phase_name)."""
    with open(curriculum_path) as f:
        ccfg = yaml.safe_load(f)

    schedule = []
    cursor = 0
    for phase in ccfg["phases"]:
        n_steps = max(1, round(phase["token_fraction"] * total_steps))
        end = min(cursor + n_steps, total_steps)
        shard = mixture_shards[phase["mixture"]]
        schedule.append((cursor, end, shard, phase["name"]))
        cursor = end
        if cursor >= total_steps:
            break
    # Make sure the schedule always reaches total_steps (rounding safety)
    if schedule and schedule[-1][1] < total_steps:
        s, e, shard, name = schedule[-1]
        schedule[-1] = (s, total_steps, shard, name)
    return schedule


def shard_for_step(schedule, step):
    for start, end, shard, name in schedule:
        if start <= step < end or (step == end == schedule[-1][1]):
            return shard, name
    return schedule[-1][2], schedule[-1][3]
