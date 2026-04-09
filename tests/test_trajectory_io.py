from pathlib import Path

from agentevolver.schema.trajectory import Reward, Trajectory
from agentevolver.utils.trajectory_io import (
    collect_trajectory_jsonl_files,
    dump_trajectories_to_jsonl,
    load_trajectories_from_jsonl,
)


def test_dump_and_load_trajectories_roundtrip(tmp_path: Path):
    trajectories = [
        _build_trajectory(
            task_id="task_a",
            rollout_id="0",
            outcome=1.0,
            steps=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        ),
        _build_trajectory(
            task_id="task_b",
            rollout_id="1",
            outcome=0.0,
            steps=[
                {"role": "user", "content": "foo"},
                {"role": "assistant", "content": "bar"},
            ],
        ),
    ]

    output_path = tmp_path / "batch_00000.jsonl"
    dump_trajectories_to_jsonl(output_path, trajectories)

    loaded = load_trajectories_from_jsonl(output_path)

    assert [trajectory.task_id for trajectory in loaded] == ["task_a", "task_b"]
    assert [trajectory.rollout_id for trajectory in loaded] == ["0", "1"]
    assert [trajectory.reward.outcome for trajectory in loaded] == [1.0, 0.0]
    assert loaded[0].steps[0]["content"] == "hello"
    assert loaded[1].steps[-1]["content"] == "bar"
    assert loaded[0].metadata == {"source": "unit_test"}


def test_collect_trajectory_jsonl_files_supports_directory_and_glob(tmp_path: Path):
    dump_trajectories_to_jsonl(tmp_path / "a.jsonl", [_build_trajectory(task_id="task_a")])
    nested_dir = tmp_path / "nested"
    dump_trajectories_to_jsonl(nested_dir / "b.jsonl", [_build_trajectory(task_id="task_b")])

    directory_files = collect_trajectory_jsonl_files(tmp_path)
    glob_files = collect_trajectory_jsonl_files(str(tmp_path / "**" / "*.jsonl"))

    assert directory_files == [tmp_path / "a.jsonl", nested_dir / "b.jsonl"]
    assert glob_files == [tmp_path / "a.jsonl", nested_dir / "b.jsonl"]


def _build_trajectory(
    task_id: str = "task_default",
    rollout_id: str = "0",
    outcome: float = 1.0,
    steps: list[dict] | None = None,
) -> Trajectory:
    trajectory = Trajectory(
        data_id="0",
        rollout_id=rollout_id,
        query="demo query",
        steps=steps or [{"role": "user", "content": "demo"}],
        reward=Reward(outcome=outcome, success_rate=outcome, madness=0.0),
        metadata={"source": "unit_test"},
    )
    trajectory.task_id = task_id
    trajectory.instance_id = f"instance_{task_id}"
    return trajectory
