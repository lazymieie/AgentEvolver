import glob
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

from agentevolver.schema.trajectory import Reward, Trajectory


TRAJECTORY_RECORD_VERSION = 1


def _reward_to_dict(reward) -> dict | None:
    if reward is None:
        return None
    if isinstance(reward, dict):
        return reward
    if hasattr(reward, "model_dump"):
        return reward.model_dump()
    return {
        "outcome": getattr(reward, "outcome", 0.0),
        "success_rate": getattr(reward, "success_rate", 0.0),
        "madness": getattr(reward, "madness", 0.0),
        "description": getattr(
            reward,
            "description",
            "Outcome 1 denotes success, and 0 denotes failure.",
        ),
        "metadata": getattr(reward, "metadata", {}) or {},
    }


def trajectory_to_dict(trajectory: Trajectory) -> dict:
    return {
        "schema_version": TRAJECTORY_RECORD_VERSION,
        "data_id": str(getattr(trajectory, "data_id", "")),
        "rollout_id": str(getattr(trajectory, "rollout_id", "")),
        "task_id": str(getattr(trajectory, "task_id", "")),
        "instance_id": str(getattr(trajectory, "instance_id", "")),
        "query": getattr(trajectory, "query", "") or "",
        "is_terminated": bool(getattr(trajectory, "is_terminated", False)),
        "reward": _reward_to_dict(getattr(trajectory, "reward", None)),
        "metadata": getattr(trajectory, "metadata", {}) or {},
        "steps": list(getattr(trajectory, "steps", []) or []),
    }


def trajectory_from_dict(record: dict) -> Trajectory:
    reward_payload = record.get("reward")
    reward = Reward(**reward_payload) if reward_payload else None
    trajectory = Trajectory(
        data_id=str(record.get("data_id", "")),
        rollout_id=str(record.get("rollout_id", "")),
        steps=list(record.get("steps", []) or []),
        query=record.get("query", "") or "",
        is_terminated=bool(record.get("is_terminated", False)),
        reward=reward,
        metadata=record.get("metadata", {}) or {},
    )
    trajectory.task_id = str(record.get("task_id", ""))
    trajectory.instance_id = str(record.get("instance_id", ""))
    return trajectory


def dump_trajectories_to_jsonl(path: str | os.PathLike[str], trajectories: Iterable[Trajectory]) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target_path.parent),
        delete=False,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        for trajectory in trajectories:
            tmp_file.write(json.dumps(trajectory_to_dict(trajectory), ensure_ascii=False))
            tmp_file.write("\n")

    os.replace(tmp_path, target_path)
    return target_path


def load_trajectories_from_jsonl(path: str | os.PathLike[str]) -> List[Trajectory]:
    trajectories: List[Trajectory] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, 1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(record).__name__}")
            trajectories.append(trajectory_from_dict(record))
    return trajectories


def collect_trajectory_jsonl_files(
    path_or_pattern: str | os.PathLike[str],
    pattern: str = "*.jsonl",
    recursive: bool = True,
) -> List[Path]:
    target = Path(path_or_pattern)
    if target.exists():
        if target.is_file():
            return [target]
        iterator = target.rglob(pattern) if recursive else target.glob(pattern)
        return sorted(path for path in iterator if path.is_file())

    matches = sorted(Path(path) for path in glob.glob(str(path_or_pattern), recursive=recursive))
    return [path for path in matches if path.is_file()]
