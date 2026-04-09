import json
import time
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from agentevolver.module.exp_manager.exp_manager import ExperienceManager
from agentevolver.utils.trajectory_io import (
    collect_trajectory_jsonl_files,
    load_trajectories_from_jsonl,
)


DONE_SUFFIX = ".summarized"


@hydra.main(config_path="../config", config_name="script_config", version_base=None)
def main(config):
    OmegaConf.resolve(config)
    run_init_exp_pool_from_trajectory(config)


def run_init_exp_pool_from_trajectory(config) -> None:
    exp_manager = ExperienceManager(config=config)

    trajectory_path = config.exp_manager.get("init_exp_trajectory_dir", None)
    if not trajectory_path:
        raise ValueError("exp_manager.init_exp_trajectory_dir must be set")

    trajectory_glob = config.exp_manager.get("init_exp_trajectory_glob", "*.jsonl")
    recursive = bool(config.exp_manager.get("init_exp_trajectory_recursive", True))
    trajectory_files = collect_trajectory_jsonl_files(
        trajectory_path,
        pattern=trajectory_glob,
        recursive=recursive,
    )
    if not trajectory_files:
        raise FileNotFoundError(
            f"No trajectory jsonl files found under {trajectory_path!r} with pattern {trajectory_glob!r}"
        )

    init_summary_request_timeout = config.exp_manager.get("init_exp_summary_request_timeout", None)
    if init_summary_request_timeout is not None:
        init_summary_request_timeout = float(init_summary_request_timeout)
    wait_indefinitely = (
        exp_manager.get_experience_pool_mode() == "state"
        and init_summary_request_timeout is None
    )
    max_concurrent_batches = int(config.exp_manager.get("init_exp_pool_max_workers", 1))

    processed_files = 0
    skipped_files = 0
    total_trajectories = 0
    started_at = time.time()

    print(
        "[InitExpPoolFromTrajectory] "
        f"found {len(trajectory_files)} files under {trajectory_path}"
    )

    for file_idx, trajectory_file in enumerate(trajectory_files, 1):
        done_marker = Path(f"{trajectory_file}{DONE_SUFFIX}")
        if done_marker.exists():
            skipped_files += 1
            print(
                "[InitExpPoolFromTrajectory] "
                f"skip {file_idx}/{len(trajectory_files)}: {trajectory_file} "
                f"(found {done_marker.name})"
            )
            continue

        trajectories = load_trajectories_from_jsonl(trajectory_file)
        total_trajectories += len(trajectories)
        print(
            "[InitExpPoolFromTrajectory] "
            f"summarize {file_idx}/{len(trajectory_files)}: {trajectory_file} "
            f"({len(trajectories)} trajectories)"
        )

        exp_manager.summarize_in_batch(
            trajectories,
            request_timeout=init_summary_request_timeout,
            wait_indefinitely=wait_indefinitely,
            max_concurrent_batches=max_concurrent_batches,
        )

        done_marker.write_text(
            json.dumps(
                {
                    "trajectory_file": str(trajectory_file),
                    "trajectory_count": len(trajectories),
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        processed_files += 1

    elapsed = time.time() - started_at
    print(
        "[InitExpPoolFromTrajectory] "
        f"done: processed_files={processed_files}, skipped_files={skipped_files}, "
        f"total_trajectories={total_trajectories}, elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
