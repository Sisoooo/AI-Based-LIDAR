#!/usr/bin/env python3

"""
Consolidation script for custom ROS 2 RViz datasets.

Outputs LeRobot dataset v2.1 format, which is required by pi0.5 and other
VLA models. The installed lerobot 0.4.x library writes v3.0 internally, so
this script bypasses LeRobotDataset entirely and constructs the v2.1 folder
structure from scratch.

v2.1 output layout:
  meta/info.json          (codebase_version: "v2.1")
  meta/stats.json
  meta/tasks.jsonl        (one JSON object per line)
  meta/episodes.jsonl     (one JSON object per line)
  data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet
  videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4

Source episodes are the per-episode directories recorded by leRobotDatasetRecorder
(lerobot v3.0 format). Their raw parquet and MP4 files are read directly without
any re-encoding.

Usage:
    python3 merge_rviz_dataset.py --data_dir ~/lerobot_ros2_rviz_dataset
"""

import json
import math
import shutil
import re
from pathlib import Path
import numpy as np
import pandas as pd
import tyro

ROBOT_TYPE = "ros2_mobile_robot"
FPS = 30
WIDTH = 1920
HEIGHT = 1200
CHUNKS_SIZE = 1000
VIDEO_KEY = "observation.images.rviz"
CODEBASE_VERSION = "v2.1"


def natural_sort_key(s: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def read_task_from_episode(ep_dir: Path) -> str:
    """Read the task string from a v3.0 per-episode directory."""
    tasks_path = ep_dir / "meta" / "tasks.parquet"
    if tasks_path.exists():
        df = pd.read_parquet(tasks_path)
        # In v3.0 the task string is the DataFrame index.
        if len(df) > 0:
            return str(df.index[0])
    return "navigate the environment"


def compute_stats(all_states: np.ndarray, all_actions: np.ndarray) -> dict:
    """Compute per-feature statistics used for normalization."""
    def _feature_stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "count": [int(len(arr))],
        }
    return {
        "observation.state": _feature_stats(all_states),
        "action": _feature_stats(all_actions),
    }


def main(
    data_dir: str,
    *,
    output: str = str(Path.home() / "ros2_rviz_dataset_v21"),
):
    input_base_dir = Path(data_dir).expanduser()
    output_path = Path(output).expanduser()

    if output_path.exists():
        print(f"Removing existing output: {output_path}")
        shutil.rmtree(output_path)

    # Discover source episodes
    episode_dirs = sorted(
        [d for d in input_base_dir.iterdir()
         if d.is_dir() and "Images_" in d.name and "_episode_" in d.name],
        key=natural_sort_key,
    )
    if not episode_dirs:
        print(f"No episode directories found in {input_base_dir}.")
        return
    print(f"Found {len(episode_dirs)} episodes to merge.")

    (output_path / "meta").mkdir(parents=True)
    (output_path / "data").mkdir()

    episodes_meta = []        # list of dicts for episodes.jsonl
    tasks_index: dict = {}    # task string -> task_index
    all_states: list = []
    all_actions: list = []
    successful_episode_dirs: list = []
    global_frame_offset = 0

    # ------------------------------------------------------------------
    # Pass 1: write per-episode parquet files and collect metadata
    # ------------------------------------------------------------------
    for src_ep_idx, ep_dir in enumerate(episode_dirs):
        out_ep_idx = len(successful_episode_dirs)
        print(f"[{src_ep_idx + 1}/{len(episode_dirs)}] {ep_dir.name}")
        try:
            src_parquet = ep_dir / "data" / "chunk-000" / "file-000.parquet"
            df = pd.read_parquet(src_parquet)
            task = read_task_from_episode(ep_dir)

            if task not in tasks_index:
                tasks_index[task] = len(tasks_index)
            task_idx = tasks_index[task]

            n_frames = len(df)
            states = np.stack(df["observation.state"].tolist()).astype(np.float32)
            actions = np.stack(df["action"].tolist()).astype(np.float32)

            # Build v2.1 frame-level parquet
            out_df = pd.DataFrame({
                "observation.state": [s for s in states],
                "action": [a for a in actions],
                "timestamp": (np.arange(n_frames, dtype=np.float32) / FPS).tolist(),
                "frame_index": np.arange(n_frames, dtype=np.int64).tolist(),
                "episode_index": np.full(n_frames, out_ep_idx, dtype=np.int64).tolist(),
                "index": np.arange(
                    global_frame_offset,
                    global_frame_offset + n_frames,
                    dtype=np.int64,
                ).tolist(),
                "task_index": np.full(n_frames, task_idx, dtype=np.int64).tolist(),
            })

            chunk_idx = out_ep_idx // CHUNKS_SIZE
            dst_data_dir = output_path / "data" / f"chunk-{chunk_idx:03d}"
            dst_data_dir.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(
                dst_data_dir / f"episode_{out_ep_idx:06d}.parquet",
                index=False,
            )

            all_states.append(states)
            all_actions.append(actions)
            episodes_meta.append({
                "episode_index": out_ep_idx,
                "tasks": [task],
                "length": n_frames,
            })
            successful_episode_dirs.append(ep_dir)
            global_frame_offset += n_frames

        except Exception as e:
            print(f"  ERROR: {e}. Skipping.")

    total_episodes = len(successful_episode_dirs)
    total_frames = global_frame_offset
    print(f"\nWritten {total_episodes} episodes, {total_frames} frames.")

    # ------------------------------------------------------------------
    # Pass 2: copy MP4 files directly (no re-encoding)
    # ------------------------------------------------------------------
    print("Copying videos...")
    missing = 0
    for ep_idx, ep_dir in enumerate(successful_episode_dirs):
        src_video = ep_dir / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4"
        chunk_idx = ep_idx // CHUNKS_SIZE
        dst_video_dir = output_path / "videos" / VIDEO_KEY / f"chunk-{chunk_idx:03d}"
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        dst_video = dst_video_dir / f"episode_{ep_idx:06d}.mp4"
        if src_video.exists():
            shutil.copy2(src_video, dst_video)
        else:
            print(f"  WARNING: missing video for {ep_dir.name}")
            missing += 1

    # ------------------------------------------------------------------
    # Pass 3: write v2.1 metadata files
    # ------------------------------------------------------------------

    # meta/tasks.jsonl
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        for task, idx in sorted(tasks_index.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # meta/episodes.jsonl
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # meta/stats.json
    all_states_arr = np.concatenate(all_states, axis=0)
    all_actions_arr = np.concatenate(all_actions, axis=0)
    stats = compute_stats(all_states_arr, all_actions_arr)
    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # meta/info.json  (v2.1 schema)
    total_chunks = math.ceil(total_episodes / CHUNKS_SIZE)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_index),
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [9],
                "names": [
                    "odom_x", "odom_y", "odom_yaw",
                    "odom_linear_x", "odom_linear_y", "odom_angular_z",
                    "prev_cmd_linear_x", "prev_cmd_linear_y", "prev_cmd_angular_z",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [3],
                "names": ["cmd_vel_linear_x", "cmd_vel_linear_y", "cmd_vel_angular_z"],
            },
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [HEIGHT, WIDTH, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(FPS),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nDone. {total_episodes - missing}/{total_episodes} episodes merged to: {output_path}")
    if missing:
        print(f"WARNING: {missing} videos were missing (parquet data was still merged).")


if __name__ == "__main__":
    tyro.cli(main)