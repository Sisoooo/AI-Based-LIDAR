#!/usr/bin/env python3

"""
Consolidation script for custom ROS 2 RViz datasets.

Same output structure as the original version, but videos are COPIED directly
instead of being decoded and re-encoded — much faster and lossless.

How it works:
  1. Uses LeRobotDataset.create (without video) to write parquet + metadata
  2. Reads state/action/task directly from source parquet files (no video decode)
  3. After finalize(), copies MP4 files directly to the correct output paths
  4. Patches info.json to re-add the video feature entry

Usage:
    python3 merge_rviz_dataset.py --data_dir ~/lerobot_ros2_rviz_dataset
"""

import json
import shutil
import re
from pathlib import Path
import numpy as np
import pandas as pd
import tyro

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets import LeRobotDataset

ROBOT_TYPE = "ros2_mobile_robot"
FPS = 30
WIDTH = 1920
HEIGHT = 1200
CHUNKS_SIZE = 1000
VIDEO_KEY = "observation.images.rviz"


def natural_sort_key(s: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def read_task_from_episode(ep_dir: Path) -> str:
    tasks_path = ep_dir / "meta" / "tasks.parquet"
    if tasks_path.exists():
        df = pd.read_parquet(tasks_path)
        # The task string is stored as the DataFrame index; task_index is a column.
        if len(df) > 0:
            return str(df.index[0])
    return "navigate the environment"


def main(
    data_dir: str,
    *,
    push_to_hub: bool = False,
    hf_repo_id: str = "your_hf_username/ros2-rviz-unified-navigation",
    output: str = str(Path.home() / "ros2_rviz_dataset_consolidated"),
):
    input_base_dir = Path(data_dir).expanduser()
    output_path = Path(output).expanduser()

    if output_path.exists():
        print(f"Removing existing output: {output_path}")
        shutil.rmtree(output_path)

    # Discover episodes
    episode_dirs = sorted(
        [d for d in input_base_dir.iterdir()
         if d.is_dir() and "Images_" in d.name and "_episode_" in d.name],
        key=natural_sort_key,
    )
    if not episode_dirs:
        print(f"No episode directories found in {input_base_dir}.")
        return
    print(f"Found {len(episode_dirs)} episodes to merge.")

    # Create dataset WITHOUT video feature — we'll copy videos and patch metadata after
    features_no_video = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": ["odom_x", "odom_y", "odom_yaw", 
                      "odom_linear_x", "odom_linear_y", "odom_angular_z", 
                      "prev_cmd_linear_x", "prev_cmd_linear_y", "prev_cmd_angular_z"],
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["cmd_vel_linear_x", "cmd_vel_linear_y", "cmd_vel_angular_z"],
        },
    }

    print(f"Creating dataset at: {output_path}")
    consolidated_dataset = LeRobotDataset.create(
        repo_id=hf_repo_id,
        root=output_path,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features=features_no_video,
        use_videos=False,
    )

    # Pass 1: write parquet + metadata (no video decode)
    successful_episode_dirs = []
    for ep_idx, ep_dir in enumerate(episode_dirs):
        print(f"[{ep_idx + 1}/{len(episode_dirs)}] {ep_dir.name}")
        try:
            parquet_path = ep_dir / "data" / "chunk-000" / "file-000.parquet"
            df = pd.read_parquet(parquet_path)
            task = read_task_from_episode(ep_dir)

            for _, row in df.iterrows():
                consolidated_dataset.add_frame({
                    "observation.state": np.array(row["observation.state"], dtype=np.float32),
                    "action": np.array(row["action"], dtype=np.float32),
                    "task": task,
                })

            consolidated_dataset.save_episode()
            successful_episode_dirs.append(ep_dir)

        except Exception as e:
            print(f"  ERROR: {e}. Skipping.")
            try:
                consolidated_dataset.clear_episode_buffer()
            except Exception:
                pass

    consolidated_dataset.finalize()

    # Pass 2: copy videos directly to output (no re-encoding)
    print("\nCopying videos...")
    missing = 0
    for ep_idx, ep_dir in enumerate(successful_episode_dirs):
        src_video = ep_dir / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4"
        chunk_idx = ep_idx // CHUNKS_SIZE
        file_idx = ep_idx % CHUNKS_SIZE
        dst_dir = output_path / "videos" / VIDEO_KEY / f"chunk-{chunk_idx:03d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_video = dst_dir / f"file-{file_idx:03d}.mp4"
        if src_video.exists():
            shutil.copy2(src_video, dst_video)
        else:
            print(f"  WARNING: missing video for {ep_dir.name}")
            missing += 1

    # Pass 3: patch info.json to add video feature back
    info_path = output_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["features"][VIDEO_KEY] = {
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
    }
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    total = len(successful_episode_dirs)
    print(f"\nDone. {total - missing}/{total} episodes merged to: {output_path}")
    if missing:
        print(f"WARNING: {missing} videos were missing (parquet data was still merged).")

    if push_to_hub:
        print(f"Uploading to HuggingFace Hub: {hf_repo_id}")
        consolidated_dataset.push_to_hub(
            tags=["ros2", "rviz", "navigation", "pi0"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)