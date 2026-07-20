#!/usr/bin/env python3
"""
Controller-agnostic evaluation logger for MiR250 navigation experiments.

Runs alongside EITHER Nav2 (mir_random_nav.py) OR the pi0.5 inference stack
(goal_monitor_node.py + inference_ros2_node.py) — it only observes topics that
both pipelines already publish, so no changes to those nodes are required.

For every episode (delimited by a new message on /episode_goal) it records:
    - success / timeout / aborted (new goal arrived before this one finished)
    - time to goal (s)
    - path length actually driven (m) vs straight-line distance (m) -> path efficiency
    - final distance to goal (m) at episode end
    - command smoothness: mean |angular_z| and angular_z std-dev over the episode
    - full (t, x, y) trajectory, saved to its own CSV for later overlay plots

Output (under --output_dir, default ~/nav_eval_logs):
    eval_metrics.csv                         one row per episode (append mode)
    trajectories/{controller}_{map}_{episode_id:04d}.csv   per-episode (t,x,y)

Usage (example, run in a spare terminal while Nav2 or pi0.5 is driving):
    python3 eval_logger_node.py --ros-args \\
        -p controller_name:=nav2 -p map_name:=maze

    python3 eval_logger_node.py --ros-args \\
        -p controller_name:=pi05 -p map_name:=maze \\
        -p cmd_vel_topic:=/diff_cont/cmd_vel_unstamped
"""

import csv
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped


def _distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class EvalLogger(Node):
    def __init__(self):
        super().__init__("mir_eval_logger")

        self.declare_parameter("odom_topic", "/diff_cont/odom")
        self.declare_parameter("goal_topic", "/episode_goal")
        self.declare_parameter("cmd_vel_topic", "/diff_cont/cmd_vel_unstamped")
        self.declare_parameter("cmd_vel_stamped", False)
        self.declare_parameter("threshold_m", 0.50)
        self.declare_parameter("timeout_s", 300.0)
        self.declare_parameter("check_hz", 10.0)
        self.declare_parameter("trajectory_hz", 5.0)
        self.declare_parameter("controller_name", "nav2")   # label, e.g. nav2 / pi05
        self.declare_parameter("map_name", "maze")
        self.declare_parameter("output_dir", str(Path.home() / "nav_eval_logs"))

        gp = lambda n: self.get_parameter(n).get_parameter_value()
        odom_topic          = gp("odom_topic").string_value
        goal_topic           = gp("goal_topic").string_value
        cmd_vel_topic        = gp("cmd_vel_topic").string_value
        cmd_vel_stamped      = gp("cmd_vel_stamped").bool_value
        self.threshold       = float(gp("threshold_m").double_value)
        self.timeout_s       = float(gp("timeout_s").double_value)
        check_hz             = float(gp("check_hz").double_value)
        self.trajectory_dt   = 1.0 / float(gp("trajectory_hz").double_value)
        self.controller_name = gp("controller_name").string_value
        self.map_name        = gp("map_name").string_value

        self.output_dir = Path(gp("output_dir").string_value).expanduser()
        self.traj_dir = self.output_dir / "trajectories"
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "eval_metrics.csv"
        self._ensure_csv_header()

        self.latest_odom: Optional[Odometry] = None
        self.current_goal: Optional[PoseStamped] = None
        self.episode_id = self._next_episode_id()
        self.episode_active = False
        self.episode_start_time = 0.0
        self.start_pos = (0.0, 0.0)
        self.last_traj_t = 0.0
        self.trajectory: list = []          # [(t, x, y), ...]
        self.ang_vels: list = []            # angular_z samples this episode
        self.lin_vels: list = []            # linear_x samples this episode

        self.create_subscription(Odometry, odom_topic, self._odom_cb, 50)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb, 10)
        cmd_type = TwistStamped if cmd_vel_stamped else Twist
        self.create_subscription(cmd_type, cmd_vel_topic, self._cmd_cb, 50)
        self._cmd_vel_stamped = cmd_vel_stamped

        self.create_timer(1.0 / check_hz, self._check_episode)

        self.get_logger().info(
            f"Eval logger ready | controller={self.controller_name} map={self.map_name} "
            f"odom={odom_topic} goal={goal_topic} cmd_vel={cmd_vel_topic} "
            f"threshold={self.threshold:.2f} m timeout={self.timeout_s:.0f} s "
            f"-> logging to {self.csv_path}"
        )

    # ------------------------------------------------------------------ #
    def _ensure_csv_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode_id", "timestamp", "controller", "map", "result", "success",
                    "time_s", "path_length_m", "straight_dist_m", "path_efficiency",
                    "final_dist_m", "mean_abs_ang_vel", "ang_vel_std",
                    "mean_lin_vel", "start_x", "start_y", "goal_x", "goal_y",
                    "trajectory_file",
                ])

    def _next_episode_id(self) -> int:
        """Resume numbering across runs so files don't get overwritten."""
        if not self.csv_path.exists():
            return 0
        with open(self.csv_path) as f:
            n = sum(1 for _ in f) - 1  # minus header
        return max(0, n)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    def _cmd_cb(self, msg):
        t = msg.twist if self._cmd_vel_stamped else msg
        self.ang_vels.append(t.angular.z)
        self.lin_vels.append(t.linear.x)

    def _goal_cb(self, msg: PoseStamped):
        if self.episode_active:
            self._finalize_episode(result="aborted")
        self.current_goal = msg
        self._start_episode()

    def _start_episode(self):
        if self.latest_odom is None:
            self.get_logger().warn("New goal received but no odom yet; waiting...")
        x = self.latest_odom.pose.pose.position.x if self.latest_odom else 0.0
        y = self.latest_odom.pose.pose.position.y if self.latest_odom else 0.0
        self.start_pos = (x, y)
        self.episode_start_time = time.time()
        self.last_traj_t = 0.0
        self.trajectory = [(0.0, x, y)]
        self.ang_vels = []
        self.lin_vels = []
        self.episode_active = True
        self.get_logger().info(
            f"[{self.controller_name}/{self.map_name}] Episode {self.episode_id} started "
            f"goal=({self.current_goal.pose.position.x:.2f},{self.current_goal.pose.position.y:.2f})"
        )

    # ------------------------------------------------------------------ #
    # Periodic check
    # ------------------------------------------------------------------ #
    def _check_episode(self):
        if not self.episode_active or self.latest_odom is None or self.current_goal is None:
            return

        rx = self.latest_odom.pose.pose.position.x
        ry = self.latest_odom.pose.pose.position.y
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        dist = _distance_2d(rx, ry, gx, gy)
        elapsed = time.time() - self.episode_start_time

        if elapsed - self.last_traj_t >= self.trajectory_dt:
            self.trajectory.append((elapsed, rx, ry))
            self.last_traj_t = elapsed

        if dist < self.threshold:
            self._finalize_episode(result="success")
        elif elapsed > self.timeout_s:
            self._finalize_episode(result="timeout")

    # ------------------------------------------------------------------ #
    # Episode finalization
    # ------------------------------------------------------------------ #
    def _finalize_episode(self, result: str):
        self.episode_active = False
        rx, ry = (self.latest_odom.pose.pose.position.x, self.latest_odom.pose.pose.position.y) \
            if self.latest_odom else self.start_pos
        gx, gy = (self.current_goal.pose.position.x, self.current_goal.pose.position.y) \
            if self.current_goal else self.start_pos

        elapsed = time.time() - self.episode_start_time
        path_length = sum(
            _distance_2d(self.trajectory[i][1], self.trajectory[i][2],
                         self.trajectory[i + 1][1], self.trajectory[i + 1][2])
            for i in range(len(self.trajectory) - 1)
        )
        straight_dist = _distance_2d(self.start_pos[0], self.start_pos[1], gx, gy)
        path_efficiency = (straight_dist / path_length) if path_length > 1e-6 else 0.0
        final_dist = _distance_2d(rx, ry, gx, gy)
        mean_abs_ang = float(np.mean(np.abs(self.ang_vels))) if self.ang_vels else 0.0
        ang_std = float(np.std(self.ang_vels)) if self.ang_vels else 0.0
        mean_lin = float(np.mean(self.lin_vels)) if self.lin_vels else 0.0

        traj_file = self.traj_dir / f"{self.controller_name}_{self.map_name}_{self.episode_id:04d}.csv"
        with open(traj_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y"])
            w.writerows(self.trajectory)

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode_id, time.time(), self.controller_name, self.map_name, result,
                int(result == "success"), round(elapsed, 3), round(path_length, 3),
                round(straight_dist, 3), round(path_efficiency, 3), round(final_dist, 3),
                round(mean_abs_ang, 4), round(ang_std, 4), round(mean_lin, 4),
                round(self.start_pos[0], 3), round(self.start_pos[1], 3),
                round(gx, 3), round(gy, 3), traj_file.name,
            ])

        self.get_logger().info(
            f"[{self.controller_name}/{self.map_name}] Episode {self.episode_id} -> {result} "
            f"(t={elapsed:.1f}s, path={path_length:.2f}m, eff={path_efficiency:.2f}, "
            f"final_dist={final_dist:.2f}m)"
        )
        self.episode_id += 1


def main():
    rclpy.init()
    node = EvalLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.episode_active:
            node._finalize_episode(result="interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
