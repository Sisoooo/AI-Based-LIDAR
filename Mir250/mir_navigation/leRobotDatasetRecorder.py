#!/usr/bin/env python3

"""
ROS 2 node for recording a LeRobotDataset v3-style navigation dataset.

FIXED VERSION (COGAR_A3)
------------------------
Fixes vs the original recorder:

  1. cmd_vel TOPIC BUG (the reason action was all-zero):
     Odometry is published on /diff_cont/odom -> the robot uses ros2_control's
     diff_drive_controller named "diff_cont", whose velocity-command topic is
     /diff_cont/cmd_vel_unstamped (Twist) or /diff_cont/cmd_vel (TwistStamped),
     NOT plain /cmd_vel.  Subscribing to /cmd_vel captured nothing, so the
     action stayed at the zero-initialized default for every frame.
     -> The cmd_vel topic and message type are now ROS PARAMETERS, defaulting
        to /diff_cont/cmd_vel_unstamped.

  2. SILENT-ZERO GUARD:
     The node counts cmd_vel messages. If no publisher exists on the command
     topic at startup, or the first episode receives zero cmd_vel messages, it
     logs a LOUD error and ABORTS — so you never again record a whole dataset
     of dead actions.

  3. STATE = [odom(6), cmd_vel[t-1](3)]  (9 dims), ACTION = cmd_vel[t] (3 dims).
     The previous executed command is appended to the proprioceptive state.
     prev_cmd resets to zero at the start of every episode and is updated to
     the last action after each recorded frame -> state[t] = [odom[t], cmd[t-1]].

FIND YOUR REAL COMMAND TOPIC (run on the robot before recording):
     ros2 topic list | grep -i cmd_vel
     ros2 topic info -v /diff_cont/cmd_vel_unstamped     # type + publisher count
     ros2 topic echo /diff_cont/cmd_vel_unstamped        # must show NONZERO while driving
   Then launch with the matching topic/type, e.g.:
     ros2 run <pkg> leRobotDatasetRecorder --ros-args \
         -p cmd_vel_topic:=/diff_cont/cmd_vel_unstamped -p cmd_vel_stamped:=false
"""

import math
import os
import time
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Point
from visualization_msgs.msg import Marker

try:
    from lerobot.datasets import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


# =============================================================================
# Dataset configuration
# =============================================================================

FPS = 30
WIDTH = 1920
HEIGHT = 1200

DATASET_BASE_DIR = Path.home() / "lerobot_ros2_rviz_dataset"
ROBOT_TYPE = "ros2_mobile_robot"

NUM_EPISODES = 100

# =============================================================================
# ROS topics — DEFAULTS (override with --ros-args -p ...)
# =============================================================================

GOAL_TOPIC = "/episode_goal"
ODOM_TOPIC_DEFAULT = "/diff_cont/odom"
# This is the command topic that actually drives the diff_cont controller.
# Verify with: ros2 topic list | grep cmd_vel
CMD_VEL_TOPIC_DEFAULT = "/diff_cont/cmd_vel_unstamped"
CMD_VEL_STAMPED_DEFAULT = False  # True if topic is geometry_msgs/TwistStamped
MARKER_TOPIC = "/dataset_goal_marker"

# =============================================================================
# Episode configuration
# =============================================================================

GOAL_REACHED_THRESHOLD_M = 0.50
MAX_EPISODE_DURATION_SEC = 180.0

SCREEN_CAPTURE_REGION = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}

# =============================================================================
# Visual goal styles
# =============================================================================

COLORS = {
    "blue":   {"rgba": (0.0, 0.2, 1.0, 1.0)},
    "green":  {"rgba": (0.0, 1.0, 0.0, 1.0)},
    "red":    {"rgba": (1.0, 0.0, 0.0, 1.0)},
    "yellow": {"rgba": (1.0, 1.0, 0.0, 1.0)},
}
SHAPES = ["dot", "triangle", "square"]


# =============================================================================
# Utilities
# =============================================================================

def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# =============================================================================
# Main ROS 2 node
# =============================================================================

class LeRobotRvizDatasetRecorder(Node):
    def __init__(self):
        super().__init__("lerobot_rviz_dataset_recorder")

        self.declare_parameter("world", "maze")
        self.declare_parameter("odom_topic", ODOM_TOPIC_DEFAULT)
        self.declare_parameter("cmd_vel_topic", CMD_VEL_TOPIC_DEFAULT)
        self.declare_parameter("cmd_vel_stamped", CMD_VEL_STAMPED_DEFAULT)

        world_name = self.get_parameter("world").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self.cmd_vel_stamped = self.get_parameter("cmd_vel_stamped").get_parameter_value().bool_value

        self.dataset_root = DATASET_BASE_DIR / f"Images_{world_name}"
        self.repo_id = f"local/ros2-rviz-{world_name}-navigation"

        self.latest_odom: Optional[Odometry] = None
        self.latest_cmd_vel: Twist = Twist()

        self.cmd_vel_msg_count = 0
        self.prev_cmd = np.zeros(3, dtype=np.float32)
        self.pending_goal: Optional[PoseStamped] = None

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 50)
        cmd_type = TwistStamped if self.cmd_vel_stamped else Twist
        self.cmd_vel_sub = self.create_subscription(cmd_type, self.cmd_vel_topic, self.cmd_vel_callback, 50)
        self.goal_sub = self.create_subscription(PoseStamped, GOAL_TOPIC, self.goal_callback, 10)
        self.marker_pub = self.create_publisher(Marker, MARKER_TOPIC, 10)

        self.get_logger().info("LeRobot RViz dataset recorder started (FIXED).")
        self.get_logger().info(f"World: {world_name}")
        self.get_logger().info(f"odom_topic    : {self.odom_topic}")
        self.get_logger().info(f"cmd_vel_topic : {self.cmd_vel_topic} (stamped={self.cmd_vel_stamped})")
        self.get_logger().info(f"Recording dataset to: {self.dataset_root}")

        time.sleep(1.0)
        rclpy.spin_once(self, timeout_sec=0.5)
        n_pub = self.count_publishers(self.cmd_vel_topic)
        if n_pub == 0:
            self.get_logger().error(
                f"!!! NO PUBLISHER on '{self.cmd_vel_topic}'. The action would be ALL ZERO "
                f"(the exact bug we fixed). Run: ros2 topic list | grep cmd_vel  and relaunch "
                f"with -p cmd_vel_topic:=<topic> -p cmd_vel_stamped:=<true|false>."
            )
        else:
            self.get_logger().info(f"OK: {n_pub} publisher(s) on {self.cmd_vel_topic}")

    # ---- callbacks ----
    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def cmd_vel_callback(self, msg):
        self.latest_cmd_vel = msg.twist if self.cmd_vel_stamped else msg
        self.cmd_vel_msg_count += 1

    def goal_callback(self, msg: PoseStamped):
        self.pending_goal = msg
        self.get_logger().info(
            f"Received goal: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, "
            f"frame={msg.header.frame_id}"
        )

    # ---- dataset creation ----
    def create_lerobot_dataset(self, episode_index: int) -> LeRobotDataset:
        episode_root = self.dataset_root.parent / f"{self.dataset_root.name}_episode_{episode_index}"
        features = {
            "observation.images.rviz": {
                "dtype": "video", "shape": (HEIGHT, WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32", "shape": (9,),
                "names": ["odom_x", "odom_y", "odom_yaw",
                          "odom_linear_x", "odom_linear_y", "odom_angular_z",
                          "prev_cmd_linear_x", "prev_cmd_linear_y", "prev_cmd_angular_z"],
            },
            "action": {
                "dtype": "float32", "shape": (3,),
                "names": ["cmd_vel_linear_x", "cmd_vel_linear_y", "cmd_vel_angular_z"],
            },
        }
        return LeRobotDataset.create(
            repo_id=self.repo_id, root=episode_root, fps=FPS, features=features,
            robot_type=ROBOT_TYPE, use_videos=True, streaming_encoding=True,
            vcodec="h264", encoder_threads=2,
        )

    # ---- prompts ----
    def choose_goal_style(self) -> Tuple[str, str]:
        return random.choice(list(COLORS.keys())), random.choice(SHAPES)

    def create_prompt(self, color_name: str, shape_name: str) -> str:
        return f"reach the {color_name} {shape_name}"

    # ---- markers ----
    def publish_goal_marker(self, goal_pose: PoseStamped, color_name: str, shape_name: str):
        marker = Marker()
        marker.header.frame_id = goal_pose.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lerobot_dataset_goal"; marker.id = 0; marker.action = Marker.ADD
        marker.pose.position.x = goal_pose.pose.position.x
        marker.pose.position.y = goal_pose.pose.position.y
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        r, g, b, a = COLORS[color_name]["rgba"]
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = float(r), float(g), float(b), float(a)
        if shape_name == "dot":
            marker.type = Marker.SPHERE
            marker.scale.x = marker.scale.y = 0.35; marker.scale.z = 0.08
        elif shape_name == "square":
            marker.type = Marker.CUBE
            marker.scale.x = marker.scale.y = 0.35; marker.scale.z = 0.06
        elif shape_name == "triangle":
            marker.type = Marker.TRIANGLE_LIST
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            size = 0.35
            p1 = Point(); p1.x, p1.y, p1.z = 0.0, size, 0.0
            p2 = Point(); p2.x, p2.y, p2.z = -size, -size, 0.0
            p3 = Point(); p3.x, p3.y, p3.z = size, -size, 0.0
            marker.points = [p1, p2, p3]
        else:
            raise ValueError(f"Unknown shape: {shape_name}")
        self.marker_pub.publish(marker)

    def delete_goal_marker(self, frame_id: str):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lerobot_dataset_goal"; marker.id = 0; marker.action = Marker.DELETE
        self.marker_pub.publish(marker)

    # ---- screen capture ----
    def capture_rviz_image(self, sct) -> np.ndarray:
        raw = np.array(sct.grab(SCREEN_CAPTURE_REGION))
        image_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if image_rgb.shape[0] != HEIGHT or image_rgb.shape[1] != WIDTH:
            image_rgb = cv2.resize(image_rgb, (WIDTH, HEIGHT))
        return image_rgb.astype(np.uint8)

    # ---- state/action ----
    def build_state_action_and_distance(self, goal_pose: PoseStamped):
        if self.latest_odom is None:
            return None, None, None
        odom = self.latest_odom
        cmd = self.latest_cmd_vel
        odom_x = odom.pose.pose.position.x
        odom_y = odom.pose.pose.position.y
        odom_yaw = quaternion_to_yaw(odom.pose.pose.orientation)
        odom_lx = odom.twist.twist.linear.x
        odom_ly = odom.twist.twist.linear.y
        odom_wz = odom.twist.twist.angular.z

        action = np.array([cmd.linear.x, cmd.linear.y, cmd.angular.z], dtype=np.float32)
        state = np.array(
            [odom_x, odom_y, odom_yaw, odom_lx, odom_ly, odom_wz,
             float(self.prev_cmd[0]), float(self.prev_cmd[1]), float(self.prev_cmd[2])],
            dtype=np.float32,
        )
        dist = distance_2d(odom_x, odom_y, goal_pose.pose.position.x, goal_pose.pose.position.y)
        return state, action, dist

    # ---- episode handling ----
    def wait_for_next_goal(self) -> PoseStamped:
        while rclpy.ok() and self.pending_goal is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        goal = self.pending_goal
        self.pending_goal = None
        return goal

    def wait_until_odom_available(self):
        while rclpy.ok() and self.latest_odom is None:
            self.get_logger().info("Waiting for odom...")
            rclpy.spin_once(self, timeout_sec=0.5)

    def record_one_episode(self, episode_index: int):
        self.get_logger().info(f"Waiting for goal for episode {episode_index}...")
        goal_pose = self.wait_for_next_goal()
        if goal_pose is None:
            return

        color_name, shape_name = self.choose_goal_style()
        task_prompt = self.create_prompt(color_name, shape_name)
        self.get_logger().info(f"Episode {episode_index}: task='{task_prompt}'")

        dataset = self.create_lerobot_dataset(episode_index)
        for _ in range(10):
            self.publish_goal_marker(goal_pose, color_name, shape_name)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.wait_until_odom_available()

        frame_period = 1.0 / float(FPS)
        start_time = time.time()
        last_marker = 0.0
        frame_count = 0
        episode_result = "unknown"
        self.pending_goal = None
        self.prev_cmd = np.zeros(3, dtype=np.float32)
        cmd_count_at_start = self.cmd_vel_msg_count

        try:
            with mss.MSS(display=os.environ.get("DISPLAY", ":0")) as sct:
                self.get_logger().info("Screen capture started")
                while rclpy.ok():
                    loop_start = time.time()
                    rclpy.spin_once(self, timeout_sec=0.001)

                    if self.pending_goal is not None:
                        new_goal = self.pending_goal
                        self.pending_goal = None
                        if distance_2d(new_goal.pose.position.x, new_goal.pose.position.y,
                                       goal_pose.pose.position.x, goal_pose.pose.position.y) > 0.5:
                            self.pending_goal = new_goal
                            episode_result = "new_goal"
                            break

                    now = time.time()
                    elapsed = now - start_time
                    if now - last_marker > 1.0:
                        self.publish_goal_marker(goal_pose, color_name, shape_name)
                        last_marker = now

                    state, action, dist = self.build_state_action_and_distance(goal_pose)
                    if state is None:
                        time.sleep(0.01)
                        continue

                    image = self.capture_rviz_image(sct)
                    dataset.add_frame({
                        "observation.images.rviz": image,
                        "observation.state": state,
                        "action": action,
                        "task": task_prompt,
                    })
                    frame_count += 1
                    self.prev_cmd = action   # becomes cmd_vel[t-1] next frame

                    if dist < GOAL_REACHED_THRESHOLD_M:
                        episode_result = "success"
                        self.get_logger().info(f"Goal reached. dist={dist:.3f} m")
                        break
                    if elapsed > MAX_EPISODE_DURATION_SEC:
                        episode_result = "timeout"
                        break

                    sleep_time = frame_period - (time.time() - loop_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except Exception as e:
            import traceback
            self.get_logger().error(f"Episode error: {type(e).__name__}: {e}\n{traceback.format_exc()}")

        cmd_this_ep = self.cmd_vel_msg_count - cmd_count_at_start
        self.get_logger().info(f"Episode {episode_index}: cmd_vel msgs = {cmd_this_ep}")
        if cmd_this_ep == 0:
            self.get_logger().error(
                f"!!! ZERO cmd_vel messages on '{self.cmd_vel_topic}'. ACTION WOULD BE ALL ZERO. "
                f"Fix the topic (ros2 topic list | grep cmd_vel) and relaunch. Aborting."
            )
            try:
                dataset.clear_episode_buffer()
            except Exception:
                pass
            dataset.finalize()
            raise SystemExit(2)

        if frame_count > 0:
            dataset.save_episode()
            self.get_logger().info(
                f"Saved episode {episode_index}: frames={frame_count}, result={episode_result}"
            )
        else:
            dataset.clear_episode_buffer()
            self.get_logger().warn(f"Episode {episode_index} had zero frames; discarded.")

        dataset.finalize()
        self.delete_goal_marker(goal_pose.header.frame_id)

    def close_dataset(self):
        self.get_logger().info("Shutdown: all datasets finalized.")


def main():
    rclpy.init()
    node = LeRobotRvizDatasetRecorder()
    try:
        for episode_index in range(NUM_EPISODES):
            if not rclpy.ok():
                break
            node.record_one_episode(episode_index)
    except KeyboardInterrupt:
        node.get_logger().warn("Keyboard interrupt received.")
    finally:
        node.close_dataset()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
