#!/usr/bin/env python3

"""
ROS 2 node for recording a LeRobotDataset v3-style navigation dataset.

Each episode starts when a goal is received on /episode_goal.
The node:
  - chooses a random visual target style, e.g. blue dot, green triangle
  - publishes a RViz marker at the goal position
  - creates a language prompt, e.g. "reach the blue dot"
  - captures the RViz window/screen at 30 FPS
  - records /odom as observation.state
  - records /cmd_vel as action
  - stops the episode when the robot reaches the goal or times out
  - saves the episode with LeRobotDataset.save_episode()

Important:
  - The planned path is NOT recorded.
  - The goal coordinates are NOT stored in observation.state.
  - The goal is used internally only for stopping the episode.
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
from geometry_msgs.msg import Twist, PoseStamped, Point
from visualization_msgs.msg import Marker

try:
    from lerobot.datasets import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


# =============================================================================
# Dataset configuration
# =============================================================================

FPS = 30
WIDTH = 1280
HEIGHT = 720

DATASET_BASE_DIR= Path.home() / "lerobot_ros2_rviz_dataset"
ROBOT_TYPE = "ros2_mobile_robot"

NUM_EPISODES = 100

# =============================================================================
# ROS topics
# =============================================================================

GOAL_TOPIC = "/episode_goal"
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel"
MARKER_TOPIC = "/dataset_goal_marker"

# =============================================================================
# Episode configuration
# =============================================================================

GOAL_REACHED_THRESHOLD_M = 0.30
MAX_EPISODE_DURATION_SEC = 180.0

# =============================================================================
# Screen capture configuration
# =============================================================================
#
# This captures a fixed region of the screen.
# Put your RViz rendering area at this position.
#
# On Ubuntu 22.04, this works best on X11.
# If you are on Wayland, use "Ubuntu on Xorg" at login.
#

SCREEN_CAPTURE_REGION = {
    "top": 0,
    "left": 0,
    "width": WIDTH,
    "height": HEIGHT,
}


# =============================================================================
# Visual goal styles
# =============================================================================

COLORS = {
    "blue": {
        "rgba": (0.0, 0.2, 1.0, 1.0),
    },
    "green": {
        "rgba": (0.0, 1.0, 0.0, 1.0),
    },
    "red": {
        "rgba": (1.0, 0.0, 0.0, 1.0),
    },
    "yellow": {
        "rgba": (1.0, 1.0, 0.0, 1.0),
    },
}

SHAPES = [
    "dot",
    "triangle",
    "square",
]


# =============================================================================
# Utility functions
# =============================================================================

def quaternion_to_yaw(q) -> float:
    """
    Convert quaternion to yaw angle.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Euclidean distance in 2D.
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# =============================================================================
# Main ROS 2 node
# =============================================================================

class LeRobotRvizDatasetRecorder(Node):
    def __init__(self):
        super().__init__("lerobot_rviz_dataset_recorder")

        self.declare_parameter("world_name", "maze")
        world_name =self.get_parameter("world_name").get_parameter_value().string_value
        self.dataset_root = DATASET_BASE_DIR / f"Images_{world_name}"
        self.repo_id = f"local/ros2-rviz-{world_name}-navigation"

        self.latest_odom: Optional[Odometry] = None
        self.latest_cmd_vel: Twist = Twist()

        self.pending_goal: Optional[PoseStamped] = None

        self.odom_sub = self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.odom_callback,
            50,
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            CMD_VEL_TOPIC,
            self.cmd_vel_callback,
            50,
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            GOAL_TOPIC,
            self.goal_callback,
            10,
        )

        self.marker_pub = self.create_publisher(
            Marker,
            MARKER_TOPIC,
            10,
        )

        self.dataset = self.create_lerobot_dataset()

        self.get_logger().info("LeRobot RViz dataset recorder started.")
        self.get_logger().info(f"World: {world_name}")
        self.get_logger().info(f"Waiting for goals on topic: {GOAL_TOPIC}")
        self.get_logger().info(f"Recording dataset to: {self.dataset_root}")


    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def cmd_vel_callback(self, msg: Twist):
        self.latest_cmd_vel = msg

    def goal_callback(self, msg: PoseStamped):
        self.pending_goal = msg
        self.get_logger().info(
            f"Received new episode goal: "
            f"x={msg.pose.position.x:.3f}, "
            f"y={msg.pose.position.y:.3f}, "
            f"frame={msg.header.frame_id}"
        )

    # -------------------------------------------------------------------------
    # LeRobot dataset creation
    # -------------------------------------------------------------------------

    def create_lerobot_dataset(self) -> LeRobotDataset:
        """
        Create a LeRobotDataset.

        Features:
          observation.images.rviz:
            Captured RViz image.

          observation.state:
            Robot odometry only.
            Goal coordinates are intentionally not stored here.

          action:
            /cmd_vel command.
        """

        features = {
            "observation.images.rviz": {
                "dtype": "video",
                "shape": (HEIGHT, WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": [
                    "odom_x",
                    "odom_y",
                    "odom_yaw",
                    "odom_linear_x",
                    "odom_linear_y",
                    "odom_angular_z",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (3,),
                "names": [
                    "cmd_vel_linear_x",
                    "cmd_vel_linear_y",
                    "cmd_vel_angular_z",
                ],
            },
        }

        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.dataset_root,
            fps=FPS,
            features=features,
            robot_type=ROBOT_TYPE,
            use_videos=True,
            streaming_encoding=True,
            vcodec="h264",
            encoder_threads=2,
        )

        return dataset

    # -------------------------------------------------------------------------
    # Goal style and prompt
    # -------------------------------------------------------------------------

    def choose_goal_style(self) -> Tuple[str, str]:
        """
        Randomly choose a color and shape for the current episode.
        """
        color_name = random.choice(list(COLORS.keys()))
        shape_name = random.choice(SHAPES)
        return color_name, shape_name

    def create_prompt(self, color_name: str, shape_name: str) -> str:
        """
        Create the natural-language task prompt.
        """
        return f"reach the {color_name} {shape_name}"

    # -------------------------------------------------------------------------
    # RViz marker publishing
    # -------------------------------------------------------------------------

    def publish_goal_marker(
        self,
        goal_pose: PoseStamped,
        color_name: str,
        shape_name: str,
    ):
        """
        Publish a colored marker at the goal position.

        The marker is visible in RViz and becomes part of the captured image.
        """

        marker = Marker()

        marker.header.frame_id = goal_pose.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "lerobot_dataset_goal"
        marker.id = 0
        marker.action = Marker.ADD

        marker.pose.position.x = goal_pose.pose.position.x
        marker.pose.position.y = goal_pose.pose.position.y
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0

        r, g, b, a = COLORS[color_name]["rgba"]
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)

        if shape_name == "dot":
            marker.type = Marker.SPHERE
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.08

        elif shape_name == "square":
            marker.type = Marker.CUBE
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.06

        elif shape_name == "triangle":
            marker.type = Marker.TRIANGLE_LIST
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0

            size = 0.35

            p1 = Point()
            p1.x = 0.0
            p1.y = size
            p1.z = 0.0

            p2 = Point()
            p2.x = -size
            p2.y = -size
            p2.z = 0.0

            p3 = Point()
            p3.x = size
            p3.y = -size
            p3.z = 0.0

            marker.points = [p1, p2, p3]

        else:
            raise ValueError(f"Unknown shape: {shape_name}")

        self.marker_pub.publish(marker)

    def delete_goal_marker(self, frame_id: str):
        """
        Delete the current marker from RViz.
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lerobot_dataset_goal"
        marker.id = 0
        marker.action = Marker.DELETE

        self.marker_pub.publish(marker)

    # -------------------------------------------------------------------------
    # Screen capture
    # -------------------------------------------------------------------------

    def capture_rviz_image(self, sct: mss.mss) -> np.ndarray:
        """
        Capture the RViz screen region.

        Returns:
            RGB uint8 image with shape [HEIGHT, WIDTH, 3].
        """

        raw = np.array(sct.grab(SCREEN_CAPTURE_REGION))

        # mss returns BGRA.
        image_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if image_rgb.shape[0] != HEIGHT or image_rgb.shape[1] != WIDTH:
            image_rgb = cv2.resize(image_rgb, (WIDTH, HEIGHT))

        return image_rgb.astype(np.uint8)

    # -------------------------------------------------------------------------
    # State/action construction
    # -------------------------------------------------------------------------

    def build_state_action_and_distance(
        self,
        goal_pose: PoseStamped,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Build the LeRobot frame data from the latest ROS messages.

        Important:
            The goal position is NOT added to observation.state.
            It is only used to compute episode termination.
        """

        if self.latest_odom is None:
            return None, None, None

        odom = self.latest_odom
        cmd = self.latest_cmd_vel

        odom_x = odom.pose.pose.position.x
        odom_y = odom.pose.pose.position.y
        odom_yaw = quaternion_to_yaw(odom.pose.pose.orientation)

        odom_linear_x = odom.twist.twist.linear.x
        odom_linear_y = odom.twist.twist.linear.y
        odom_angular_z = odom.twist.twist.angular.z

        state = np.array(
            [
                odom_x,
                odom_y,
                odom_yaw,
                odom_linear_x,
                odom_linear_y,
                odom_angular_z,
            ],
            dtype=np.float32,
        )

        action = np.array(
            [
                cmd.linear.x,
                cmd.linear.y,
                cmd.angular.z,
            ],
            dtype=np.float32,
        )

        goal_x = goal_pose.pose.position.x
        goal_y = goal_pose.pose.position.y

        distance_to_goal = distance_2d(
            odom_x,
            odom_y,
            goal_x,
            goal_y,
        )

        return state, action, distance_to_goal

    # -------------------------------------------------------------------------
    # Episode handling
    # -------------------------------------------------------------------------

    def wait_for_next_goal(self) -> PoseStamped:
        """
        Wait until a new goal is received.
        """
        self.pending_goal = None

        while rclpy.ok() and self.pending_goal is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.pending_goal

    def wait_until_odom_available(self):
        """
        Wait until /odom has been received at least once.
        """
        while rclpy.ok() and self.latest_odom is None:
            self.get_logger().info("Waiting for /odom...")
            rclpy.spin_once(self, timeout_sec=0.5)

    def record_one_episode(self, episode_index: int):
        """
        Record one navigation episode.
        """

        self.get_logger().info(f"Waiting for goal for episode {episode_index}...")
        goal_pose = self.wait_for_next_goal()

        if goal_pose is None:
            return

        color_name, shape_name = self.choose_goal_style()
        task_prompt = self.create_prompt(color_name, shape_name)

        self.get_logger().info(
            f"Starting episode {episode_index}: "
            f"task='{task_prompt}', "
            f"goal_x={goal_pose.pose.position.x:.3f}, "
            f"goal_y={goal_pose.pose.position.y:.3f}"
        )

        # Publish marker several times before starting capture.
        # This helps ensure RViz receives it before the first recorded frame.
        for _ in range(10):
            self.publish_goal_marker(goal_pose, color_name, shape_name)
            rclpy.spin_once(self, timeout_sec=0.05)

        frame_period = 1.0 / float(FPS)
        start_time = time.time()
        last_marker_publish_time = 0.0

        frame_count = 0
        episode_result = "unknown"

        with mss.MSS(display=os.environ.get("DISPLAY", ":0")) as sct:
            while rclpy.ok():
                loop_start = time.time()

                # Process ROS callbacks.
                rclpy.spin_once(self, timeout_sec=0.001)

                now = time.time()
                elapsed = now - start_time

                # Republish the marker periodically.
                if now - last_marker_publish_time > 1.0:
                    self.publish_goal_marker(goal_pose, color_name, shape_name)
                    last_marker_publish_time = now

                state, action, distance_to_goal = self.build_state_action_and_distance(
                    goal_pose
                )

                if state is None:
                    time.sleep(0.01)
                    continue

                image = self.capture_rviz_image(sct)

                frame = {
                    "observation.images.rviz": image,
                    "observation.state": state,
                    "action": action,
                    "task": task_prompt,
                }

                self.dataset.add_frame(frame)
                frame_count += 1

                if distance_to_goal < GOAL_REACHED_THRESHOLD_M:
                    episode_result = "success"
                    self.get_logger().info(
                        f"Goal reached. Distance={distance_to_goal:.3f} m"
                    )
                    break

                if elapsed > MAX_EPISODE_DURATION_SEC:
                    episode_result = "timeout"
                    self.get_logger().warn(
                        f"Episode timeout after {elapsed:.2f} seconds."
                    )
                    break

                sleep_time = frame_period - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if frame_count > 0:
            self.dataset.save_episode()
            self.get_logger().info(
                f"Saved episode {episode_index}: "
                f"frames={frame_count}, "
                f"result={episode_result}, "
                f"task='{task_prompt}'"
            )
        else:
            self.dataset.clear_episode_buffer()
            self.get_logger().warn(
                f"Episode {episode_index} had zero frames and was discarded."
            )

        self.delete_goal_marker(goal_pose.header.frame_id)

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def close_dataset(self):
        """
        Finalize the LeRobot dataset.

        This is required before reading the dataset or pushing it to the Hub.
        """
        self.get_logger().info("Finalizing LeRobot dataset...")
        self.dataset.finalize()
        self.get_logger().info("Dataset finalized.")


# =============================================================================
# Main
# =============================================================================

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
