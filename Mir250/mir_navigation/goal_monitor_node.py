#!/usr/bin/env python3
"""
Goal monitor node for MiR250 inference evaluation.

Monitors odometry and detects when the robot reaches the current goal.
Publishes the RViz goal marker (blue dot / green square / etc.) so the
policy sees the same visual target it was trained on.
Designed to run alongside inference_ros2_node.py.

Subscribes:
    /diff_cont/odom          (nav_msgs/Odometry)         — robot pose
    /map                     (nav_msgs/OccupancyGrid)    — used to pick random free-cell goals
    /inference_prompt        (std_msgs/String)           — trigger: send prompt to generate a goal
    /episode_goal            (geometry_msgs/PoseStamped) — external goal override

Publishes:
    /episode_goal            (geometry_msgs/PoseStamped) — randomly chosen goal position
    /dataset_goal_marker     (visualization_msgs/Marker) — visual goal in RViz
    /goal_reached            (std_msgs/Bool)             — latched True when goal reached
    cmd_vel_topic            (Twist or TwistStamped)     — zero cmd on goal reached

Parameters:
    odom_topic        (str)   default: /diff_cont/odom
    goal_topic        (str)   default: /episode_goal
    cmd_vel_topic     (str)   default: /diff_cont/cmd_vel_unstamped
    cmd_vel_stamped   (bool)  default: false
    threshold_m       (float) default: 0.50  — goal reached distance in metres
    check_hz          (float) default: 10.0  — how often to evaluate distance
    map_topic         (str)   default: /map
    margin_m          (float) default: 0.50  — obstacle clearance for candidate cells
    map_yaml          (str)   default: ""    — if set, load map directly from this .yaml file
                                               (skips the /map topic entirely)

Usage:
    python3 goal_monitor_node.py --ros-args \\
        -p cmd_vel_topic:=/diff_cont/cmd_vel_unstamped

Then send a prompt to generate a random goal and marker automatically:
    ros2 topic pub --once /inference_prompt std_msgs/String "{data: 'reach the blue dot'}"

Valid colors: blue, green, red, yellow
Valid shapes: dot, square, triangle
The color and shape are parsed from the prompt text automatically.
"""

import math
import random
import yaml
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Point, PoseWithCovarianceStamped
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

# Must match the recorder's COLORS dict
_COLORS = {
    "blue":   (0.0, 0.2, 1.0, 1.0),
    "green":  (0.0, 1.0, 0.0, 1.0),
    "red":    (1.0, 0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0, 1.0),
}


def _distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class GoalMonitor(Node):
    def __init__(self):
        super().__init__("mir_goal_monitor")

        self.declare_parameter("odom_topic",      "/diff_cont/odom")
        self.declare_parameter("goal_topic",      "/episode_goal")
        self.declare_parameter("cmd_vel_topic",   "/diff_cont/cmd_vel_unstamped")
        self.declare_parameter("cmd_vel_stamped", False)
        self.declare_parameter("threshold_m",     0.50)
        self.declare_parameter("check_hz",        10.0)
        self.declare_parameter("map_topic",       "/map")
        self.declare_parameter("margin_m",        0.50)
        self.declare_parameter("map_yaml",         "")
        self.declare_parameter("initial_pose_x",   0.0)
        self.declare_parameter("initial_pose_y",   0.0)
        self.declare_parameter("initial_pose_yaw", 0.0)

        gp = lambda n: self.get_parameter(n).get_parameter_value()
        odom_topic      = gp("odom_topic").string_value
        goal_topic      = gp("goal_topic").string_value
        cmd_vel_topic   = gp("cmd_vel_topic").string_value
        cmd_vel_stamped = gp("cmd_vel_stamped").bool_value
        self.threshold  = float(gp("threshold_m").double_value)
        check_hz        = float(gp("check_hz").double_value)
        map_topic       = gp("map_topic").string_value
        self.margin_m   = float(gp("margin_m").double_value)
        map_yaml        = gp("map_yaml").string_value
        init_x          = float(gp("initial_pose_x").double_value)
        init_y          = float(gp("initial_pose_y").double_value)
        init_yaw        = float(gp("initial_pose_yaw").double_value)

        self.latest_odom: Optional[Odometry] = None
        self.current_goal: Optional[PoseStamped] = None
        self.goal_reached = False
        self.free_cells: list = []
        self.marker_color = "blue"   # updated dynamically from prompt
        self.marker_shape = "dot"    # updated dynamically from prompt

        self.create_subscription(Odometry,    odom_topic, self._odom_cb,   50)
        self.create_subscription(PoseStamped, goal_topic, self._goal_cb,   10)
        self.create_subscription(String, "/inference_prompt", self._prompt_cb, 10)

        # Load map: from file (fast, no Nav2 needed) or from /map topic
        if map_yaml:
            self._load_map_from_yaml(map_yaml)
        else:
            map_qos = QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(OccupancyGrid, map_topic, self._map_cb, map_qos)

        # Latched publisher so late subscribers still see the last result
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.reached_pub   = self.create_publisher(Bool,        "/goal_reached",        latched_qos)
        self.marker_pub    = self.create_publisher(Marker,      "/dataset_goal_marker", 10)
        self.goal_pub      = self.create_publisher(PoseStamped, "/episode_goal",         10)
        self.init_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)

        pub_type = TwistStamped if cmd_vel_stamped else Twist
        self.cmd_pub = self.create_publisher(pub_type, cmd_vel_topic, 10)
        self._cmd_vel_stamped = cmd_vel_stamped

        self.create_timer(1.0 / check_hz, self._check_goal)
        # Refresh the marker every second so it stays visible in RViz
        self.create_timer(1.0, self._refresh_marker)

        # Publish AMCL initial pose so map->odom TF is established immediately.
        # Without this the marker floats at raw map coords instead of being
        # anchored in the map frame alongside the laser scan.
        self._publish_initial_pose(init_x, init_y, init_yaw)

        self.get_logger().info(
            f"Goal monitor ready | odom={odom_topic} "
            f"map={'file:'+map_yaml if map_yaml else map_topic} "
            f"cmd_vel={cmd_vel_topic} threshold={self.threshold:.2f} m "
            f"margin={self.margin_m:.2f} m | send prompt to /inference_prompt to start"
        )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    def _publish_initial_pose(self, x: float, y: float, yaw: float):
        """Send initial pose to AMCL so it publishes the map->odom TF.
        Without this the marker (map frame) is not aligned with the laser
        scan (odom frame) in RViz."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        half = yaw / 2.0
        msg.pose.pose.orientation.z = math.sin(half)
        msg.pose.pose.orientation.w = math.cos(half)
        msg.pose.covariance[0]  = 0.25   # x
        msg.pose.covariance[7]  = 0.25   # y
        msg.pose.covariance[35] = 0.068  # yaw
        for _ in range(3):
            self.init_pose_pub.publish(msg)
        self.get_logger().info(
            f"Initial pose published: x={x:.2f} y={y:.2f} "
            f"yaw={math.degrees(yaw):.1f} deg -> AMCL will establish map->odom TF"
        )

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal = msg
        self.goal_reached = False           # reset on new goal
        self.get_logger().info(
            f"New goal received: x={msg.pose.position.x:.3f} "
            f"y={msg.pose.position.y:.3f} frame={msg.header.frame_id}"
        )
        self._publish_marker(msg)

    def _refresh_marker(self):
        """Re-publish the marker every second so RViz doesn't expire it."""
        if self.current_goal is not None and not self.goal_reached:
            self._publish_marker(self.current_goal)

    # ------------------------------------------------------------------ #
    # Map loading
    # ------------------------------------------------------------------ #
    def _load_map_from_yaml(self, yaml_path: str):
        """Load map directly from a ROS-style .yaml + .pgm file pair."""
        yaml_path = Path(yaml_path).expanduser().resolve()
        if not yaml_path.exists():
            self.get_logger().error(f"map_yaml not found: {yaml_path}")
            return
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        pgm_path = yaml_path.parent / meta["image"]
        img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error(f"Cannot read map image: {pgm_path}")
            return

        res         = float(meta["resolution"])
        origin      = meta["origin"]
        ox, oy      = float(origin[0]), float(origin[1])
        negate      = int(meta.get("negate", 0))
        occ_thresh  = float(meta.get("occupied_thresh", 0.65))
        free_thresh = float(meta.get("free_thresh", 0.25))

        # PGM row 0 = image top = map y-max; flip so row 0 = map y-min (ROS convention)
        img = img[::-1, :]
        h, w = img.shape

        pix = img.astype(np.float32) / 255.0
        occ = pix if negate else (1.0 - pix)

        margin_cells = int(math.ceil(self.margin_m / res))
        too_close: set = set()
        for row in range(h):
            for col in range(w):
                if occ[row, col] >= occ_thresh:
                    for dr in range(-margin_cells, margin_cells + 1):
                        for dc in range(-margin_cells, margin_cells + 1):
                            if dr * dr + dc * dc <= margin_cells * margin_cells:
                                nr, nc = row + dr, col + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    too_close.add(nr * w + nc)

        start_col = max(0, min(w - 1, int((-ox) / res)))
        start_row = max(0, min(h - 1, int((-oy) / res)))
        start_idx = start_row * w + start_col
        visited: set = {start_idx}
        queue = deque([start_idx])
        free = []
        while queue:
            idx = queue.popleft()
            if idx in too_close:
                continue
            row = idx // w
            col = idx % w
            if occ[row, col] <= free_thresh:
                free.append((ox + (col + 0.5) * res, oy + (row + 0.5) * res))
                for nr, nc in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        nidx = nr * w + nc
                        if nidx not in visited:
                            visited.add(nidx)
                            queue.append(nidx)
        self.free_cells = free
        self.get_logger().info(
            f"Map loaded from file '{yaml_path.name}': "
            f"{len(free)} free cells (margin={self.margin_m:.2f} m)"
        )

    def _map_cb(self, msg: OccupancyGrid):
        """Parse the occupancy grid once and store all reachable free cells."""
        if self.free_cells:   # already loaded
            return
        info = msg.info
        res  = info.resolution
        width, height = info.width, info.height
        data = msg.data

        margin_cells = int(math.ceil(self.margin_m / res))
        too_close: set = set()
        for idx, val in enumerate(data):
            if val > 0:
                col = idx % width
                row = idx // width
                for dr in range(-margin_cells, margin_cells + 1):
                    for dc in range(-margin_cells, margin_cells + 1):
                        if dr * dr + dc * dc <= margin_cells * margin_cells:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                too_close.add(nr * width + nc)

        ox = info.origin.position.x
        oy = info.origin.position.y
        start_col = max(0, min(width  - 1, int((-ox) / res)))
        start_row = max(0, min(height - 1, int((-oy) / res)))
        start_idx = start_row * width + start_col
        visited: set = {start_idx}
        queue = deque([start_idx])
        free = []
        while queue:
            idx = queue.popleft()
            if data[idx] == 0 and idx not in too_close:
                col = idx % width
                row = idx // width
                free.append((ox + (col + 0.5) * res, oy + (row + 0.5) * res))
                for nr, nc in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                    if 0 <= nr < height and 0 <= nc < width:
                        nidx = nr * width + nc
                        if nidx not in visited and data[nidx] == 0 and nidx not in too_close:
                            visited.add(nidx)
                            queue.append(nidx)
        self.free_cells = free
        self.get_logger().info(
            f"Map loaded from /map topic: {len(free)} free cells (margin={self.margin_m:.2f} m)"
        )

    # ------------------------------------------------------------------ #
    # Prompt handler
    # ------------------------------------------------------------------ #
    def _prompt_cb(self, msg: String):
        """Parse color+shape from prompt, pick a random free cell, publish goal+marker."""
        prompt = msg.data.strip().lower()
        color = next((c for c in _COLORS if c in prompt), None)
        shape = next((s for s in ("dot", "square", "triangle") if s in prompt), None)
        if color is None or shape is None:
            self.get_logger().error(
                f"Cannot parse color/shape from '{msg.data}'. "
                f"Use: {list(_COLORS.keys())} and dot/square/triangle."
            )
            return
        if not self.free_cells:
            self.get_logger().error(
                "No map free cells yet — is the map server running? "
                "Or set -p map_yaml:=/path/to/map.yaml"
            )
            return
        self.marker_color = color
        self.marker_shape = shape
        x, y = random.choice(self.free_cells)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self.get_logger().info(
            f"Prompt '{msg.data}' -> random goal ({x:.2f}, {y:.2f}) "
            f"marker={color} {shape}"
        )

    # ------------------------------------------------------------------ #
    # Distance check
    # ------------------------------------------------------------------ #
    def _check_goal(self):
        if self.goal_reached:
            return
        if self.latest_odom is None or self.current_goal is None:
            return

        rx = self.latest_odom.pose.pose.position.x
        ry = self.latest_odom.pose.pose.position.y
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        dist = _distance_2d(rx, ry, gx, gy)

        if dist < self.threshold:
            self.goal_reached = True
            self.get_logger().info(
                f"GOAL REACHED — distance={dist:.3f} m (threshold={self.threshold:.2f} m)"
            )
            self._delete_marker(self.current_goal.header.frame_id)
            self._publish_reached()
            self._stop_robot()

    # ------------------------------------------------------------------ #
    # Marker helpers  (mirrors leRobotDatasetRecorder.publish_goal_marker)
    # ------------------------------------------------------------------ #
    def _publish_marker(self, goal: PoseStamped):
        marker = Marker()
        marker.header.frame_id = goal.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lerobot_dataset_goal"
        marker.id = 0
        marker.action = Marker.ADD
        marker.pose.position.x = goal.pose.position.x
        marker.pose.position.y = goal.pose.position.y
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        r, g, b, a = _COLORS[self.marker_color]
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, a
        if self.marker_shape == "dot":
            marker.type = Marker.SPHERE
            marker.scale.x = marker.scale.y = 0.35
            marker.scale.z = 0.08
        elif self.marker_shape == "square":
            marker.type = Marker.CUBE
            marker.scale.x = marker.scale.y = 0.35
            marker.scale.z = 0.06
        elif self.marker_shape == "triangle":
            marker.type = Marker.TRIANGLE_LIST
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            size = 0.35
            p1 = Point(); p1.x, p1.y, p1.z =  0.0,  size, 0.0
            p2 = Point(); p2.x, p2.y, p2.z = -size, -size, 0.0
            p3 = Point(); p3.x, p3.y, p3.z =  size, -size, 0.0
            marker.points = [p1, p2, p3]
        self.marker_pub.publish(marker)

    def _delete_marker(self, frame_id: str):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lerobot_dataset_goal"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)

    # ------------------------------------------------------------------ #
    # Publishers
    # ------------------------------------------------------------------ #
    def _publish_reached(self):
        msg = Bool()
        msg.data = True
        self.reached_pub.publish(msg)

    def _stop_robot(self):
        if self._cmd_vel_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
        else:
            msg = Twist()
        self.cmd_pub.publish(msg)
        self.get_logger().info("Zero cmd_vel published — robot stopped.")


def main():
    rclpy.init()
    node = GoalMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
