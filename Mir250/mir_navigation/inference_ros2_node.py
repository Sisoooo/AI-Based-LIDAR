#!/usr/bin/env python3
"""
Inference ROS 2 node for the MiR250 RViz navigation policy (pi0.5 via openpi).

Closed loop, mirrors the recorder observation:
    observation/state = [odom(6), cmd_vel[t-1](3)]   (9 dims)
    observation/image = RViz screen capture (same region as recorder, 1920x1200 RGB)
    prompt            = e.g. "reach the red square"  (colors: blue/green/red/yellow; shapes: dot/triangle/square)
  -> query the openpi policy server -> action chunk (N, 3) of cmd_vel [lin_x, lin_y(~0), ang_z]
  -> publish each action, update prev_cmd, re-query.

VERIFIED (ws2, 2026-06-25): openpi serving does NOT apply the repack transform ->
this node correctly sends the INTERNAL keys observation/image, observation/state, prompt.

--- ON ws2 (GPU box 130.251.13.151) start the policy server first ---
    cd ~/vla_nav/openpi
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 UV_NO_SYNC=1 \
      uv run --no-sync scripts/serve_policy.py policy:checkpoint \
      --policy.config=pi05_mir_nav --policy.dir=../ckpt/mir_nav/29999
  (leave it running, e.g. in tmux -> websocket on :8000)

--- run THIS node where RViz is visible (it screen-captures RViz) ---
  * node ON ws2 (RViz on ws2):   -p policy_host:=127.0.0.1
  * node on another machine:     -p policy_host:=130.251.13.151
    python3 inference_ros2_node.py --ros-args \
        -p policy_host:=130.251.13.151 -p policy_port:=8000 \
        -p prompt:="reach the red square" \
        -p cmd_vel_topic:=/diff_cont/cmd_vel_unstamped -p cmd_vel_stamped:=false \
        -p actions_per_query:=1   # 1 = receding horizon (most reactive); 10 = run the whole chunk

Needs (on the node machine): rclpy, cv2, mss, numpy, openpi-client, an X display with RViz.
"""

import math
import os
import time
from typing import Optional

import cv2
import mss
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped

from openpi_client import websocket_client_policy as _wcp


# ---- must match the recorder ----
WIDTH, HEIGHT = 1920, 1200
SCREEN_CAPTURE_REGION = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class NavPolicyRunner(Node):
    def __init__(self):
        super().__init__("mir_nav_policy_runner")

        self.declare_parameter("odom_topic", "/diff_cont/odom")
        self.declare_parameter("cmd_vel_topic", "/diff_cont/cmd_vel_unstamped")
        self.declare_parameter("cmd_vel_stamped", False)
        self.declare_parameter("policy_host", "130.251.13.151")   # ws2 (use 127.0.0.1 if node runs on ws2)
        self.declare_parameter("policy_port", 8000)
        self.declare_parameter("prompt", "reach the red square")
        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("actions_per_query", 1)            # 1 = receding horizon (reactive)
        self.declare_parameter("send_image_size", 0)              # 0 = full res (matches training); >0 squashes

        gp = lambda n: self.get_parameter(n).get_parameter_value()
        self.odom_topic = gp("odom_topic").string_value
        self.cmd_vel_topic = gp("cmd_vel_topic").string_value
        self.cmd_vel_stamped = gp("cmd_vel_stamped").bool_value
        host = gp("policy_host").string_value
        port = int(gp("policy_port").integer_value)
        self.prompt = gp("prompt").string_value
        self.control_hz = float(gp("control_hz").double_value)
        self.actions_per_query = int(gp("actions_per_query").integer_value)
        self.send_image_size = int(gp("send_image_size").integer_value)

        self.latest_odom: Optional[Odometry] = None
        self.prev_cmd = np.zeros(3, dtype=np.float32)

        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 50)
        pub_type = TwistStamped if self.cmd_vel_stamped else Twist
        self.cmd_pub = self.create_publisher(pub_type, self.cmd_vel_topic, 10)

        self.get_logger().info(f"Connecting to policy server {host}:{port} ...")
        self.client = _wcp.WebsocketClientPolicy(host=host, port=port)
        self.get_logger().info(f"Connected. odom={self.odom_topic} cmd_vel={self.cmd_vel_topic} "
                               f"prompt='{self.prompt}' actions_per_query={self.actions_per_query}")

    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    def _build_state(self) -> Optional[np.ndarray]:
        if self.latest_odom is None:
            return None
        o = self.latest_odom
        return np.array([
            o.pose.pose.position.x, o.pose.pose.position.y,
            quaternion_to_yaw(o.pose.pose.orientation),
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z,
            float(self.prev_cmd[0]), float(self.prev_cmd[1]), float(self.prev_cmd[2]),
        ], dtype=np.float32)

    def _capture(self, sct) -> np.ndarray:
        raw = np.array(sct.grab(SCREEN_CAPTURE_REGION))
        img = cv2.cvtColor(cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2RGB)
        if img.shape[0] != HEIGHT or img.shape[1] != WIDTH:
            img = cv2.resize(img, (WIDTH, HEIGHT))
        if self.send_image_size > 0:
            img = cv2.resize(img, (self.send_image_size, self.send_image_size))
        return img.astype(np.uint8)

    def _publish_cmd(self, a: np.ndarray):
        if self.cmd_vel_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            t = msg.twist
        else:
            msg = Twist()
            t = msg
        t.linear.x = float(a[0]); t.linear.y = float(a[1]); t.angular.z = float(a[2])
        self.cmd_pub.publish(msg)
        self.prev_cmd = np.asarray(a[:3], dtype=np.float32)

    def run(self):
        period = 1.0 / self.control_hz
        while rclpy.ok() and self.latest_odom is None:
            self.get_logger().info("Waiting for odom...")
            rclpy.spin_once(self, timeout_sec=0.5)

        with mss.MSS(display=os.environ.get("DISPLAY", ":0")) as sct:
            while rclpy.ok():
                state = self._build_state()
                if state is None:
                    self.get_logger().warn("No odom yet, skipping inference...")
                    rclpy.spin_once(self, timeout_sec=0.05)
                    continue
                image = self._capture(sct)
                obs = {"observation/image": image, "observation/state": state, "prompt": self.prompt}
                try:
                    result = self.client.infer(obs)
                except Exception as e:
                    self.get_logger().error(f"policy infer failed: {e}")
                    self._publish_cmd(np.zeros(3, np.float32))
                    time.sleep(0.2)
                    continue

                actions = np.asarray(result["actions"], dtype=np.float32)  # (N, 3)
                if actions.ndim == 1:
                    actions = actions[None, :]
                n_exec = max(1, min(self.actions_per_query, actions.shape[0]))
                for k in range(n_exec):
                    if not rclpy.ok():
                        break
                    loop_start = time.time()
                    self._publish_cmd(actions[k])
                    rclpy.spin_once(self, timeout_sec=0.001)
                    sleep = period - (time.time() - loop_start)
                    if sleep > 0:
                        time.sleep(sleep)

    def stop(self):
        try:
            self._publish_cmd(np.zeros(3, np.float32))
        except Exception:
            pass


def main():
    rclpy.init()
    node = NavPolicyRunner()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
