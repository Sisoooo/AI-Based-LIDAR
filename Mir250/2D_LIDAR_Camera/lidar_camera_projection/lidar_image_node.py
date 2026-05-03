import math
import os
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Output image dimensions
IMAGE_W = 800
IMAGE_H = 480

# Raycaster scale: wall height in pixels when an obstacle is exactly 1 m away
WALL_SCALE = 300.0

# Horizontal field of view centred on robot forward (+X), in radians.
# 120 deg matches a wide-angle camera; reduce for a narrower "zoom" effect.
FOV_RAD = math.radians(120.0)

# Distances beyond this are treated as open space (no wall drawn)
RANGE_MAX_M = 15.0


class LiDARImageNode(Node):
    """Renders a raycaster-style first-person perspective image from 2D LiDAR
    data (similar to a camera feed) and publishes it on /lidar/image_raw."""

    def __init__(self):
        super().__init__('lidar_image_node')

        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.scan_subscriber_ = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, sensor_qos)

        self.image_publisher_ = self.create_publisher(Image, '/lidar/image_raw', 1)

        # Render timer (~33 Hz)
        self.timer_ = self.create_timer(0.03, self.render_callback)

        self.latest_scan: LaserScan | None = None

        # PNG saving parameters
        self.declare_parameter('save_images', False)
        self.declare_parameter('output_dir', os.path.expanduser('~/Desktop/AI-Based-LIDAR/Images_maze'))
        self._frame_index = 0

    # ------------------------------------------------------------------
    def scan_callback(self, scan_msg: LaserScan):
        self.latest_scan = scan_msg

    # ------------------------------------------------------------------
    def render_callback(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan
        half_h = IMAGE_H // 2

        # --- Background: sky gradient (top) + floor gradient (bottom) ---
        canvas = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        for row in range(half_h):
            t = row / max(half_h - 1, 1)                       # 0=top, 1=horizon
            canvas[row, :] = (int(55 + 25 * t),                # B – deep blue
                              int(25 + 15 * t),                # G
                              int(5  +  5 * t))                # R
        for row in range(half_h, IMAGE_H):
            t = (IMAGE_H - row) / max(half_h, 1)               # 0=bottom, 1=horizon
            v = int(15 + 25 * t)
            canvas[row, :] = (v, v, v)                         # dark grey floor

        # --- Pre-compute angle → scan-index mapping for every column ---
        half_fov = FOV_RAD / 2.0
        # Left column = +half_fov (robot left), right column = -half_fov (robot right)
        col_angles = half_fov - np.arange(IMAGE_W) * (FOV_RAD / IMAGE_W)
        col_indices = np.round(
            (col_angles - scan.angle_min) / scan.angle_increment
        ).astype(int)
        ranges_arr = np.array(scan.ranges, dtype=np.float32)
        n_rays = len(ranges_arr)

        # --- Draw one vertical wall slice per column ---
        for col in range(IMAGE_W):
            idx = col_indices[col]
            if idx < 0 or idx >= n_rays:
                continue

            distance = float(ranges_arr[idx])
            if not math.isfinite(distance) or distance < scan.range_min:
                continue
            distance = min(distance, RANGE_MAX_M)

            # Wall height: taller = closer
            wall_h = min(int(WALL_SCALE / distance), IMAGE_H)
            top    = half_h - wall_h // 2
            bottom = half_h + wall_h // 2

            # Colour gradient: near=warm red-orange, far=cool blue-grey
            t = distance / RANGE_MAX_M          # 0=near, 1=far
            r = int(220 * (1.0 - t) +  60 * t)
            g = int(100 * (1.0 - t) +  80 * t)
            b = int( 40 * (1.0 - t) + 180 * t)

            # Edge darkening: centre columns slightly brighter (simulated lighting)
            shade = 0.75 + 0.25 * math.cos((col / IMAGE_W - 0.5) * math.pi)
            color = (int(b * shade), int(g * shade), int(r * shade))

            cv2.line(canvas, (col, top), (col, bottom), color, 1)

        # --- HUD overlay ---
        cv2.putText(canvas, 'LiDAR First-Person View', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(canvas,
                    f'FOV: {int(math.degrees(FOV_RAD))}\xb0   max: {RANGE_MAX_M:.0f} m',
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # --- Publish ---
        ros_image = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'base_link'
        self.image_publisher_.publish(ros_image)

        # --- Local display ---
        cv2.imshow('LiDAR First-Person View', canvas)
        cv2.waitKey(1)

        # --- Save PNG if enabled ---
        if self.get_parameter('save_images').get_parameter_value().bool_value:
            out_dir = self.get_parameter('output_dir').get_parameter_value().string_value
            os.makedirs(out_dir, exist_ok=True)
            filename = os.path.join(out_dir, f'lidar_{self._frame_index:06d}.png')
            cv2.imwrite(filename, canvas)
            self._frame_index += 1


def main(args=None):
    rclpy.init(args=args)

    lidar_image_node = LiDARImageNode()

    rclpy.spin(lidar_image_node)

    lidar_image_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
