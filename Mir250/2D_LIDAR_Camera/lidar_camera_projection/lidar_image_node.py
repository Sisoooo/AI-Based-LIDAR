import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2

from .utils.math import polar_to_xy


# Bird's-eye-view canvas parameters
IMAGE_SIZE   = 800          # pixels (square)
RANGE_MAX_M  = 15.0         # metres shown from centre to edge
M_PER_PIXEL  = RANGE_MAX_M / (IMAGE_SIZE / 2)  # metres represented by 1 px

# MIR250 body footprint (metres), drawn for spatial reference
ROBOT_W = 0.58 / 2          # half-width
ROBOT_L = 0.90 / 2          # half-length


def metres_to_pixel(x_m: float, y_m: float) -> tuple:
    """Convert robot-frame metric coordinates to canvas pixel coordinates.

    Robot +X  →  canvas up   (image row decreases)
    Robot +Y  →  canvas left (image col decreases)
    Origin is the canvas centre.
    """
    cx = IMAGE_SIZE // 2
    cy = IMAGE_SIZE // 2
    px = int(cx - y_m / M_PER_PIXEL)
    py = int(cy - x_m / M_PER_PIXEL)
    return px, py


class LiDARImageNode(Node):
    """Converts merged /scan LiDAR data to a bird's-eye-view image and
    publishes it on /lidar/image_raw while also displaying it locally."""

    def __init__(self):
        super().__init__('lidar_image_node')

        self.bridge = CvBridge()

        # Subscribers
        self.scan_subscriber_ = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 1)

        # Publisher
        self.image_publisher_ = self.create_publisher(Image, '/lidar/image_raw', 1)

        # Render timer (~33 Hz)
        self.timer_ = self.create_timer(0.03, self.render_callback)

        self.latest_scan: LaserScan | None = None

    # ------------------------------------------------------------------
    def scan_callback(self, scan_msg: LaserScan):
        self.latest_scan = scan_msg

    # ------------------------------------------------------------------
    def render_callback(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan

        # Dark canvas
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        # Draw range rings (every 3 m)
        ring_step_m = 3.0
        r_m = ring_step_m
        while r_m <= RANGE_MAX_M:
            r_px = int(r_m / M_PER_PIXEL)
            cv2.circle(canvas, (IMAGE_SIZE // 2, IMAGE_SIZE // 2), r_px,
                       (40, 40, 40), 1)
            label_px, label_py = metres_to_pixel(r_m, 0)
            cv2.putText(canvas, f'{int(r_m)}m', (label_px + 4, label_py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)
            r_m += ring_step_m

        # Draw cardinal axes
        cx = cy = IMAGE_SIZE // 2
        cv2.line(canvas, (cx, 0), (cx, IMAGE_SIZE - 1), (30, 30, 30), 1)
        cv2.line(canvas, (0, cy), (IMAGE_SIZE - 1, cy), (30, 30, 30), 1)

        # Draw robot footprint
        half_w_px = int(ROBOT_W / M_PER_PIXEL)
        half_l_px = int(ROBOT_L / M_PER_PIXEL)
        cv2.rectangle(canvas,
                      (cx - half_w_px, cy - half_l_px),
                      (cx + half_w_px, cy + half_l_px),
                      (0, 180, 0), 1)
        # Forward direction indicator
        fwd_tip = metres_to_pixel(ROBOT_L + 0.15, 0)
        cv2.arrowedLine(canvas, (cx, cy), fwd_tip, (0, 220, 0), 2,
                        tipLength=0.4)

        # Draw scan points
        angle = scan.angle_min
        for distance in scan.ranges:
            if math.isfinite(distance) and scan.range_min <= distance <= scan.range_max:
                x_m, y_m = polar_to_xy(angle, distance)
                px, py = metres_to_pixel(x_m, y_m)
                if 0 <= px < IMAGE_SIZE and 0 <= py < IMAGE_SIZE:
                    # Colour by distance: near=red, far=cyan
                    t = min(distance / RANGE_MAX_M, 1.0)
                    b = int(255 * t)
                    r = int(255 * (1.0 - t))
                    cv2.circle(canvas, (px, py), 2, (b, 0, r), -1)
            angle += scan.angle_increment

        # Label
        cv2.putText(canvas, 'LiDAR Bird-Eye View', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Publish
        ros_image = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'base_link'
        self.image_publisher_.publish(ros_image)

        # Local display
        cv2.imshow('LiDAR Bird-Eye View', canvas)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    lidar_image_node = LiDARImageNode()

    rclpy.spin(lidar_image_node)

    lidar_image_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
