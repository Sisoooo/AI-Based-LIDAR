#!/usr/bin/env python3
"""
Autonomous random navigation node for MiR250.

On startup, subscribes to /map once to discover every free cell in the
currently loaded map. Each iteration then:
  1. Picks a random free cell as the navigation goal
  2. Sends it to Nav2
  3. Waits for the result
  4. Waits the configured cooldown (default 3 s)
  5. Repeats until keyboard interrupt

Usage:
    python3 mir_random_nav.py
    python3 mir_random_nav.py --cooldown 5.0
"""

import argparse
import math
import random
import time

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid


def fetch_free_cells(node: Node, map_topic: str, margin_m:float) -> list:
    """Block until one /map message arrives; return list of free (x, y) world coords."""
    free_cells = []
    done = [False]

    def on_map(msg: OccupancyGrid):
        if done[0]:
            return
        done[0] = True
        info = msg.info
        res = info.resolution

        margin_cells = int(math.ceil(margin_m/res))
        too_close = set()
        for idx, val in enumerate(data):
            if val != 0:
                col idx % width
                row = idx // width
                for dr in range(-margin_cells, margin_cells + 1):
                    for dc in range(-margin_cells, margin_cells + 1):
                        if dr * dr + dc * dc <= margin_cells * margin_cells:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                too_close.add(nr * width + nc)

        ox = info.origin.position.x
        oy = info.origin.position.y
        for idx, val in enumerate(data):
            if val == 0 and idx not in too_close:
                col = idx % width
                row = idx // width
                free_cells.append((
                    ox + (col + 0.5) * res,
                    oy + (row + 0.5) * res,
                ))
        node.get_logger().info(
            f'Map received: {width}x{height} cells @ {res:.3f} m/cell — '
            f'{len(free_cells)} safe cells (margin={margin_m}).'
        )

    sub = node.create_subscription(OccupancyGrid, '/map', on_map, 1)
    node.get_logger().info('Waiting for /map topic...')
    while rclpy.ok() and not done[0]:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_subscription(sub)
    return free_cells


def generate_waypoint(free_cells: list) -> tuple:
    """Pick a random free cell and assign a random yaw."""
    x, y = random.choice(free_cells)
    yaw = random.uniform(-180.0, 180.0)
    return x, y, yaw


def make_pose(navigator: BasicNavigator, x: float, y: float, yaw_deg: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    half = math.radians(yaw_deg) / 2.0
    pose.pose.orientation.z = math.sin(half)
    pose.pose.orientation.w = math.cos(half)
    return pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cooldown', type=float, default=3.0,
                        help='Seconds to wait after each goal (default: 3.0)')
    parser.add_argument('--margin', type=float, default=30,
                        help='Min distance from obstacles for goal cells, metres (default: 5)')
    parser.add_argument('--map_topic', type=str, default='/map',
                        help='Map topic name (default: /map)')
    args = parser.parse_args()

    rclpy.init()

    # Read the map before starting the navigator
    map_node = rclpy.create_node('map_reader')
    free_cells = fetch_free_cells(map_node, args.map_topic, args.margin)
    map_node.destroy_node()

    if not free_cells:
        print('ERROR: No free cells found in /map. Is the map server running?')
        rclpy.shutdown()
        return

    navigator = BasicNavigator()
    navigator.setInitialPose(make_pose(navigator, 0.0, 0.0, 0.0))
    navigator.waitUntilNav2Active()

    print(f'Nav2 active. {len(free_cells)} candidate cells loaded. '
          f'Cooldown: {args.cooldown}s.')

    goal_count = 0
    try:
        while rclpy.ok():
            x, y, yaw = generate_waypoint(free_cells)
            goal_count += 1

            goal = make_pose(navigator, x, y, yaw)
            print(f'Goal #{goal_count}: ({x:.2f}, {y:.2f}, {yaw:.1f}°)')
            navigator.goToPose(goal)

            while not navigator.isTaskComplete():
                feedback = navigator.getFeedback()
                if feedback:
                    print(f'  Distance remaining: {feedback.distance_remaining:.2f} m', end='\r')

            result = navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                print(f'\nGoal #{goal_count} succeeded.')
            elif result == TaskResult.CANCELED:
                print(f'\nGoal #{goal_count} was canceled.')
            elif result == TaskResult.FAILED:
                print(f'\nGoal #{goal_count} failed.')

            print(f'Waiting {args.cooldown}s...')
            time.sleep(args.cooldown)

    except KeyboardInterrupt:
        print('\nShutting down.')
        navigator.cancelTask()

    navigator.lifecycleShutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
