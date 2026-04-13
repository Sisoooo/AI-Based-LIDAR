# Subgroup A3: AI-Based 2D LiDAR Navigation  

Assignment A3: VLA-Inspired 2D Navigation System (SIMULATION)

What to do: Create novel navigation using 2D LiDAR data as "images" for AI-based planning
1) Convert 2D LiDAR scans to image-like representations
2) Research VLA (Vision-Language-Action) approaches for navigation
3) Implement learning-based local planner
4) Compare against standard Nav2 DWB planner

Software needed: ROS2 Humble, PyTorch/TensorFlow, OpenCV, Nav2 interfaces

Research needed: VLA papers, learning-based navigation, LiDAR-to-image conversion

Deliverables: Custom AI ROS2 node, trained network model, comparison study report, novel approach documentation

 # Starting point:
1. Understand Nav2 (The Baseline)
- Run Nav2 with TurtleBot3 in Gazebo
- Understand the pipeline: LaserScan -> Costmap -> DWB Planner -> cmd_vel
- Learn what DWB does: samples trajectories, scores them with critics, picks best
- This is what you're comparing against (know it inside out)
2. Study VLA (The Innovation)
- Read π0 (Pi-Zero): https://www.pi.website/blog/pi0 (Vision-Language-Action model)
- Read RT-2 by Google DeepMind (how VLMs output robot actions)
- Key insight: π0 uses camera images -> you replace them with LiDAR-as-image (BEV/polar)
- Understand: Vision = LiDAR image, Language = Goal position, Action = cmd_vel
