## Subgroup A3: AI-Based 2D LiDAR Navigation  

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

-> https://github.com/Rudresh172/mir250_robot_ros2

## How to run:
Three phases: dataset creation, model training, inference testing

# Dataset creation
Set of operations that allows screen recording of a preset Rviz configuration in a set of episodes in leRobot format.
Six terminals needed. Necessary to run commands following the next list's order.
Where required, change <world_name> with the map of your choosing  

0. In terminal 1, build the environment
```
cd AI-Based-LIDAR
CMAKE_POLICY_VERSION_MINIMUM=3.5 colcon build && source install/setup.bash
```
The policy version is specified due to compatibility issues when building from scratch.
Run the same command without the policy version in all other terminals.

1. Terminal 1 (same as before): Launch Gazebo and Rviz environments
```
ros2 launch mir_gazebo mir_gazebo_launch.py world:=<world_name> rviz_config_file:=$(ros2 pkg prefix mir_navigation)/share/mir_navigation/rviz/vla_mir_nav_TD.rviz
```
This command will launch both Gazebo and Rviz, although only the latter is necessary for the dataset creation.
In Rviz, go in the Panels menu in the top left and disable all the extra panels, since the training must be done on full screen. 

2. Terminal 2: Localization node
```
ros2 launch mir_navigation amcl.py use_sim_time:=true map:=$(ros2 pkg prefix mir_navigation)/share/mir_navigation/maps/<world_name>.yaml
```

3. Terminal 3: Navigation node 
```
ros2 launch mir_navigation navigation.py use_sim_time:=true
```
Terminal 2 and 3 will return warning messages when run by themselves, this is normal and will be fixed with the last command

4. Terminal 4: Dataset recorder node
```
ros2 run mir_navigation leRobotDatasetRecorder.py --ros-args -p world:=<world_name>
```
Sets up the recorder. Note that if episodes for a certain map have already been recorded and are present in the same path pointed by the node a FileExists error will show up. To avoid this, datasets should be either moved or deleted before new attempts.

5. Terminal 5: Random goal publishing 
```
ros2 run mir_navigation mir_random_nav.py
```
Node for random goal generation, it fixes terminal 2 by passing the robot's initial position on /PoseStampede, which consequentially fixes terminal 3.


Now the Rviz simulation is running by itself and the screen recording is being done in the background. Be aware that the screen must see Rviz in its entirety, so other tabs should remain closed or iconized.
The recorder will cap at 100 episodes per map and autoclose when reaching that point; after that, there is no point in keeping the simulation running. After that, run the following commands to form a complete leRobot v2.1 dataset with all the gathered episodes. 


6. Running dataset merger
```
cd Mir250/mir_navigation
python3 merge_rviz_dataset.py --data_dir ~/lerobot_ros2_rviz_dataset
```

The outputed merged dataset will then be ready for training!


# Model training

Use the dataset created in the previous phase to train a model. In the current setting, the pi0.5 model has been trained with a dataset of 267 episodes in three maps.


# Inference testing

Once the model has been trained, it is possible to test its functionalities since the model publishes cmd_vel directly, wihtout using Nav2. To test this, five terminal will be needed, of which terminal 1 and 2 must launch the gazebo environment and the localization node as done in the recording phase. This is done because the model is able to see the screen as in the recording phase, and as such it aims to publish directly the commands to move the robot.

After running the Rviz environment and the localization node, the following commands must be run in order and in separate terminals (assuming AI-Based-LIDAR being already built):

3. Terminal 3: Goal monitoring node 
```
cd AI-Based-LIDAR/Mir250/mir_navigation
python3 goal_monitor_node.py --ros-args -p cmd_vel_topic:=/diff_cont/cmd_vel_unstamped
```

This node initializes the logic that controls if the goal has been reached, along with actually publishing the marker.
The node also sends the robot's initial position to /PoseStampede, bypassing the manual publication.

3. Terminal 4: Inference node 
```
cd AI-Based-LIDAR/Mir250/mir_navigation
python3 inference_ros2_node.py --ros-args \
        -p policy_host:=130.251.13.151 -p policy_port:=8000 \
        -p prompt:="reach the red square" \
        -p cmd_vel_topic:=/diff_cont/cmd_vel_unstamped -p cmd_vel_stamped:=false \
        -p actions_per_query:=1 
```

Running inference node, change the prompt to associate new tasks with colors and shapes among the allowed ones

4. Terminal 5: Prompt
```
cd AI-Based-LIDAR/Mir250/mir_navigation
ros2 topic pub --once /inference_prompt std_msgs/String "{data: 'reach the red square'}"
```
Publishes a single prompt to the inference topic to let the robot start moving.

