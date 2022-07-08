da Vinci Research Kit (dVRK) in Isaac Gym
====================
This repository has codes to run dVRK in Isaac Gym with deformable objects. It follows my ICRA 2022 paper: https://sites.google.com/view/deformernet/home.  

# Installation and Documentation
* Install Isaac Gym: Carefully follow the official installation guide and documentation from Isaac Gym. You should be able to find the documentation on isaacgym/docs/index.html. Make sure you select NVIDIA GPU before running the examples.
* Set up:
```sh
# Step 1: Create a catkin workspace called dvrk_ws: http://wiki.ros.org/catkin/Tutorials/create_a_workspace

# Step 2: Clone this repo into the src folder
cd src
git clone https://github.com/Utah-ARMLab/dvrk_isaac

# Step 3: Clone other repos into the src folder:
git clone https://github.com/eric-wieser/ros_numpy.git
git clone https://github.com/gt-ros-pkg/hrl-kdl.git
git clone https://baotruyenthach@bitbucket.org/robot-learning/point_cloud_segmentation.git
```


# Steps to launch the simulation
```sh
## Step 1: Launch dvrk_isaac.launch. Consists of MoveIt, preshape, and PyKDL server.
roslaunch shape_servo_control dvrk_isaac.launch

## Step 2: Run Python script. Here are some examples:

# Collect data for DeformerNet
rosrun shape_servo_control box_collect_data_multi_shapes.py --data_recording_path /home/baothach/Documents/random_stuff --save_data True

# Collect random goal point clouds to evaluate DeformerNet
rosrun shape_servo_control collect_goals_box.py --save_data True

# Evaluate DeformerNet:
rosrun shape_servo_control evaluate_box.py --save_results True
```

# For first time user:
* Download box object dataset and provide the path to the `--object_meshes_path` flag. https://drive.google.com/drive/folders/1mlWV0hWaKhqZY7dxP8XIgD8dW6WXZoK-?usp=sharing
* To evaluate DeformerNet with my pre-collected goals, download the goals and provide the path to the `--data_recording_path` flag. https://drive.google.com/drive/folders/1trpjxR7OQRMzK762f74TwZ2F509yH5g1?usp=sharing


# Steps to run RL in Isaac Gym
* I highly recommend that you follow the RL examples and explanations in the official documentation before running my code. Also briefly look through the example codes. They are simple and easier to read while having the same structure as mine.
* If you want to create a new RL task, the official documentation has a section about how to do that.
* To run my code:
    * Step 1: 
Download the box object dataset and set the path to the `saved_initial_states_path` variable inside `shape_servo.py` file. https://drive.google.com/file/d/13GLpvsC_f-5WCuXe9pWShnDfY4d_2bXi/view?usp=sharing

    * Step 2: Run RL agent (with PPO algorithm). Example usage:
```sh
python3 train.py --task ShapeServo --flex --experiment 123 --pipeline cpu

```

# List of Packages:
* shape_servo_control **[maintained]**
  * Main files to launch simulations of the dVRK manipulating deformable objects
  * util files to create deformable objects' .tet mesh files and URDF files, which can then be used to launch the object into the simulation environment.
* dvrk_description **[maintained]**
  * URDF files & CAD models & meshes
* dvrk_moveit_config **[maintained]**
  * Moveit config files. Used to control dVRK motion.

