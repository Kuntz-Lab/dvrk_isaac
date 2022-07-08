from copy import deepcopy
import numpy as np
import os

from tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import math
from tasks.task_utils import *
import pickle
import rospy
from chamferdist import ChamferDistance
import random

import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
pkg_path = '/home/baothach/dvrk_ws/src/dvrk_env/shape_servo_control'
sys.path.append(pkg_path + '/src')
from core import Robot
from behaviors import MoveToPose, TaskVelocityControl

sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *


ROBOT_Z_OFFSET = 0.25
two_robot_offset = 0.86
class ShapeServo(BaseTask):

    """
    ShapeServo with PPO
    3 parts: pre_physics_step, simulate, and post_physics_step. 
    pre_physics_step should be implemented to perform any computations required before stepping the physics simulation. As an example, applying actions from the policy should happen in pre_physics_step. 
    simulate is then called to step the physics simulation. 
    post_physics_step should implement computations performed after stepping the physics simulation, e.g. computing rewards and observations.
    """
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
       
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.chamfer_success_threshold = 0.30 
        self.chamfer_fail_threshold = 2.5
        self.num_points = 1024
        self.total_num_goals = 1000
        
        # Load saved initial states of robot and deformable object
        saved_initial_states_path = '/home/baothach/shape_servo_data/RL_shapeservo/saved_init_states/box_init_states.pickle'
        with open(saved_initial_states_path, 'rb') as handle:
            data = pickle.load(handle)
            self.saved_obj_state = data["saved_obj_state"]
            self.saved_robot_state = data["saved_robot_state"]
            self.saved_frame_state = data["saved_frame_state"]
        
        # Load saved goal point clouds
        self.goal_data_path = "/home/baothach/shape_servo_data/RL_shapeservo/box/goal_data"

        # Shuffle and set up goal indices
        all_indices = list(np.arange(2990))
        random.seed(2021)
        self.goal_indices = random.sample(all_indices, len(all_indices)) 
        self.curr_idx = 0   # Need change

        # Cfg
        self.cfg["env"]["numObservations"] = self.num_points*3*2    # current pc and goal pc
        self.cfg["env"]["numActions"] = 3
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        super().__init__(cfg=self.cfg, enable_camera_sensors=True)

        # Others
        self.mtp_behavior = None
        self.finished_eps = False
        # print("sim_params:", self.sim_params.flex.num_inner_iterations)


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # dvrk asset
        dvrk_pose = gymapi.Transform()
        dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
        dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.001

        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.max_angular_velocity = 40000.

        asset_root = "/home/baothach/dvrk_ws/src/dvrk_env"
        dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
        print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
        dvrk_asset = self.gym.load_asset(self.sim, asset_root, dvrk_asset_file, asset_options)

        # Soft object asset
        asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
        soft_asset_file = "box.urdf"
        soft_pose = gymapi.Transform()
        soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.01818)
        soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
        soft_thickness = 0.0005 

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.thickness = soft_thickness
        asset_options.disable_gravity = True
        soft_asset = self.gym.load_asset(self.sim, asset_root, soft_asset_file, asset_options)


        # Set dvrk dof properties
        dof_props = self.gym.get_asset_dof_properties(dvrk_asset)
        self.vel_limits = dof_props['velocity']
        # print("====vel limits:", self.vel_limits)
   
        # Set up the env grid
        spacing = 0.0
        num_envs = 1
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(num_envs))


        self.dvrks = []
        self.envs = []
        self.envs_obj = []
        self.object_handles = []
        for i in range(self.num_envs):
            # Create env instance
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)



            dvrk_handle = self.gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)
            self.gym.set_actor_dof_properties(env, dvrk_handle, dof_props)

            # Add soft object                
            env_obj = env                     
            soft_actor = self.gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)            
            self.object_handles.append(soft_actor)                  
            self.envs.append(env)
            self.envs_obj.append(env_obj)
            self.dvrks.append(dvrk_handle)        

        # Create a Robot object for easy usage
        self.robot = Robot(self.gym, self.sim, self.envs[0], self.dvrks[0])
        
        # Set up camera
        cam_width = 256
        cam_height = 256    
        cam_pos = gymapi.Vec3(0.1, -0.5, 0.2)
        cam_target = gymapi.Vec3(0.0, -0.45, 0.00)    
        self.cam_handle, self.cam_prop = setup_cam(self.gym, self.envs[0], cam_width, cam_height, cam_pos, cam_target)
        

        self.reset()        


    def compute_reward(self):
        """ Compute rewards and check contact between robot and obj"""
        
        # Check whether robot has lost contact with the soft object
        contacts = [contact[4] for contact in self.gym.get_soft_contacts(self.sim)]
        lose_contact = not(9 in contacts and 10 in contacts)
        # print("===lose contacts: ", lose_contact)
        
        self.rew_buf[:], self.reset_buf[:] = compute_shape_servo_reward(
            self.reset_buf, self.pc, self.progress_buf, self.max_episode_length, self.pc_goal,
            self.chamfer_success_threshold, self.chamfer_fail_threshold, lose_contact
        )

    def compute_observations(self):
        """ Obtain point cloud of object from camera """
       
        pc = get_partial_point_cloud(self.gym, self.sim, self.envs[0], self.cam_handle, self.cam_prop)
        farthest_indices,_ = farthest_point_sampling(pc, self.num_points)
        self.pc = torch.from_numpy(pc[farthest_indices])
        self.obs_buf = torch.cat((self.pc.permute(0,2,1).flatten(start_dim=1), self.pc_goal.permute(0,2,1).flatten(start_dim=1)), 1)

        return self.obs_buf.to(self.device)

    def pre_physics_step(self, actions, num_transitions):
        
        self.step_count += 1

        if num_transitions % 10 == 0 or self.mtp_behavior is None:

            act = actions.squeeze().cpu().numpy()/30.
            ee_pose = self.robot.get_ee_cartesian_position() 
            new_pose = [-ee_pose[0] + act[0], -ee_pose[1] + act[1], 
                        ee_pose[2] - ROBOT_Z_OFFSET + act[2]]       # Robot frame      
            new_pose = np.clip(new_pose, self.low_ee_lim, self.high_ee_lim)  # clamp to lims
            target_pose = list(new_pose) + [0, 0.707107, 0.707107, 0]            
            self.mtp_behavior = MoveToPose(target_pose, self.robot, self.sim_params.dt, 10*self.sim_params.dt, open_gripper=False)
        
        if not self.mtp_behavior.is_complete_failure():
            action = self.mtp_behavior.get_action()
            if action is not None:
                self.gym.set_actor_dof_position_targets(self.robot.env_handle, self.robot.robot_handle, action.get_joint_position())      
        else:
            self.mtp_behavior = None

                              

    def post_physics_step(self, compute_rew_obs=True):        
        if self.reset_buf[0] == 1:
            self.reset()
            self.finished_eps = True
        if compute_rew_obs:
            self.progress_buf += 1
            self.compute_observations()
            self.compute_reward()


    def reset(self):

        print("========================RESETING===========================")

        self.gym.set_actor_rigid_body_states(self.envs[0], self.dvrks[0], self.saved_robot_state, gymapi.STATE_ALL) 
        self.gym.set_particle_state_tensor(self.sim, gymtorch.unwrap_tensor(self.saved_obj_state))        

        self.gym.set_joint_target_position(self.envs[0], self.gym.get_joint_handle(self.envs[0], "dvrk", "psm_tool_gripper1_joint"), 0.35)
        self.gym.set_joint_target_position(self.envs[0], self.gym.get_joint_handle(self.envs[0], "dvrk", "psm_tool_gripper2_joint"), -0.35) 

        self.init_ee_pose = self.robot.get_ee_cartesian_position() 
        self.init_ee_pose[:2] = -self.init_ee_pose[:2]  # Robot frame
        self.init_ee_pose[2] -= ROBOT_Z_OFFSET

        self.low_ee_lim =  self.init_ee_pose[:3] + np.array([-0.1, 0, 0])
        self.high_ee_lim =  self.init_ee_pose[:3] + np.array([0.1, 0.1, 0.1])

        
        self.progress_buf[0] = 0         
        self.reset_buf[0] = 0
        self.step_count = 0
        
        # print("******* curr_idx, goal_idx:", self.curr_idx, self.goal_indices[self.curr_idx])
        with open(os.path.join(self.goal_data_path, "sample " + str(self.goal_indices[self.curr_idx]) + ".pickle"), 'rb') as handle:
            pc_goal = pickle.load(handle)["partial pc"]
            farthest_indices,_ = farthest_point_sampling(pc_goal, self.num_points)
            self.pc_goal = torch.from_numpy(pc_goal[farthest_indices])      #size (1,P,3)


        self.curr_idx += 1
        if self.curr_idx >= len(self.goal_indices):
            self.curr_idx = 0




def compute_shape_servo_reward(reset_buf, current_pc, progress_buf, max_episode_length, pc_goal, 
    chamfer_success_threshold, chamfer_fail_threshold, lose_contact):
 
    chamferDist = ChamferDistance()
    chamfer_dist = np.sqrt(chamferDist(pc_goal, current_pc).item())  # chamfer distance computed using chamferdist package is squared
    # print("**Chamfer dist, eps:", chamfer_dist, progress_buf[0])
    rewards = - torch.tensor([chamfer_dist])

    if chamfer_dist >= chamfer_fail_threshold or chamfer_dist <= chamfer_success_threshold or \
        progress_buf >= max_episode_length or lose_contact:
        reset_buf = torch.tensor([1])
    else:
        reset_buf = torch.tensor([0])

    return rewards, reset_buf


'''
paramters needed to change:
  chamfer_success_threshold
  noptepochs: 20
  nsteps
  nminibatches 
  mini_batch_size
  num_eps_per_train_iteration
  episodeLength
  experiment name
  is_testing
'''