import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.append("/home/baothach/dvrk_ws/src/dvrk_env/shape_servo_control/src/rlgpu/rl-pytorch/rl_pytorch/ppo")
from pointconv_util_groupnorm import PointConvDensitySetAbstraction


class DeformerNet(nn.Module):
    def __init__(self, out_dim, in_num_points, normal_channel=True):
        super(DeformerNet, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel=32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=64 + 3, mlp=[128], bandwidth = 0.4, group_all=True)   

        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.GroupNorm(1, 64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.GroupNorm(1, 32)
        self.fc5 = nn.Linear(32, out_dim)

        self.in_num_points = in_num_points

    def forward(self, xyz, xyz_goal):
        assert xyz.shape == xyz_goal.shape
        xyz = xyz.view(-1, 3, self.in_num_points)
        xyz_goal = xyz_goal.view(-1, 3, self.in_num_points)
        
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 128)


        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:,:3,:]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        g = l3_points.view(B, 128)

        x = g - x
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc5(x)

        return x 


