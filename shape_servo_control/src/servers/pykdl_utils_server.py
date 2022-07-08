#!/usr/bin/env python


from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
from shape_servo_control.srv import *
import copy


class PyKDLutils():
    '''
    Task velocity control.
    '''

    def __init__(self):
        rospy.init_node('PyKDL_server')
        self.robot = URDF.from_parameter_server()
        base_link = 'world'
        end_link = 'psm_tool_yaw_link'
        self.kdl_kin = KDLKinematics(self.robot, base_link, end_link)
       

    def create_get_pykdl_server(self):
        '''
        Create service to get info from pykdl_utils.
        '''
        rospy.Service('get_pykdl', PyKDL,
                      self.handle_get_pykdl)
        rospy.loginfo('Service get_pykdl:')
        rospy.loginfo('Ready to get info from pykdl_utils.')        

    def handle_get_pykdl(self, req):
        q_cur = req.q_cur # 8 joint positions of daVinci
        response = PyKDLResponse()
        jacobian = np.array(self.kdl_kin.jacobian(q_cur))    # Jacobian matrix of robot
        # rospy.loginfo("jacobian type: " + str(type(jacobian)))
        # rospy.loginfo("jacobian.flatten().shape: " + str(jacobian.flatten().shape))
        response.jacobian_shape = list(jacobian.shape)
        response.jacobian_flattened = list(jacobian.flatten())
        return response
        


if __name__ == '__main__':
    pykdl_info = PyKDLutils()
    pykdl_info.create_get_pykdl_server()
    rospy.spin()


 