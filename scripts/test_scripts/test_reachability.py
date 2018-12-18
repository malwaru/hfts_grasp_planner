#! /usr/bin/python

import openravepy as orpy
import numpy as np
import time
import os
import random
import hfts_grasp_planner.placement.reachability as reachability
import hfts_grasp_planner.ik_solver as ik_module
import IPython
import rospy


def create_map(env, robot, manip_name, urdf_file):
    robot.SetActiveManipulator(manip_name)
    manip = robot.GetActiveManipulator()
    ik_solver = ik_module.IKSolver(manip, urdf_file)
    rmap = reachability.SimpleReachabilityMap(manip, ik_solver)
    # env.SetViewer('qtcoin')
    filename = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/simple_reachability/' + robot.GetName() + '_' + \
        manip.GetName()
    rmap.create(0.05, 0)
    rmap.save(filename)
    print "Saved reachability map %s" % filename


def load_map(env, robot, manip_name, urdf_file):
    robot.SetActiveManipulator(manip_name)
    manip = robot.GetActiveManipulator()
    filename = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/simple_reachability/' + robot.GetName() + '_' + \
        manip.GetName() + '.npy'
    ik_solver = ik_module.IKSolver(manip, urdf_file)
    rmap = reachability.SimpleReachabilityMap(manip, ik_solver)
    rmap.load(filename)
    return rmap


if __name__ == '__main__':
    rospy.init_node("test_reachability", log_level=rospy.DEBUG)
    env = orpy.Environment()
    env_file = os.path.abspath(os.path.dirname(__file__)) + '/../../data/environments/cluttered_env_yumi.xml'
    env.Load(env_file)
    urdf_file = os.path.abspath(os.path.dirname(__file__)) + '/../../models/yumi/yumi.urdf'
    robot = env.GetRobots()[0]
    # create_map(env, robot, 'right_arm_with_gripper', urdf_file)
    # create_map(env, robot, 'left_arm_with_gripper', urdf_file)
    left_rmap = load_map(env, robot, 'left_arm_with_gripper', urdf_file)
    right_rmap = load_map(env, robot, 'right_arm_with_gripper', urdf_file)
    manip = robot.GetManipulator('left_arm_with_gripper')
    eeftf = manip.GetEndEffectorTransform()
    pose = np.empty(7)
    pose[:3] = eeftf[:3, 3]
    pose[3:] = orpy.quatFromRotationMatrix(eeftf[:3, :3])
    dist, rpose, rconfig = left_rmap.query(np.array([pose]))
    start_config = robot.GetDOFValues()
    print dist, rpose, rconfig
    print pose
    env.SetViewer('qtcoin')
    IPython.embed()
