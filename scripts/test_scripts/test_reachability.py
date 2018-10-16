#! /usr/bin/python

import openravepy as orpy
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
    ik_solver = ik_module.IKSolver(env, robot.GetName(), urdf_file)
    rmap = reachability.ReachabilityMap(manip, ik_solver)
    # env.SetViewer('qtcoin')
    filename = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/reachability/' + robot.GetName() + '_' + \
        manip.GetName()
    rmap.create(0.05, 0)
    rmap.save(filename)
    print "Saved reachability map %s" % filename


def load_map(env, robot, manip_name, urdf_file):
    robot.SetActiveManipulator(manip_name)
    manip = robot.GetActiveManipulator()
    filename = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/reachability/' + robot.GetName() + '_' + \
        manip.GetName() + '.npy'
    ik_solver = ik_module.IKSolver(env, robot.GetName(), urdf_file)
    rmap = reachability.ReachabilityMap(manip, ik_solver)
    rmap.load(filename)
    return rmap


if __name__ == '__main__':
    rospy.init_node("test_reachability", log_level=rospy.DEBUG)
    env = orpy.Environment()
    env_file = os.path.dirname(__file__) + '/../../data/environments/cluttered_env_yumi.xml'
    env.Load(env_file)
    urdf_file = os.path.dirname(__file__) + '/../../models/yumi/yumi.urdf'
    robot = env.GetRobots()[0]
    # create_map(env, robot, 'right_arm_with_gripper', urdf_file)
    # create_map(env, robot, 'left_arm_with_gripper', urdf_file)
    rmap = load_map(env, robot, 'left_arm_with_gripper', urdf_file)
    # env.SetViewer('qtcoin')
    IPython.embed()
