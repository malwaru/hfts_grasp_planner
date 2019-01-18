#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.robot as robot_sdf_module
import openravepy as orpy
import numpy as np
import argparse

ROBOT_FILE = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/models/yumi/yumi.xml'
ROBOT_BALL_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/models/yumi/ball_description.yaml'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a ball approximation of a robot.")
    parser.add_argument('robot_file', type=str, help='Path to an OpenRAVE robot')
    parser.add_argument('ball_file', type=str, help='Path ball description')
    args = parser.parse_args()

    env = orpy.Environment()
    env.Load(args.robot_file)
    robot = env.GetRobots()[0]
    env.SetViewer('qtcoin')
    robot_sdf = robot_sdf_module.RobotBallApproximation(robot, args.ball_file)
    robot_sdf.visualize_balls()
    IPython.embed()
