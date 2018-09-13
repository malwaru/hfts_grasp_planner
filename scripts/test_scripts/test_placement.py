#! /usr/bin/python

import os
import yaml
import rospy
import argparse
import IPython
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf_module
# import hfts_grasp_planner.sdf.robot as robot_sdf_module
# import hfts_grasp_planner.sdf.costs as costs_module
import openravepy as orpy
import numpy as np
import hfts_grasp_planner.placement.placement_planning as pp_module
from hfts_grasp_planner.utils import ObjectFileIO, is_dynamic_body

ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/cluttered_env.xml'
SDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/cluttered_test_env.sdf'
DATA_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data'
# ROBOT_BALL_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/r850_robotiq/ball_description.yaml'


def draw_volume(env, volume):
    return env.drawbox(0.5 * (volume[0] + volume[1]), 0.5 * (volume[1] - volume[0]), np.array([0.3, 0.3, 0.3, 0.3]))


def execute_placement_planner(placement_planner, body):
    result = placement_planner.sample(placement_planner.get_max_depth())
    # print 'Best val is ', result.quality_value
    if result.is_valid():
        if result.is_goal():
            print "Result is a goal."
        else:
            print "Result is valid, but not a goal."
    else:
        print "Result is invalid"
    if result.obj_pose is not None:
        body.SetTransform(result.obj_pose)
    else:
        print "Failed to find a solution"
    return result


def resolve_paths(problem_desc, yaml_file):
    global_yaml = str(yaml_file)
    if not os.path.isabs(global_yaml):
        cwd = os.getcwd()
        global_yaml = cwd + '/' + global_yaml
    head, _ = os.path.split(global_yaml)
    for key in ['or_env', 'sdf_file', 'urdf_file', 'data_path', 'gripper_file', 'grasp_file']:
        if key in problem_desc:
            problem_desc[key] = os.path.normpath(head + '/' + problem_desc[key])


if __name__ == "__main__":
    # NOTE If the OpenRAVE viewer is created too early, nothing works! Collision checks may be incorrect!
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_desc', help="Path to a yaml file specifying what world, robot to use etc.", type=str)
    parser.add_argument('--debug', help="If provided, run in debug mode", action="store_true")
    parser.add_argument('--show_plcmnt_volume', help="If provided, visualize placement volume", action="store_true")
    parser.add_argument('--show_sdf_volume', help="If provided, visualize sdf volume", action="store_true")
    parser.add_argument('--show_sdf', help="If provided, visualize sdf", action="store_true")
    args = parser.parse_args()
    rospy.init_node("TestPlacement", anonymous=True, log_level=rospy.DEBUG)
    with open(args.problem_desc, 'r') as f:
        problem_desc = yaml.load(f)
        resolve_paths(problem_desc, args.problem_desc)

    env = orpy.Environment()
    env.Load(problem_desc['or_env'])
    dynamic_bodies = [body for body in env.GetBodies() if is_dynamic_body(body)]
    scene_sdf = sdf_module.SceneSDF(env, [], excluded_bodies=dynamic_bodies)
    if os.path.exists(problem_desc['sdf_file']):
        scene_sdf.load(problem_desc['sdf_file'])
    else:
        sdf_volume_robot = problem_desc['sdf_volume']
        robot = env.GetRobots(problem_desc['robot_name'])
        # transform sdf volume to world frame
        robot_tf = robot.GetTransform()
        tvals = np.array([np.dot(robot_tf[:3, :3], sdf_volume_robot[:3]) + robot_tf[:3, 3],
                          np.dot(robot_tf[:3, :3], sdf_volume_robot[3:]) + robot_tf[:3, 3]])
        sdf_volume = np.array([np.min(tvals, axis=0), np.max(tvals, axis=0)]).flatten()
        handle = draw_volume(env, (sdf_volume[:3], sdf_volume[3:]))
        resolution = 0.005
        print 'Check volume!'
        IPython.embed()
        scene_sdf.create_sdf(sdf_volume, resolution, resolution)
        scene_sdf.save(problem_desc['sdf_file'])
    target_obj_name = problem_desc['target_name']
    placement_planner = pp_module.PlacementGoalPlanner(problem_desc['data_path'], env, scene_sdf)
    # placement_volume = (np.array([-0.35, 0.55, 0.66]), np.array([0.53, 0.9, 0.77]))  # on top of shelf
    # placement_volume = (np.array([-0.35, 0.55, 0.42]), np.array([0.53, 0.9, 0.77]))  # all of the shelf
    # placement_volume = (np.array([-0.35, 0.55, 0.42]), np.array([0.53, 0.9, 0.64]))  # inside shelf
    # placement_volume = (np.array([-0.35, 0.4, 0.42]), np.array([0.53, 0.55, 0.64]))  # front of shelf
    # placement_volume = (np.array([-0.35, 0.55, 0.44]), np.array([0.53, 0.9, 0.49]))  # bottom inside shelf
    # placement_volume = (np.array([0.24, 0.58, 0.51]), np.array([0.29, 0.8, 0.55]))  # volume in small gap
    placement_volume = (np.array(problem_desc['plcmnt_volume'][:3]), np.array(problem_desc['plcmnt_volume'][3:]))
    placement_planner.set_placement_volume(placement_volume)
    placement_planner.set_object(target_obj_name)
    body = env.GetKinBody(target_obj_name)
    # env.SetViewer('qtcoin')
    placement_planner._placement_heuristic._env.SetViewer('qtcoin')
    handle = draw_volume(env, placement_volume)
    print "Check the placement volume!", placement_volume
    IPython.embed()
    handle = None
    execute_placement_planner(placement_planner, body)
    IPython.embed()
