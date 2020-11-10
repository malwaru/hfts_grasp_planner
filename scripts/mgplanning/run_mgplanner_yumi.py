#!/usr/bin/python
import os
import time
import yaml
import argparse
import random
import numpy as np
import openravepy as orpy
import IPython
from hfts_grasp_planner.placement.anytime_planner import MGMotionPlanner
from hfts_grasp_planner.placement.goal_sampler.interfaces import PlacementGoalSampler
import hfts_grasp_planner.utils as hfts_utils

# def sample_arm_configs(manip, num):
#     robot = manip.GetRobot()
#     env = robot.GetEnv()
#     with robot:
#         robot.SetActiveDOFs(manip.GetArmIndices())
#         lower, upper = robot.GetActiveDOFLimits()
#         joint_range = upper - lower
#         samples = []
#         while len(samples) < num:
#             config = np.random.random(manip.GetArmDOF())
#             config = config * joint_range + lower
#             robot.SetActiveDOFValues(config)
#             if not env.CheckCollision(
#                     robot) and not robot.CheckSelfCollision():
#                 samples.append(config)
#     return samples


def show_goals(goals, manip, target_object):
    robot = manip.GetRobot()
    robot.SetActiveDOFs(manip.GetArmIndices())
    for goal in goals:
        robot.SetActiveDOFValues(goal.arm_config)
        hfts_utils.set_grasp(manip, target_object,
                             hfts_utils.inverse_transform(goal.grasp_tf),
                             goal.grasp_config)
        time.sleep(1.3)


def time_traj(robot, traj, vel_scale=0.4):
    """
        Retime the given trajectory.
    """
    with robot:
        vel_limits = robot.GetDOFVelocityLimits()
        robot.SetDOFVelocityLimits(vel_scale * vel_limits)
        orpy.planningutils.RetimeTrajectory(traj, hastimestamps=False)
        robot.SetDOFVelocityLimits(vel_limits)
    return traj


def show_traj(robot, target_object, traj, goal):
    # execute traj
    hfts_utils.set_grasp(goal.manip, target_object,
                         hfts_utils.inverse_transform(goal.grasp_tf),
                         goal.grasp_config)
    robot.GetController().SetPath(traj)
    robot.GetEnv().StartSimulation(0.025, True)
    robot.WaitForController(0)
    robot.GetEnv().StopSimulation()


def yaml_dict_to_tf_matrix(adict):
    """Convert a yaml representation of a tf matrix into a numpy tf matrix.

    Args:
        adict (dict): A dictionary with two items 'pos': [px, py, pz] and 'quat': [qx, qy, qz, qw]
            where 'pos' is the position and 'quat' the rotation of the tf.
    Returns:
        tf, np.array of shape (4, 4)
    """
    pos = np.array(adict['pos'])
    quat = np.array(adict['quat'])
    m = orpy.matrixFromQuat(quat)
    m[:3, 3] = pos
    return m


def load_goals(robot, target_object, filename):
    """Load goals from the given file.

    Args:
        robot (OpenRAVE Robot): robot to load goals for
        target_object (OpenRAVE KinBody): the target object
        filename (str): Filename to load goals from
    Returns:
        list of PlacementGoals
    """
    with open(filename, 'r') as goal_file:
        goal_data = yaml.load(goal_file)
    # first create a map from grasp id to grasp information
    grasp_infos = {}
    for grasp in goal_data['grasps']:
        grasp_infos[grasp['id']] = {
            'config': np.array(grasp['config']),
            'oTe': yaml_dict_to_tf_matrix(grasp['oTe']),
            'cost': grasp['cost'],
        }
    # now create the list of goals
    manip = robot.GetManipulator(goal_data['manipulator'])
    if manip is None:
        raise ValueError("Could not retrieve manipulator %s" %
                         goal_data['manipulator'])
    goals = []
    for goal in goal_data['goals']:
        gid = goal['grasp_id']
        grasp_info = grasp_infos[gid]
        # set grasp to compute object tf
        with robot.CreateKinBodyStateSaver():
            hfts_utils.set_grasp(
                manip, target_object,
                hfts_utils.inverse_transform(grasp_info['oTe']),
                grasp_info['config'])
            obj_tf = target_object.GetTransform()
        # create actual goal
        goals.append(
            PlacementGoalSampler.PlacementGoal(
                manip=manip,
                arm_config=goal['arm_config'],
                obj_tf=obj_tf,
                key=len(goals),
                objective_value=goal['objective_value'],
                grasp_tf=grasp_info['oTe'],
                grasp_config=grasp_info['config'],
                grasp_id=gid))
    return goals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script to run and test a multi-grasp motion planner on Yumi.")
    parser.add_argument('env_file',
                        help="Path to a OpenRAVE xml specifying the scene",
                        type=str)
    parser.add_argument(
        'target_obj_file',
        help='Path to an OpenRAVE xml describing the target Kinbody.',
        type=str)
    parser.add_argument('goal_file',
                        help="Path to a yaml file containing goals",
                        type=str)
    parser.add_argument('algorithm_name',
                        help="Name of the algorithm to use",
                        type=str)
    parser.add_argument('graph_name',
                        help="Name of the graph to use",
                        type=str)
    parser.add_argument('--lmbda',
                        help="Lambda, the tradeoff between path and goal cost",
                        type=float,
                        default=1.0)
    parser.add_argument(
        '--stats_file',
        help=
        "Filename to log calling stats (runtime, number of function calls) to.",
        type=str,
        default=None)
    parser.add_argument(
        '--results_file',
        help="Filename to log results (solution costs, ...) to.",
        type=str,
        default=None)
    parser.add_argument(
        '--planner_log',
        help="Basename to log planning logs (roadmap, evaluation) to.",
        type=str,
        default=None)
    parser.add_argument('--show_viewer',
                        help="Show viewer at the end of planning.",
                        action='store_true')
    args = parser.parse_args()

    # base_path = os.path.dirname(__file__)
    # env_file = os.path.normpath(
    #     base_path + '/../../models/environments/placement_exp_0.xml')
    env = orpy.Environment()
    orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
    # load scene and target
    with env:
        env.StopSimulation()
        env.Load(args.env_file)
        if not env.Load(args.target_obj_file):
            raise ValueError("Could not load target object. Aborting")
        # ensure the object has a useful name
        target_obj_name = "target_object"
        target_object = env.GetBodies()[-1]  # object that was loaded last
        target_object.SetName(target_obj_name)
        robot = env.GetRobots()[0]
        # load goals
        goals = load_goals(robot, target_object, args.goal_file)
        # goals = [goal for goal in goals if goal.objective_value > 0.7]
        manip = goals[0].manip
        robot.SetActiveDOFs(manip.GetArmIndices())
        start_config = robot.GetDOFValues()
    # create planner
    planner = MGMotionPlanner("%s;%s" % (args.algorithm_name, args.graph_name),
                              manip)
    planner.setup(target_object,
                  lmbda=args.lmbda,
                  log_file=args.planner_log,
                  batchsize=1000)
    planner.addGoals(goals)
    # plan
    trajectories, reached_goals = planner.plan(10.0)
    trajectories = [time_traj(robot, traj) for traj in trajectories]
    if args.stats_file:
        planner.save_stats(args.stats_file)
    if args.results_file:
        planner.save_solutions(args.results_file)
    # print "Found %d trajectories" % len(trajectories)
    if args.show_viewer:
        env.SetViewer('qtcoin')
        IPython.embed()
