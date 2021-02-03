#!/usr/bin/python
import os
import time
import numpy as np
import openravepy as orpy
import IPython
import random
from hfts_grasp_planner.placement.anytime_planner import MGMotionPlanner
import hfts_grasp_planner.utils as hfts_utils


class SimpleGoal(object):
    grasp_config = np.array([0.5])
    grasp_tf = np.eye(4)
    arm_config = None
    grasp_id = 0
    key = 0


def sample_arm_configs(manip, num):
    robot = manip.GetRobot()
    env = robot.GetEnv()
    with robot:
        robot.SetActiveDOFs(manip.GetArmIndices())
        lower, upper = robot.GetActiveDOFLimits()
        joint_range = upper - lower
        samples = []
        while len(samples) < num:
            config = np.random.random(manip.GetArmDOF())
            config = config * joint_range + lower
            robot.SetActiveDOFValues(config)
            if not env.CheckCollision(
                    robot) and not robot.CheckSelfCollision():
                samples.append(config)
    return samples


def show_goals(goals, manip, target_object):
    robot = manip.GetRobot()
    robot.SetActiveDOFs(manip.GetArmIndices())
    for goal in goals:
        robot.SetActiveDOFValues(goal.arm_config)
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


def show_traj(robot, traj):
    # execute traj
    robot.GetController().SetPath(traj)
    robot.WaitForController(0)


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    env_file = os.path.normpath(
        base_path + '/../../models/environments/placement_exp_0.xml')
    env = orpy.Environment()
    env.Load(env_file)
    orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
    # orpy.RaveLoadPlugin("ormultigraspmp")
    # planner = orpy.RaveCreatePlanner(env, "ParallelMGBiRRT")
    # module = orpy.RaveCreateModule(env, "SequentialMGBiRRT")
    robot = env.GetRobots()[0]
    manip = robot.GetActiveManipulator()
    # set goal
    target_object = env.GetKinBody("crayola")
    gpose = [
        -0.17763906, -0.00195599, 0.015208, 0.48656468, 0.50384748, 0.53856192,
        0.46834132
    ]
    SimpleGoal.grasp_tf = orpy.matrixFromQuat(gpose[3:])
    SimpleGoal.grasp_tf[:3, 3] = gpose[:3]
    SimpleGoal.grasp_config = np.array([0.0184])
    hfts_utils.set_grasp(manip, target_object,
                         hfts_utils.inverse_transform(SimpleGoal.grasp_tf),
                         SimpleGoal.grasp_config)

    start_config = robot.GetDOFValues()
    # planner.addGoals([goal])
    # ids, trajs = planner.plan(0.5)

    # arm_configs = np.array(
    #     [[1.25258009, -1.93622083, -1.77695288,  1.01818061, -4.83087776, -1.03938177, -0.82435254],
    #      [1.11597062, -0.28492879,  2.78850779, -0.87283519, -3.15051852, 1.32253751,  2.97237394],
    #      [1.00281921,  0.51366662, -2.16203263, -2.13098839,  5.03660588, -1.26111986,  3.91137734],
    #      [-0.85163025, -1.4762257,  0.69984902, -0.69804897,  4.2517744, -0.37408561, -3.3408329],
    #      [-0.10277229, -2.37251086,  0.61479298, -1.43631191, -4.38698001, 0.02923189,  0.87474853],
    #      [-0.19612599, -1.33136467, -1.86323676,  0.61238113,  3.25175716, 1.4431787,  1.05532095],
    #      [-1.51859361, -0.61244252, -1.4749193, -1.59033694, -0.28297413, 0.78712868, -1.53804721],
    #      [-1.70009506, -0.18605498, -1.56330288, -0.27473375, -0.00539935, 1.85570684, -3.71383803],
    #      [0.11127244, -0.65327576,  2.4378246, -1.93000283, -5.02064425, 2.10443872,  2.55133075],
    #      [-0.17231944, -1.05836212,  2.18901789, -0.24468057, -1.43317494, 0.83649861, -1.66937163]]
    # )
    arm_configs = np.array(
        [[
            2.70336916e+00, -2.04023496e+00, -3.86651939e-02, 8.88009840e-01,
            3.14050424e-08, -8.03649407e-01, -3.68416715e-01
        ],
         [
             -2.94087981e+00, -1.90494670e+00, -5.25384227e-10,
             -1.30288008e-01, -8.00311886e-01, -9.12691365e-02, 7.92558864e-08
         ],
         [
             2.59854555e+00, -1.99433166e+00, -2.32199394e-01, 9.17191684e-01,
             -1.01988192e-07, -9.92958815e-01, -3.94698039e-01
         ],
         [
             2.27892757e+00, -1.94999996e+00, 5.21516392e-09, 2.44353127e-09,
             -3.17857833e-08, -4.11101022e-09, -3.17857834e-08
         ]])
    goals = []
    print "setting goals"
    env.SetViewer('qtcoin')
    for idx, config in enumerate(arm_configs):
        goals.append(SimpleGoal())
        goals[-1].arm_config = config
        goals[-1].key = idx
        goals[-1].grasp_id = idx % 2
        # robot.SetDOFValues(config, manip.GetArmIndices())
        # time.sleep(0.1)
    robot.SetDOFValues(start_config)
    robot.SetActiveDOFs(manip.GetArmIndices())
    for i in range(1):
        print "Creating planner"
        # planner = MGMotionPlanner("SequentialMGBiRRT", manip)
        # planner = MGMotionPlanner("LWAstar;FoldedMultiGraspGraphDynamic",
        #                           manip)
        planner = MGMotionPlanner(
            "LazySP_LWLPAstar;LazyWeightedMultiGraspGraph", manip)
        planner.setup(target_object)
        print "Adding goals"
        planner.addGoals(goals)
        reached_goals = []
        trajectories = []

        # print "PLANNING TOWARDS:\n", str(np.array([goal.arm_config for goal in goals))
        print "Entering planning loop"
        trajectories, reached_goals = planner.plan(1.0)
        trajectories = [time_traj(robot, traj) for traj in trajectories]
        # print "Reached goals: ", [g.key for g in reached_goals]
        # unreached_goals = [g for g in goals if g not in reached_goals]
        # if len(unreached_goals) > 0:
        #     goals_to_remove = random.sample(unreached_goals, min(3, len(unreached_goals)))
        #     # remove a random goal
        #     print "Removing randomly selected unreached goals: ", [g.key for g in unreached_goals]
        #     planner.removeGoals(goals_to_remove)
        # print "Planning for remaining goals"
        # while len(goals) - len(goals_to_remove) > len(reached_goals):
        #     trajs, indices = planner.plan(0.1)
        #     trajectories.extend(trajs)
        #     reached_goals.extend(indices)
    IPython.embed()
