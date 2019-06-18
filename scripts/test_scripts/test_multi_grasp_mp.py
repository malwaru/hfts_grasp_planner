#!/usr/bin/python
import os
import numpy as np
import openravepy as orpy
import IPython
from hfts_grasp_planner.placement.anytime_planner import MGMotionPlanner
import hfts_grasp_planner.utils as hfts_utils

class FakePlacementGoal(object):
    def __init__(self):
        self.grasp_id = 0
        self.grasp_tf = None
        self.grasp_config = None
        self.arm_config = None
        self.key = 0

if __name__ == "__main__":
    env = orpy.Environment()
    orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
    # orpy.RaveLoadPlugin("ormultigraspmp")
    # planner = orpy.RaveCreatePlanner(env, "ParallelMGBiRRT")
    # module = orpy.RaveCreateModule(env, "SequentialMGBiRRT")
    path = os.path.abspath(os.path.dirname(__file__))
    env_path = path + '/../../data/environments/cluttered_env.xml'
    env.Load(env_path)
    robot = env.GetRobots()[0]
    robot.SetActiveManipulator('arm_with_robotiq')
    env.GetBodies()
    bunny = env.GetKinBody('bunny')
    # set goal
    goal = FakePlacementGoal()
    goal.grasp_tf = np.array([[  1.42108462e-14,  -1.57009249e-16,   1.00000000e+00, -1.84945107e-01],
       [  1.01452490e-21,   1.00000000e+00,   1.57009249e-16, 8.68353993e-04],
       [ -1.00000000e+00,   1.01452490e-21,   1.42108462e-14, 1.01835728e-02],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])
    goal.grasp_config = np.array([ -8.32667268e-16,   4.05037785e-01])
    # goal.arm_config = np.array([-1.80418472, -0.55566296,  0.8929357 ,  3.13613866,  0.12603443, 1.51253753])
    goal.arm_config = np.array([  1.30767653e+00,  -5.28488517e-01,   1.33226763e-15,
        -1.57009246e-16,  -1.02056010e-15,   7.26167762e-16])
    #
    start_config = robot.GetDOFValues()
    manip = robot.GetActiveManipulator()
    planner = MGMotionPlanner("SequentialMGBiRRT", manip)
    planner.setup(bunny)
    planner.addGoals([goal])
    ids, trajs = planner.plan(0.5)
    IPython.embed()
