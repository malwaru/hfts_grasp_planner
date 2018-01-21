#! /usr/bin/python

import IPython
# import hfts_grasp_planner.sdf.core as sdf_module
# import hfts_grasp_planner.sdf.robot as robot_sdf_module
# import hfts_grasp_planner.sdf.costs as costs_module
import openravepy as orpy
import numpy as np
import hfts_grasp_planner.placement.placement_planning as pp_module

ENV_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/data/environments/table_r850.xml'
SDF_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/data/sdfs/table_r850.sdf'
# ROBOT_BALL_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/r850_robotiq/ball_description.yaml'

if __name__=="__main__":
    env = orpy.Environment()
    env.Load(ENV_PATH)
    env.SetViewer('qtcoin')
    placement_planner = pp_module.PlacementGoalPlanner(None, env)
    placement_volume = (np.array([-1.0, -2.0, 0.5]), np.array([0.2, 0.8, 0.9]))
    placement_planner.set_placement_volume(placement_volume)
    placement_planner.set_object('crayola')
    placement_planner.sample(1)
    IPython.embed()