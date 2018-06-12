#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf_module
import openravepy as orpy
import numpy as np
import os
import hfts_grasp_planner.placement.placement_planning as pp_module

ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/table_r850.xml'
SDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/table_r850.sdf'
# ROBOT_BALL_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/r850_robotiq/ball_description.yaml'

if __name__=="__main__":
    env = orpy.Environment()
    env.Load(ENV_PATH)
    env.SetViewer('qtcoin')
    robot_name = env.GetRobots()[0].GetName()
    scene_sdf = sdf_module.SceneSDF(env, ['crayola'], excluded_bodies=[robot_name, 'bunny'])
    if os.path.exists(SDF_PATH):
        scene_sdf.load(SDF_PATH)
    else:
        volume = np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.2])
        scene_sdf.create_sdf(volume, 0.02, 0.01)
        scene_sdf.save(SDF_PATH)
    bunny = env.GetKinBody('bunny')
    bunny_octree = kinbody_sdf_module.OccupancyOctree(10e-9, bunny)
    IPython.embed()