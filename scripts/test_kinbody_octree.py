#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf_module
import openravepy as orpy
import numpy as np
import os
import hfts_grasp_planner.placement.placement_planning as pp_module
import timeit

ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/cluttered_env.xml'
SDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/cluttered_test_env.sdf'
# ROBOT_BALL_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/r850_robotiq/ball_description.yaml'

# if __name__=="__main__":
#     setup_code = """\
# import hfts_grasp_planner.sdf.core as sdf_module
# import hfts_grasp_planner.sdf.kinbody as kinbody_sdf_module
# import openravepy as orpy
# import numpy as np
# import os
# import hfts_grasp_planner.placement.placement_planning as pp_module
# ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/table_r850.xml'
# SDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/table_r850.sdf'

# env = orpy.Environment()
# env.Load(ENV_PATH)
# # env.SetViewer('qtcoin')
# robot_name = env.GetRobots()[0].GetName()
# scene_sdf = sdf_module.SceneSDF(env, ['crayola'], excluded_bodies=[robot_name, 'bunny'])
# if os.path.exists(SDF_PATH):
#     scene_sdf.load(SDF_PATH)
# else:
#     volume = np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.2])
#     scene_sdf.create_sdf(volume, 0.02, 0.01)
#     scene_sdf.save(SDF_PATH)
# bunny = env.GetKinBody('bunny')
# bunny.SetTransform(np.array([[  1.00000000e+00,  -3.59293981e-35,   1.12213715e-49, 1.59666687e-01], [  3.59293981e-35,   1.00000000e+00,  -6.24634537e-15, 7.46351600e-01], [  1.12213715e-49,   6.24634537e-15,   1.00000000e+00, -2.19669342e-02],[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]]))
# bunny_octree = kinbody_sdf_module.OccupancyOctree(10e-9, bunny)
# """
#     exec_code = "bunny_octree.compute_intersection(scene_sdf)"
#     print timeit.timeit(stmt=exec_code, setup=setup_code, number=1000) / 1000
#     print "Done"
def draw_volume(env, volume):
    return env.drawbox(0.5 * (volume[:3] + volume[3:]), 0.5 * (volume[3:] - volume[:3]), np.array([0.3, 0.3, 0.3, 0.3]))

if __name__=="__main__":
    env = orpy.Environment()
    env.Load(ENV_PATH)
    env.SetViewer('qtcoin')
    robot_name = env.GetRobots()[0].GetName()
    scene_sdf = sdf_module.SceneSDF(env, [], excluded_bodies=[robot_name, 'bunny', 'crayola', 'stick'])
    if os.path.exists(SDF_PATH):
        scene_sdf.load(SDF_PATH)
        # volume = np.array([-0.6, 0.3, 0.5, 0.7, 1.2, 1.0])
        # sdf_vis = sdf_module.ORSDFVisualization(env)
        # sdf_vis.visualize(scene_sdf, volume, resolution=0.01, max_sat_value=0.4, style='sprites')
    else:
        # volume = np.array([-1.3, -1.3, -0.5, 1.3, 1.3, 1.2])
        volume = np.array([-0.6, 0.3, 0.5, 0.7, 1.2, 1.0])
    # placement_volume = (np.array([-0.35, 0.55, 0.64]), np.array([0.53, 0.9, 0.75]))
        handle = draw_volume(env, volume)
        resolution = 0.005
        print 'Check volume!'
        IPython.embed()
        scene_sdf.create_sdf(volume, resolution, resolution)
        scene_sdf.save(SDF_PATH)
        sdf_vis = sdf_module.ORSDFVisualization(env)
        sdf_vis.visualize(scene_sdf, volume, resolution=0.01, max_sat_value=0.7, style='sprites')
    test_body = env.GetKinBody('stick')
    test_octree = kinbody_sdf_module.OccupancyOctree(0.005, test_body)
    IPython.embed()