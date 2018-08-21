#! /usr/bin/python

import os
import IPython
import openravepy as orpy
import timeit
import numpy as np
import hfts_grasp_planner.placement.placement_planning as pp_module
from hfts_grasp_planner.utils import ObjectFileIO

ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/cluttered_env.xml'
# SDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/cluttered_test_env.sdf'
DATA_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data'
# ROBOT_BALL_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/r850_robotiq/ball_description.yaml'


def draw_volume(env, volume):
    return env.drawbox(0.5 * (volume[0] + volume[1]), 0.5 * (volume[1] - volume[0]), np.array([0.3, 0.3, 0.3, 0.3]))


def setup_timeit():
    ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/cluttered_env.xml'
    env = orpy.Environment()
    env.Load(ENV_PATH)
    quality_fn = pp_module.QuasistaticFallingQuality(env)
    # placement_volume = (np.array([-0.35, 0.55, 0.61]), np.array([0.53, 0.9, 0.72]))
    placement_volume = (np.array([-0.35, 0.55, 0.51]), np.array([0.53, 0.9, 0.61]))  # inside shelf
    quality_fn.set_placement_volume(placement_volume)
    body = env.GetKinBody('crayola')
    quality_fn.set_target_object(body)
    body.SetTransform(np.array([[0.96211303, -0.08257759, -0.25984508, -0.33390048],
                                [-0.12367726,  0.71716285, -0.68584356,  0.86756551],
                                [0.24298654,  0.69199595,  0.67977875,  0.56252837],
                                [0.,  0.,  0.,  1.]]))
    # body.SetTransform(np.array([[7.64019645e-01,  -2.64647726e-01,  -5.88417848e-01, 7.03253746e-02],
    #                             [-9.46312306e-09,   9.12002862e-01,  -4.10183836e-01, 7.57531166e-01],
    #                             [6.45192981e-01,   3.13388514e-01,   6.96788100e-01, 8.50161552e-01],
    #                             [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]]))
    return quality_fn, body


def measure_time():
    print timeit.timeit("quality_fn.compute_quality(body.GetTransform())",
                        setup="from __main__ import setup_timeit; quality_fn, body = setup_timeit()",
                        number=100) / 100.0


def manual_test():
    # orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
    env = orpy.Environment()
    env.Load(ENV_PATH)
    robot_name = env.GetRobots()[0].GetName()
    target_obj_name = 'crayola'
    quality_fn = pp_module.QuasistaticFallingQuality(env)
    # placement_volume = (np.array([-0.35, 0.55, 0.61]), np.array([0.53, 0.9, 0.72]))  # on top of shelf
    placement_volume = (np.array([-0.35, 0.55, 0.51]), np.array([0.53, 0.9, 0.61]))  # inside shelf
    # placement_volume = (np.array([0.24, 0.58, 0.73]), np.array([0.29, 0.8, 0.8]))  # volume in small gap
    quality_fn.set_placement_volume(placement_volume)
    body = env.GetKinBody(target_obj_name)
    # body.SetTransform(np.array([[9.99504956e-01,  -3.14596166e-02,   3.67252210e-04, 1.68144822e-01],
    #                             [3.13400655e-02,   9.94545318e-01,  -9.94857287e-02, 6.57463729e-01],
    #                             [2.76453391e-03,   9.94479886e-02,   9.95038921e-01, 8.29895318e-01],
    #                             [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]]))
    body.SetTransform(np.array([[0.96211303, -0.08257759, -0.25984508, -0.33390048],
                                [-0.12367726,  0.71716285, -0.68584356,  0.86756551],
                                [0.24298654,  0.69199595,  0.67977875,  0.56252837],
                                [0.,  0.,  0.,  1.]]))
    env.GetKinBody('bunny').SetTransform(np.array([[1.00000000e+00,   4.07142262e-08,   8.97183551e-08,
                                                    1.61587194e-01],
                                                   [4.07142262e-08,   6.58464440e-01,  -7.52611840e-01,
                                                    7.74527550e-01],
                                                   [-8.97183551e-08,   7.52611840e-01,   6.58464440e-01,
                                                    7.31117427e-01],
                                                   [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                                    1.00000000e+00]]))
    quality_fn.set_target_object(body)
    # quality_fn._env.SetViewer('qtcoin')
    # handle = draw_volume(quality_fn._env, placement_volume)
    # print "Check the placement volume!", placement_volume
    # handle = None
    # IPython.embed()
    quality_fn.compute_quality(body.GetTransform())
    # IPython.embed()


if __name__ == "__main__":
    measure_time()
    # manual_test()
