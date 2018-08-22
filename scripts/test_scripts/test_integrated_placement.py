#! /usr/bin/python

import os
import IPython
import numpy as np
import openravepy as orpy
import hfts_grasp_planner.integrated_plcmt_planner as ipp_module

ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/cluttered_env.xml'
SDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/cluttered_test_env.sdf'
DATA_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data'
ROBOT_NAME = 'r850_robotiq'
MANIP_NAME = 'arm_with_robotiq'


def draw_volume(env, volume):
    return env.drawbox(0.5 * (volume[0] + volume[1]), 0.5 * (volume[1] - volume[0]), np.array([0.3, 0.3, 0.3, 0.3]))


grasp_map = {
    'bunny': {
        'grasp_pose': np.array([[-4.89838933e-09,   4.10908549e-02,   9.99155414e-01, -1.61563370e-02],
                                [2.38317911e-07,  -9.99155414e-01,   4.10908549e-02, 2.94656865e-02],
                                [1.00000000e+00,   2.38317910e-07,  -4.89843450e-09, 1.66712295e-01],
                                [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]]),
        'start_config':  np.array([0.,  0.,  0.,  0.,  0., 0.,  0.13168141,  0.72114204]),
    },
    # TODO other objects
}


def execute_placement_planner(planner, name, volume, time_limit):
    grasp_info = grasp_map['bunny']
    robot = planner._env.GetRobot(ROBOT_NAME)
    # set robot to specific grasp configuration for testing
    robot.SetDOFValues(grasp_info['start_config'])
    # set the object to the pose it belongs to
    eef_pose = robot.GetManipulator(MANIP_NAME).GetEndEffectorTransform()
    target_object = planner._env.GetKinBody(target_obj_name)
    target_object.SetTransform(np.dot(eef_pose, grasp_info['grasp_pose']))
    path, pose = planner.plan_placement(name, volume, grasp_info['grasp_pose'],
                                        grasp_info['start_config'], time_limit)
    if path:
        # TODO visualize path by executing it
        print path
        robot.SetActiveDOFValues(path[-1].get_configuration())


if __name__ == "__main__":
    target_obj_name = 'bunny'
    # Grasp settings for bunny
    # obj_pose = np.array([[1.00000000e+00,   2.38317910e-07,  -4.89843428e-09, 7.30712295e-01],
    #                      [2.38317911e-07,  -9.99155414e-01,   4.10908549e-02, 2.94656865e-02],
    #                      [4.89838955e-09,  -4.10908549e-02,  -9.99155414e-01, 8.06156337e-01],
    #                      [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])
    # grasp_pose = np.array([[-4.89838933e-09,   4.10908549e-02,   9.99155414e-01, -1.61563370e-02],
    #                        [2.38317911e-07,  -9.99155414e-01,   4.10908549e-02, 2.94656865e-02],
    #                        [1.00000000e+00,   2.38317910e-07,  -4.89843450e-09, 1.66712295e-01],
    #                        [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])
    time_limit = 60.0
    sdf_volume = (np.array([-0.6, 0.3, 0.3]), np.array([0.7, 1.2, 0.8]))
    planner = ipp_module.IntegratedPlacementPlanner(ENV_PATH, SDF_PATH, sdf_volume, DATA_PATH,
                                                    ROBOT_NAME, MANIP_NAME, draw_search_tree=True)
    # set a placement target volume
    placement_volume = (np.array([-0.35, 0.45, 0.42]), np.array([0.53, 0.9, 0.64]))  # inside shelf
    # planner._env.SetViewer('qtcoin')
    handle = draw_volume(planner._env, placement_volume)
    print "Check the placement volume!", placement_volume
    IPython.embed()
    handle = None
    execute_placement_planner(planner, target_obj_name, placement_volume, time_limit)
    IPython.embed()
