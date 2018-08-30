#! /usr/bin/python

import os
import time
import IPython
import logging
import hfts_grasp_planner.utils
from hfts_grasp_planner.integrated_hfts_planner import IntegratedHFTSPlanner

PKG_PATH = os.path.abspath(os.path.dirname(__file__)) + "/../../"
DATA_PATH = PKG_PATH + "data/"
ENV_FILE = DATA_PATH + "environments/test_env_r850_sdh.xml"
HAND_FILE = PKG_PATH + "models/schunk-sdh/schunk-sdh.zae"
HAND_CONFIG_FILE = PKG_PATH + "models/schunk-sdh/hand_config.yaml"
HAND_BALL_FILE = PKG_PATH + "models/schunk-sdh/ball_description.yaml"
HAND_CACHE_FILE = DATA_PATH + "cache/schunk_hand.npy"
# HAND_FILE = PKG_PATH + "models/robotiq/urdf_openrave_conversion/robotiq_s.xml"
# HAND_CONFIG_FILE = PKG_PATH + "models/robotiq/hand_config.yaml"
# HAND_BALL_FILE = PKG_PATH + "models/robotiq/ball_description.yaml"
# HAND_CACHE_FILE = DATA_PATH + "cache/robotiq.npy"
ROBOT_NAME = 'kuka-r850-sdh'
MANIPULATOR_NAME = 'arm_with_sdh'


if __name__ == "__main__":
    object_name = 'crayola'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    planner = IntegratedHFTSPlanner(env_file=ENV_FILE, hand_file=HAND_FILE, hand_cache_file=HAND_CACHE_FILE,
                                    hand_config_file=HAND_CONFIG_FILE, hand_ball_file=HAND_BALL_FILE,
                                    robot_name=ROBOT_NAME, manipulator_name=MANIPULATOR_NAME,
                                    data_root_path=DATA_PATH, b_visualize_system=True,
                                    b_visualize_grasps=False, b_show_search_tree=True,
                                    b_show_traj=True, compute_velocities=True)
    planner.load_target_object(obj_id=object_name)
    robot = planner.get_robot()
    start_config = robot.GetDOFValues()
    IPython.embed()
    sol, grasp_pose = planner.plan(start_config)
    IPython.embed()
