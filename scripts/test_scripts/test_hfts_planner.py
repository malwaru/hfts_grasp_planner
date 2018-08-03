#! /usr/bin/python

import os
import time
import IPython
import hfts_grasp_planner.utils
from hfts_grasp_planner.core import HFTSSampler, HFTSNode

PKG_PATH = os.path.abspath(os.path.dirname(__file__)) + "/../../"
DATA_PATH = PKG_PATH + "data/"
# HAND_FILE = PKG_PATH + "models/schunk-sdh/schunk-sdh.zae"
# HAND_CONFIG_FILE = PKG_PATH + "models/schunk-sdh/hand_config.yaml"
# HAND_BALL_FILE = PKG_PATH + "models/schunk-sdh/ball_description.yaml"
# HAND_CACHE_FILE = DATA_PATH + "cache/schunk_hand.npy"

HAND_FILE = PKG_PATH + "models/robotiq/urdf_openrave_conversion/robotiq_s.xml"
HAND_CONFIG_FILE = PKG_PATH + "models/robotiq/hand_config.yaml"
HAND_BALL_FILE = PKG_PATH + "models/robotiq/ball_description.yaml"
HAND_CACHE_FILE = DATA_PATH + "cache/robotiq.npy"

if __name__ == "__main__":
    object_name = 'crayola'
    object_io = hfts_grasp_planner.utils.ObjectFileIO(DATA_PATH)
    root_node = HFTSNode()
    planner = HFTSSampler(object_io, num_hops=4, vis=True)
    planner.load_hand(hand_file=HAND_FILE,
                      hand_cache_file=HAND_CACHE_FILE,
                      hand_config_file=HAND_CONFIG_FILE,
                      hand_ball_file=HAND_BALL_FILE)
    planner.load_object(object_name)
    finished = False
    while not finished:
        return_node = planner.sample_grasp(root_node, planner.get_maximum_depth(), post_opt=True)
        finished = return_node.is_goal()

    IPython.embed()
