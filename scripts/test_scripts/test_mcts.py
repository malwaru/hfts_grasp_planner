import IPython
import sys
import os
import argparse
import numpy as np
# import hfts_grasp_planner.placement
import hfts_grasp_planner.placement.goal_sampler.random_sampler as rnd_sampler_mod
import hfts_grasp_planner.placement.goal_sampler.mcts_sampler as mcts_sampler_mod
import hfts_grasp_planner.placement.test_image2d.core as image_hierarchy_mod


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # create ImageGoalRegion hierarchy
    image_file_name = os.path.dirname(os.path.abspath(__file__)) + '/../../data/2d/mcts_red_green.png'
    hierarchy = image_hierarchy_mod.ImageGoalRegion(image_file_name)
    random_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, hierarchy, hierarchy, hierarchy, ['fake_manip'])
    # mcts_sampler = mcts_sampler_mod.MCTSPlacementSampler(hierarchy, hierarchy, hierarchy, hierarchy, ["fake_manip"])

    solutions = random_sampler.sample(10000, 100000)
    hierarchy.show()
    IPython.embed()
