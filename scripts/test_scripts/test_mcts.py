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
    mcts_sampler = mcts_sampler_mod.MCTSPlacementSampler(hierarchy, hierarchy, hierarchy, hierarchy, ["fake_manip"],
        c=20)

    mcts_solutions, mcts_num_solutions = mcts_sampler.sample(10000, 10000)
    # rnd_solutions, rnd_num_solutions = random_sampler.sample(1000, 10000)
    hierarchy.show()
    print "MCTS solutions: ", mcts_num_solutions
    # print "Random solutions: ", rnd_num_solutions
    IPython.embed()
