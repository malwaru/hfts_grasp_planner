import IPython
import sys
import os
import rospy
import argparse
import numpy as np
# import hfts_grasp_planner.placement
import hfts_grasp_planner.placement.goal_sampler.random_sampler as rnd_sampler_mod
import hfts_grasp_planner.placement.goal_sampler.mcts_sampler as mcts_sampler_mod
import hfts_grasp_planner.placement.goal_sampler.mcts_visualization as mcts_visualizer_mod
import hfts_grasp_planner.placement.test_image2d.core as image_hierarchy_mod


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    rospy.init_node("MCTSTester", log_level=rospy.DEBUG)
    # create ImageGoalRegion hierarchy
    image_file_name = os.path.dirname(os.path.abspath(__file__)) + '/../../data/2d/mcts_red_green.png'
    hierarchy = image_hierarchy_mod.ImageGoalRegion(image_file_name)
    random_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, hierarchy, hierarchy, hierarchy, ['fake_manip'])
    # mcts_visualizer = mcts_visualizer_mod.MCTSVisualizer(None, None)
    mcts_visualizer = None
    mcts_sampler = mcts_sampler_mod.MCTSPlacementSampler(hierarchy, hierarchy, hierarchy, hierarchy, ["fake_manip"],
                                                         c=1, debug_visualizer=mcts_visualizer)
    mcts_solutions, mcts_num_solutions = mcts_sampler.sample(1000, 10000)
    mcts_stats = (hierarchy.get_num_construction_calls(), hierarchy.get_num_validity_calls(),
                  hierarchy.get_num_relaxation_calls(), hierarchy.get_num_evaluate_calls())
    rnd_solutions, rnd_num_solutions = random_sampler.sample(1000, 10000)
    rnd_stats = (hierarchy.get_num_construction_calls(), hierarchy.get_num_validity_calls(),
                 hierarchy.get_num_relaxation_calls(), hierarchy.get_num_evaluate_calls())
    # hierarchy.show()
    print "MCTS solutions: ", mcts_num_solutions,\
        "#constructions: %i, #validity: %i, #relaxations: %i, #evaluate: %i" % mcts_stats
    print "Random solutions: ", rnd_num_solutions,\
        "#constructions: %i, #validity: %i, #relaxations: %i, #evaluate: %i" % rnd_stats
    IPython.embed()
