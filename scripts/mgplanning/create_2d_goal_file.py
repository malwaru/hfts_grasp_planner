#!/usr/bin/python
import argparse
import numpy as np
import os
import sys
import yaml


def sample_goals(gid, num_goals, cost_space, sample_area):
    (min_x, min_y), (max_x, max_y) = sample_area
    valid_rows, valid_cols = np.where(np.logical_not(np.isinf(cost_space[min_y:max_y, min_x:max_x])))
    sampled_indices = np.random.choice(len(valid_rows), num_goals)
    return [[gid, int(min_x + valid_cols[idx]),
             int(min_y + valid_rows[idx]),
             np.random.rand()] for idx in sampled_indices]


def verify_valid_start(start_config, cost_spaces):
    return not np.array([np.isinf(space[start_config[1], start_config[0]])
                         for _, space in cost_spaces.iteritems()]).any()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Create a yaml file with start and goal configurations on 2D cost spaces with random goal quality.")
    parser.add_argument('cost_spaces', help='Path to a folder containing cost spaces.', type=str)
    parser.add_argument('output_file', help='Filename to store goal list in', type=str)
    parser.add_argument('start_config', help='The start configuration q0 (x, y)', type=int, nargs=2)
    parser.add_argument('--num_goals_per_grasp',
                        help='Specify the number of goals to sample per grasp',
                        type=int,
                        default=1)
    parser.add_argument('--sample_area',
                        help='Bounding box to sample in. If not provided, samples everywhere.',
                        nargs=4,
                        type=int)
    args = parser.parse_args()
    # load cost spaces
    if not os.path.isdir(args.cost_spaces):
        print "Error: The given cost spaces path is not a folder."
        sys.exit(1)
    cost_spaces = {}
    for filename in os.listdir(args.cost_spaces):
        grasp_id_str = os.path.basename(filename).split('.')[0]
        try:
            grasp_id = int(grasp_id_str)
            cost_spaces[grasp_id] = np.load(args.cost_spaces + '/' + filename)
        except ValueError as e:
            print "Could not parse cost space from: ", filename
            continue

    # check whether the given start vertex is valid
    if not verify_valid_start(args.start_config, cost_spaces):
        print "Error: The given start vertex is not collision-free for all grasps"
        sys.exit(1)
    # parse sample area for goals
    sample_area = (np.array([0, 0], dtype=int), np.array(cost_spaces.values()[0].shape, dtype=int))
    if args.sample_area:
        sample_area = (np.array([args.sample_area[0], args.sample_area[1]],
                                dtype=int), np.array([args.sample_area[2], args.sample_area[3]], dtype=int))
    # sample goals
    goal_configs = []
    for gid, cost_space in cost_spaces.iteritems():
        if gid == 0:
            continue
        goal_configs.extend(sample_goals(gid - 1, args.num_goals_per_grasp, cost_space, sample_area))
    # save to disc
    with open(args.output_file, 'w') as out_file:
        yaml.dump({'start_config': args.start_config, 'goal_configs': goal_configs}, out_file)
    sys.exit(0)
