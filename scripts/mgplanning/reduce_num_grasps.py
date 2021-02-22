#!/usr/bin/python
import argparse
import yaml


def reduce_num_grasps(yaml_contents, num_grasps, verbose=False):
    """Reduce the number of grasps in a yaml goal set.

    Args:
        yaml_contents (dict): yaml dictionary containing goals etc
        num_grasps (int): the maximum number of desired grasps
        verbose (bool): if True, print statements

    Returns:
        filtered yaml_contents
    """
    # get grasp information
    grasps = {grasp['id']: grasp for grasp in yaml_contents['grasps']}
    if len(grasps) > num_grasps:
        # create a mapping from grasp_id -> goals with that grasp
        goals_per_grasp = {}
        for goal in yaml_contents['goals']:
            if goal['grasp_id'] not in goals_per_grasp:
                goals_per_grasp[goal['grasp_id']] = [goal]
            else:
                goals_per_grasp[goal['grasp_id']].append(goal)
        # get a list containing tuples (grasp_id, #goals for this grasp)
        num_goals_per_grasp = [(grasp_id, len(goals)) for grasp_id, goals in goals_per_grasp.iteritems()]
        # sort it so that grasps with more goals come first
        num_goals_per_grasp.sort(key=lambda x: x[1], reverse=True)
        # overwrite goals to only contain the goals belonging to the args.number_of_grasps first grasps
        yaml_contents['goals'] = []
        yaml_contents['grasps'] = []
        for gid, _ in num_goals_per_grasp[:num_grasps]:
            yaml_contents['goals'].extend(goals_per_grasp[gid])
            yaml_contents['grasps'].append(grasps[gid])
    elif verbose:
        print "The input only contains %i grasps; will copy original contents" % len(grasps)
    return yaml_contents


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Utility script to reduce the number of grasps in a goals yaml.")
    parser.add_argument('goal_yaml',
                        help='Path to a yaml file containing placement goals for Yumi.'
                        'The file should contain at least a list of goals and grasps.',
                        type=str)
    parser.add_argument('output_yaml', help='Filename to store filter result in.', type=str)
    parser.add_argument('number_of_grasps',
                        help='The number of grasps for which goals should be in output_yaml',
                        type=int)
    args = parser.parse_args()
    with open(args.goal_yaml, 'r') as infile:
        contents = yaml.load(infile)
    # reduce number of grasps
    contents = reduce_num_grasps(contents, args.number_of_grasps)
    # write contents to file
    assert (len(contents['grasps']) <= args.number_of_grasps)
    with open(args.output_yaml, 'w') as outfile:
        yaml.dump(contents, outfile)
