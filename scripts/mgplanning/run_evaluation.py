#!/usr/bin/python

import argparse
import subprocess
import yaml

# Hacky script to just run planner for many different algorithms

algorithm_graph_combinations = [
    # SingleGraspGraph
    ("Astar", "SingleGraspGraph"),
    ("LWAstar", "SingleGraspGraph"),
    ("LPAstar", "SingleGraspGraph"),
    ("LWLPAstar", "SingleGraspGraph"),
    ("LazySP_LLPAstar", "SingleGraspGraph"),
    # MultiGraspGraph
    ("Astar", "MultiGraspGraph"),
    ("LWAstar", "MultiGraspGraph"),
    ("LPAstar", "MultiGraspGraph"),
    ("LWLPAstar", "MultiGraspGraph"),
    ("LazySP_LLPAstar", "MultiGraspGraph"),
    # FoldedMultiGraspGraph
    ("LWAstar", "FoldedMultiGraspGraphStationary"),
    ("LWAstar", "FoldedMultiGraspGraphDynamic"),
    # LazyWeightedMultiGraspGraph
    ("LazySP_LLPAstar", "LazyWeightedMultiGraspGraph"),
    ("LazySP_LWLPAstar", "LazyWeightedMultiGraspGraph"),
    ("LazySP_LPAstar", "LazyWeightedMultiGraspGraph"),
    # LazyWedgeWeightedMultiGraspGraph
    ("LazySP_LLPAstar", "LazyEdgeWeightedMultiGraspGraph"),
    ("LazySP_LWLPAstar", "LazyEdgeWeightedMultiGraspGraph"),
    ("LazySP_LPAstar", "LazyEdgeWeightedMultiGraspGraph"),
]

test_case_info = {
    "name": "Name describing the test case",
    "domain": "can be 'placing', '2d', 'picking'",
    "scene_file": "path to OR scene or image state space folder",
    "object_xml": "path to OR kinbody xml in case of placing or picking",
    "num_grasps": "the number of grasps",
    "lambda": "lambda parameter to scale between path and goal cost",
    "configs": "file with goal and extra configs and grasp information",
}


def call_planner(algorithm, graph, log_path, test_case_info):
    """Call the planner for

    Args:
        algorithm ([type]): [description]
        graph ([type]): [description]
        log_path (str): path to the root folder of where to store results
        test_case_info ([type]): [description]
    """
    try:
        commands = []
        log_folder_name = log_path + '/' + algorithm + "__" + graph + '__' +\
            test_case_info['name'] + '__' + str(test_case_info['num_grasps'])
        if test_case_info['domain'] == 'placing':
            commands = [
                'rosrun', 'hfts_grasp_planner', 'run_mgplanner_yumi.py',
                test_case_info['scene_file'], test_case_info['object_xml'],
                test_case_info['configs'], algorithm, graph, '--lmbda',
                test_case_info['lambda'], '--stats_file',
                log_folder_name + '/run_stats', '--results_file',
                log_folder_name + '/results', '--planner_log',
                log_folder_name + '/log'
            ]
        elif test_case_info['domain'] == 'picking':
            raise NotImplementedError("Picking is not yet implemented")
        elif test_case_info['domain'] == '2d':
            # TODO assemble command line arguments to call test_imagespace_planner
            commands = [
                'rosrun', 'hfts_grasp_planner', 'test_image_space_algorithm',
                '--image_path', test_case_info['scene_file'], '--configs_file',
                test_case_info['configs'], '--lambda',
                test_case_info['lambda'], '--algorithm_type', algorithm,
                '--graph_type', graph, '--stats_file',
                log_folder_name + '/run_stats', '--results_file',
                log_folder_name + '/results', '--roadmap_log_file',
                log_folder_name + '/log_roadmap', '--evaluation_log_file',
                log_folder_name + '/log_evaluation'
            ]
        else:
            raise ValueError("ERROR: Unknown domain type %s." %
                             test_case_info['domain'])
        print "Executing command %s" % str(commands)
        subprocess.call(map(str, commands))
    except (ValueError) as e:
        print "An error occured: %s. Skipping this test case." % str(e)


def run_tests(testfile,
              log_path,
              num_runs,
              algorithm_whitelist=None,
              graph_whitelist=None,
              algorithm_blacklist=None,
              graph_blacklist=None):
    """Run the test cases specified in the yaml file <testfile>.

    Args:
        testfile (str): Path to yaml file containing test cases.
        log_path (str): Path to where to store logs
        num_runs (int): The number of executions of each algorithm to get runtime averages.
        algorithm_whitelist (list of str), optional: A subset of algorithms to test only.
        graph_whitelist (list of str), optional: A subset of graphs to test only
        algorithm_blacklist (list of str), optional: A subset of algorithms to not test.
        graph_blacklist (list of str), optional: A subset of graphs to not test.
    """
    # read testfile
    with open(testfile, 'r') as yaml_file:
        test_cases = yaml.load(yaml_file)
    for algo_name, graph_name in algorithm_graph_combinations:
        # skip combinations that aren't in the whitelist or those that are in the blacklist
        if algorithm_whitelist and algo_name not in algorithm_whitelist:
            continue
        if graph_whitelist and graph_name not in graph_whitelist:
            continue
        if algorithm_blacklist and algo_name in algorithm_blacklist:
            continue
        if graph_blacklist and graph_name in graph_blacklist:
            continue
        for tid, test_case in enumerate(test_cases):
            # TODO can we parallize this?
            for r in range(num_runs):
                print "Running run %i/%i of test case %i/%i for algorithm %s on graph %s" % (
                    r + 1, num_runs, tid + 1, len(test_cases), algo_name,
                    graph_name)
                call_planner(algo_name, graph_name, log_path, test_case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script to run a large collection of experiments on the multi-grasp motion planner across different domains."
        "The script takes as input a yaml-file containing test case information. A single test case is defined as:\n %s."
        % str(test_case_info))
    parser.add_argument('tests_file',
                        help="Path to a yaml file specifying the test cases",
                        type=str)
    parser.add_argument(
        'log_path',
        type=str,
        help='Path to base folder of where to store the results')
    parser.add_argument(
        '--num_runs',
        help="Number of runs to execute for runtime measurements.",
        type=int,
        default=10)
    parser.add_argument(
        "--limit_algorithms",
        help="Optionally limit the evaluation to the given list of algorithms",
        nargs="*",
        type=str)
    parser.add_argument(
        "--limit_graphs",
        help="Optionally limit the evaluation to the given list of graphs",
        nargs="*",
        type=str)
    parser.add_argument("--exclude_algorithms",
                        help="Optionally exclude the given list of algorithms",
                        nargs="*",
                        type=str)
    parser.add_argument("--exclude_graphs",
                        help="Optionally exclude the given list of graphs",
                        nargs="*",
                        type=str)
    args = parser.parse_args()
    run_tests(args.tests_file, args.log_path, args.num_runs,
              args.limit_algorithms, args.limit_graphs,
              args.exclude_algorithms, args.exclude_algorithms)
