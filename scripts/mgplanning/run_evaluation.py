#!/usr/bin/python

import argparse
import subprocess
from multiprocessing import Pool
import yaml
import os
import itertools

# Hacky script to just run planner for many different algorithms

algorithm_graph_combinations = [
    # SingleGraspGraph
    ("Astar", "SingleGraspGraph"),
    ("LWAstar", "SingleGraspGraph"),
    ("LPAstar", "SingleGraspGraph"),
    ("LWLPAstar", "SingleGraspGraph"),
    # ("LazySP_LLPAstar", "SingleGraspGraph"),
    # MultiGraspGraph
    ("Astar", "MultiGraspGraph"),
    ("LWAstar", "MultiGraspGraph"),
    ("LPAstar", "MultiGraspGraph"),
    ("LWLPAstar", "MultiGraspGraph"),
    ("LazySP_LLPAstar", "MultiGraspGraph"),  # should be identical to LazySP_LLPAstar on LazyEdgeWeightedMultiGraspGraph
    # FoldedMultiGraspGraph
    ("LWAstar", "FoldedMultiGraspGraphStationary"),
    ("LWAstar", "FoldedMultiGraspGraphDynamic"),
    # LazyWeightedMultiGraspGraph
    # ("LazySP_LLPAstar", "LazyWeightedMultiGraspGraph"),
    ("LazySP_LWLPAstar", "LazyWeightedMultiGraspGraph"),
    ("LazySP_LPAstar", "LazyWeightedMultiGraspGraph"),
    # LazyEdgeWeightedMultiGraspGraph
    # ("LazySP_LLPAstar", "LazyEdgeWeightedMultiGraspGraph"),
    ("LazySP_LWLPAstar", "LazyEdgeWeightedMultiGraspGraph"),
    ("LazySP_LPAstar", "LazyEdgeWeightedMultiGraspGraph"),
    # LazyGrownMultiGraspGraph
    ("LazySP_LLPAstar", "LazyGrownMultiGraspGraph"),
    ("LazySP_LWLPAstar", "LazyGrownMultiGraspGraph"),
    ("LazySP_LPAstar", "LazyGrownMultiGraspGraph"),
    # LazyGrownLazyWeightedMultiGraspGraph
    # ("LazySP_LLPAstar", "LazyGrownLazyWeightedMultiGraspGraph"),
    ("LazySP_LWLPAstar", "LazyGrownLazyWeightedMultiGraspGraph"),
    ("LazySP_LPAstar", "LazyGrownLazyWeightedMultiGraspGraph"),
    # LazyGrownLazyEdgeWeightedMultiGraspGraph
    # ("LazySP_LLPAstar", "LazyGrownLazyEdgeWeightedMultiGraspGraph"),
    ("LazySP_LWLPAstar", "LazyGrownLazyEdgeWeightedMultiGraspGraph"),
    ("LazySP_LPAstar", "LazyGrownLazyEdgeWeightedMultiGraspGraph")
]

test_case_info = {
    "name": "Name describing the test case",
    "domain": "can be 'placing', '2d', 'picking'",
    "scene_file": "path to OR scene or image state space folder",
    "object_xml": "path to OR kinbody xml in case of placing or picking",
    "num_grasps": "a list of number of grasps",
    "lambda": "lambda parameter to scale between path and goal cost",
    "configs": "file with goal and extra configs and grasp information",
    "integrator_step_sizes": "(Optional) The step size for edge cost integration",
    "batch_sizes": "(Optional) The base size of the roadmap",
    "edge_selector_type": "(Optional) List of string describing which edge selector type to use in LazySP"
}


def call_planner(algorithm, graph, log_path, test_case_info, num_runs, dry_run):
    """Call the planner for

    Args:
        algorithm ([type]): [description]
        graph ([type]): [description]
        log_path (str): path to the root folder of where to store results
        test_case_info ([type]): [description]
        num_runs(int): the number of times to run each instance of a test case
        dry_run(bool): If True, only print command, don't actually call it
    """
    try:
        commands = []
        step_sizes = ['default'
                      ] if 'integrator_step_sizes' not in test_case_info else test_case_info['integrator_step_sizes']
        batch_sizes = ['default'] if 'batch_sizes' not in test_case_info else test_case_info['batch_sizes']
        lambdas = [test_case_info['lambda']] if type(test_case_info['lambda']) != list else test_case_info['lambda']
        if 'edge_selector_type' not in test_case_info or 'LazySP' not in algorithm:
            edge_selector_types = ['default']
        else:
            edge_selector_types = test_case_info['edge_selector_type']
        for num_grasps, step_size, batch_size, lmbda, ee_selector in itertools.product(
                test_case_info['num_grasps'], step_sizes, batch_sizes, lambdas, edge_selector_types):
            log_folder_name = (log_path + "/%s__%s__%s__%d-grasps__%s-step_size__%s-batch_size__%s-lambda__%s-eetype"
                               ) % (algorithm, graph, test_case_info['name'], num_grasps, str(step_size),
                                    str(batch_size), str(lmbda), ee_selector)
            # if os.path.exists(log_folder_name + '/results'):
            # do not overwrite existing results
            # continue
            if test_case_info['domain'] == 'placing':
                commands = [
                    'rosrun', 'hfts_grasp_planner', 'run_mgplanner_yumi.py', test_case_info['scene_file'],
                    test_case_info['object_xml'], test_case_info['configs'], algorithm, graph, '--lmbda', lmbda,
                    '--stats_file', log_folder_name + '/run_stats', '--results_file', log_folder_name + '/results',
                    '--planner_log', log_folder_name + '/log'
                ]
            elif test_case_info['domain'] == 'picking':
                commands = [
                    'rosrun', 'hfts_grasp_planner', 'run_mgplanner_yumi.py', test_case_info['scene_file'],
                    test_case_info['object_xml'], test_case_info['configs'], algorithm, graph, '--lmbda', lmbda,
                    '--stats_file', log_folder_name + '/run_stats', '--results_file', log_folder_name + '/results',
                    '--planner_log', log_folder_name + '/log'
                ]
            elif test_case_info['domain'] == '2d':
                commands = [
                    'rosrun', 'hfts_grasp_planner', 'test_image_space_algorithm', '--image_path',
                    test_case_info['scene_file'], '--configs_file', test_case_info['configs'], '--lambda', lmbda,
                    '--algorithm_type', algorithm, '--graph_type', graph, '--stats_file',
                    log_folder_name + '/run_stats', '--results_file', log_folder_name + '/results',
                    '--roadmap_log_file', log_folder_name + '/log_roadmap', '--evaluation_log_file',
                    log_folder_name + '/log_evaluation'
                ]
            else:
                raise ValueError("ERROR: Unknown domain type %s." % test_case_info['domain'])
            # limit number of grasps
            commands.extend(['--limit_grasps', str(num_grasps)])
            # append optional commands for step size and roadmap size
            if step_size != "default":
                commands.extend(['--integrator_step_size', str(step_size)])
            if batch_size != "default":
                commands.extend(['--batch_size', str(batch_size)])
            if ee_selector != "default":
                commands.extend(["--edge_selector_type", ee_selector])
            for i in range(num_runs):
                print "Executing command %s for run %d/%d" % (str(commands), i + 1, num_runs)
                if not dry_run:
                    subprocess.call(map(str, commands))
    except (ValueError) as e:
        print "An error occured: %s. Skipping this test case." % str(e)


def execute_task(task):
    eval_info, test_case = task
    call_planner(eval_info[0], eval_info[1], eval_info[2], test_case, eval_info[3], eval_info[4])


def run_tests(testfile,
              log_path,
              num_runs,
              num_processes,
              algorithm_whitelist=None,
              graph_whitelist=None,
              algorithm_blacklist=None,
              graph_blacklist=None,
              dry_run=False):
    """Run the test cases specified in the yaml file <testfile>.

    Args:
        testfile (str): Path to yaml file containing test cases.
        log_path (str): Path to where to store logs
        num_runs (int): The number of executions of each algorithm to get runtime averages.
        num_processes (int): The number of parallel processes
        algorithm_whitelist (list of str), optional: A subset of algorithms to test only.
        graph_whitelist (list of str), optional: A subset of graphs to test only
        algorithm_blacklist (list of str), optional: A subset of algorithms to not test.
        graph_blacklist (list of str), optional: A subset of graphs to not test.
        dry_run(bool), optional: If True, only print test cases without executing them
    """
    # read testfile
    with open(testfile, 'r') as yaml_file:
        test_cases = yaml.load(yaml_file)

    def is_permitted(algo_name, graph_name):
        not_permitted = (algorithm_whitelist is not None and algo_name not in algorithm_whitelist) or\
                        (graph_whitelist is not None and graph_name not in graph_whitelist) or\
                        (algorithm_blacklist is not None and algo_name in algorithm_blacklist) or\
                        (graph_blacklist is not None and graph_name in graph_blacklist)
        return not not_permitted

    permitted_combinations = [(algo_name, graph_name, log_path, num_runs, dry_run)
                              for algo_name, graph_name in algorithm_graph_combinations
                              if is_permitted(algo_name, graph_name)]
    tasks = itertools.product(permitted_combinations, test_cases)

    pool = Pool(num_processes)
    pool.map(execute_task, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script to run a large collection of experiments on the multi-grasp motion planner across different domains."
        "The script takes as input a yaml-file containing test case information. A single test case is defined as:\n %s."
        % str(test_case_info))
    parser.add_argument('tests_file', help="Path to a yaml file specifying the test cases", type=str)
    parser.add_argument('log_path', type=str, help='Path to base folder of where to store the results')
    parser.add_argument('--num_runs', help="Number of runs to execute for runtime measurements.", type=int, default=10)
    parser.add_argument('--num_processes', help="Number of parallel processes to run.", type=int, default=4)
    parser.add_argument("--limit_algorithms",
                        help="Optionally limit the evaluation to the given list of algorithms",
                        nargs="*",
                        type=str)
    parser.add_argument("--limit_graphs",
                        help="Optionally limit the evaluation to the given list of graphs",
                        nargs="*",
                        type=str)
    parser.add_argument("--exclude_algorithms",
                        help="Optionally exclude the given list of algorithms",
                        nargs="*",
                        type=str)
    parser.add_argument("--exclude_graphs", help="Optionally exclude the given list of graphs", nargs="*", type=str)
    parser.add_argument("--dry_run",
                        help="If set, only print what test cases would be run, but do not execute any",
                        action='store_true')
    args = parser.parse_args()
    run_tests(args.tests_file, args.log_path, args.num_runs, args.num_processes, args.limit_algorithms,
              args.limit_graphs, args.exclude_algorithms, args.exclude_algorithms, args.dry_run)
