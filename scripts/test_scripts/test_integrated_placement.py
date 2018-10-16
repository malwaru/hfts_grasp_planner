#! /usr/bin/python

import os
import yaml
import rospy
import IPython
import argparse
import rospy
import numpy as np
import openravepy as orpy
import hfts_grasp_planner.integrated_plcmt_planner as ipp_module
import hfts_grasp_planner.ik_solver as ik_module
import hfts_grasp_planner.utils as utils
import hfts_grasp_planner.sdf.core as sdf_module


def draw_volume(env, volume, color=None):
    if color is None:
        color = np.array([0.3, 0.3, 0.3, 0.3])
    return env.drawbox(0.5 * (volume[0] + volume[1]), 0.5 * (volume[1] - volume[0]), color)


# Grasp map for Robotiq hand
# grasp_map = {
#     'bunny': {
#         'grasp_pose': np.array([[-4.89838933e-09,   4.10908549e-02,   9.99155414e-01, -1.61563370e-02],
#                                 [2.38317911e-07,  -9.99155414e-01,   4.10908549e-02, 2.94656865e-02],
#                                 [1.00000000e+00,   2.38317910e-07,  -4.89843450e-09, 1.66712295e-01],
#                                 [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]]),
#         'start_config':  np.array([0.,  0.,  0.,  0.,  0., 0.,  0.13168141,  0.72114204]),
#     },
#     # TODO other objects
# }
def show_solution(result, planner):
    if result is not None:
        traj = utils.path_to_trajectory(planner._robot, result)
        controller = planner._robot.GetController()
        controller.SetPath(traj)
        planner._robot.WaitForController(traj.GetDuration())
        controller.Reset()


def load_grasp(problem_desc):
    if 'grasp_pose' in problem_desc and 'grasp_config' in problem_desc:
        return
    assert 'grasp_file' in problem_desc
    with open(problem_desc['grasp_file']) as grasp_file:
        grasp_yaml = yaml.load(grasp_file)
        if 'grasp_id' in problem_desc:
            grasp_id = problem_desc['grasp_id']
        else:
            grasp_id = 0
        problem_desc['grasp_pose'] = grasp_yaml[grasp_id]['grasp_pose']
        # grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        # grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        # grasp_pose = utils.inverse_transform(grasp_pose)
        # problem_desc['grasp_pose'][:3] = grasp_pose[:3, 3]
        # problem_desc['grasp_pose'][3:] = orpy.quatFromRotationMatrix(grasp_pose)
        problem_desc['grasp_config'] = grasp_yaml[grasp_id]['grasp_config']


def execute_placement_planner(planner, volume, problem_desc, ik_solver):
    robot = planner._env.GetRobot(problem_desc["robot_name"])
    manip = robot.GetManipulator(problem_desc["manip_name"])
    target_object = planner._env.GetKinBody(problem_desc["target_name"])
    target_object.Enable(False)
    grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
    grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
    object_pose = target_object.GetTransform()
    start_config, b_col_free = ik_solver.compute_collision_free_ik(np.dot(object_pose, grasp_pose))
    if not b_col_free:
        print "Could not execute planner, because there is no collision-free ik solution for the initial eef-pose."
        return
    robot.SetDOFValues(start_config, dofindices=manip.GetArmIndices())
    robot.SetDOFValues(problem_desc['grasp_config'], dofindices=manip.GetGripperIndices())
    target_object.Enable(True)
    path, pose = planner.plan_placement(problem_desc["target_name"], volume, utils.inverse_transform(grasp_pose),
                                        problem_desc['grasp_config'], problem_desc["time_limit"])
    if path:
        print "Success! Found a solution:", path
        # robot.SetActiveDOFValues(path[-1].get_configuration())
        # TODO only show solution when we want it
        show_solution(path, planner)
        # TODO reset to start configuration
    else:
        print "Failed! No solution found."
    return path


def resolve_paths(problem_desc, yaml_file):
    global_yaml = str(yaml_file)
    if not os.path.isabs(global_yaml):
        cwd = os.getcwd()
        global_yaml = cwd + '/' + global_yaml
    head, _ = os.path.split(global_yaml)
    for key in ['or_env', 'sdf_file', 'urdf_file', 'data_path', 'gripper_file', 'grasp_file', 'reachability_path']:
        if key in problem_desc:
            problem_desc[key] = os.path.normpath(head + '/' + problem_desc[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_desc', help="Path to a yaml file specifying what world, robot to use etc.", type=str)
    parser.add_argument('--debug', help="If provided, run in debug mode", action="store_true")
    parser.add_argument('--show_plcmnt_volume', help="If provided, visualize placement volume", action="store_true")
    parser.add_argument('--show_sdf_volume', help="If provided, visualize sdf volume", action="store_true")
    parser.add_argument('--show_sdf', help="If provided, visualize sdf", action="store_true")
    args = parser.parse_args()
    log_level = rospy.WARN
    if args.debug:
        log_level = rospy.DEBUG
    rospy.init_node("IntegratedPlacementPlannerTest", anonymous=True, log_level=log_level)
    with open(args.problem_desc, 'r') as f:
        problem_desc = yaml.load(f)
        resolve_paths(problem_desc, args.problem_desc)
        load_grasp(problem_desc)
    # Grasp settings for bunny
    # obj_pose = np.array([[1.00000000e+00,   2.38317910e-07,  -4.89843428e-09, 7.30712295e-01],
    #                      [2.38317911e-07,  -9.99155414e-01,   4.10908549e-02, 2.94656865e-02],
    #                      [4.89838955e-09,  -4.10908549e-02,  -9.99155414e-01, 8.06156337e-01],
    #                      [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])
    # grasp_pose = np.array([[-4.89838933e-09,   4.10908549e-02,   9.99155414e-01, -1.61563370e-02],
    #                        [2.38317911e-07,  -9.99155414e-01,   4.10908549e-02, 2.94656865e-02],
    #                        [1.00000000e+00,   2.38317910e-07,  -4.89843450e-09, 1.66712295e-01],
    #                        [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])

    sdf_volume_robot = (np.array(problem_desc['sdf_volume'][:3]), np.array(problem_desc['sdf_volume'][3:]))
    rmap_file = problem_desc["reachability_path"] + "/" + \
        problem_desc["robot_name"] + "_" + problem_desc["manip_name"] + ".npy"
    planner = ipp_module.IntegratedPlacementPlanner(problem_desc['or_env'],
                                                    problem_desc['sdf_file'], sdf_volume_robot,
                                                    problem_desc['data_path'],
                                                    problem_desc['robot_name'],
                                                    problem_desc['manip_name'],
                                                    gripper_file=problem_desc['gripper_file'],
                                                    robot_urdf_file=problem_desc['urdf_file'],
                                                    obj_urdf_path=problem_desc['urdf_path'],
                                                    draw_search_tree=args.debug,
                                                    draw_hierarchy=args.debug,
                                                    reachability_path=rmap_file,
                                                    **problem_desc['parameters'])
    # create an IK solver so we can compute the start configuration
    ik_solver = ik_module.IKSolver(planner._env, problem_desc["robot_name"], problem_desc["urdf_file"])
    # set a placement target volume
    placement_volume = (np.array(problem_desc["plcmnt_volume"][:3]),
                        np.array(problem_desc["plcmnt_volume"][3:]))
    # transform robot frame sdf volume to world frame for visualization
    robot_tf = planner._robot.GetTransform()
    tvals = np.array([np.dot(robot_tf[:3, :3], sdf_volume_robot[0]) + robot_tf[:3, 3],
                      np.dot(robot_tf[:3, :3], sdf_volume_robot[1]) + robot_tf[:3, 3]])
    sdf_volume = (np.min(tvals, axis=0), np.max(tvals, axis=0))
    planner._env.SetViewer('qtcoin')
    # reset object pose, if provided
    if 'initial_obj_pose' in problem_desc:
        tb = planner._env.GetKinBody(problem_desc['target_name'])
        tf = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        tf[:3, 3] = problem_desc['initial_obj_pose'][:3]
        tb.SetTransform(tf)

    # planner._plcmt_planner._placement_heuristic._env.SetViewer('qtcoin')
    # planner._plcmt_planner._leaf_stage.robot_interface._env.SetViewer('qtcoin')
    if args.show_plcmnt_volume:
        handle_plcmnt = draw_volume(planner._env, placement_volume, color=[0.3, 0.3, 0.3, 0.3])
    if args.show_sdf_volume and not args.show_sdf:
        handle_sdf = draw_volume(planner._env, sdf_volume, color=[0.3, 0.0, 0.0, 0.3])
    if args.show_sdf:
        sdf_vis = sdf_module.ORSDFVisualization(planner._env)
        sdf_vis.visualize(planner._scene_sdf, sdf_volume, resolution=0.05, max_sat_value=0.7, style='sprites')

    # IPython.embed()
    # handle = None
    # execute_placement_planner(planner, placement_volume, problem_desc, ik_solver)
    IPython.embed()
