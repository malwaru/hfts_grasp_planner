import IPython
import sys
import os
import yaml
import rospy
import argparse
import numpy as np
import openravepy as orpy
# import hfts_grasp_planner.placement
from hfts_grasp_planner.utils import is_dynamic_body
import hfts_grasp_planner.sdf.grid as grid_module
import hfts_grasp_planner.ik_solver as ik_module
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf_module
import hfts_grasp_planner.placement.arpo_placement.placement_orientations as plcmnt_orientations_mod
import hfts_grasp_planner.placement.arpo_placement.placement_regions as plcmnt_regions_mod
import hfts_grasp_planner.placement.arpo_placement.core as arpo_placement_mod
import hfts_grasp_planner.placement.goal_sampler.random_sampler as rnd_sampler_mod


def draw_volume(env, volume):
    return env.drawbox(0.5 * (volume[0] + volume[1]), 0.5 * (volume[1] - volume[0]), np.array([0.3, 0.3, 0.3, 0.3]))


# def execute_placement_planner(placement_planner, body):
#     result = placement_planner.sample(placement_planner.get_max_depth())
#     # print 'Best val is ', result.quality_value
#     if result.is_valid():
#         if result.is_goal():
#             print "Result is a goal."
#         else:
#             print "Result is valid, but not a goal."
#     else:
#         print "Result is invalid"
#     if result.obj_pose is not None:
#         body.SetTransform(result.obj_pose)
#     else:
#         print "Failed to find a solution"
#     return result


def resolve_paths(problem_desc, yaml_file):
    global_yaml = str(yaml_file)
    if not os.path.isabs(global_yaml):
        cwd = os.getcwd()
        global_yaml = cwd + '/' + global_yaml
    head, _ = os.path.split(global_yaml)
    for key in ['or_env', 'occ_file', 'sdf_file', 'urdf_file', 'data_path', 'gripper_file', 'grasp_file']:
        if key in problem_desc:
            problem_desc[key] = os.path.normpath(head + '/' + problem_desc[key])


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


if __name__ == "__main__":
    # NOTE If the OpenRAVE viewer is created too early, nothing works! Collision checks may be incorrect!
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_desc', help="Path to a yaml file specifying what world, robot to use etc.", type=str)
    parser.add_argument('--debug', help="If provided, run in debug mode", action="store_true")
    parser.add_argument('--show_plcmnt_volume', help="If provided, visualize placement volume", action="store_true")
    parser.add_argument('--show_sdf_volume', help="If provided, visualize sdf volume", action="store_true")
    parser.add_argument('--show_sdf', help="If provided, visualize sdf", action="store_true")
    args = parser.parse_args()
    rospy.init_node("TestPlacement2", anonymous=True, log_level=rospy.DEBUG)
    with open(args.problem_desc, 'r') as f:
        problem_desc = yaml.load(f)
        resolve_paths(problem_desc, args.problem_desc)
        load_grasp(problem_desc)

    env = orpy.Environment()
    env.Load(problem_desc['or_env'])
    # reset object pose, if demanded
    # if 'initial_obj_pose' in problem_desc:
    #     tb = env.GetKinBody(problem_desc['target_name'])
    #     tf = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
    #     tf[:3, 3] = problem_desc['initial_obj_pose'][:3]
    #     tb.SetTransform(tf)

    dynamic_bodies = [body for body in env.GetBodies() if is_dynamic_body(body)]
    scene_occ = None
    scene_sdf = sdf_module.SceneSDF(env, [], excluded_bodies=dynamic_bodies)
    try:
        scene_occ = grid_module.VoxelGrid.load(problem_desc['occ_file'])
    except IOError as e:
        print "Could not load %s. Please provide an occupancy grid of the scene."
        print "There is a script to create one!" % problem_desc['occ_file']
        sys.exit(0)

    if os.path.exists(problem_desc['sdf_file']):
        scene_sdf.load(problem_desc['sdf_file'])
    else:
        print "Could not load %s. Please provide a signed distance field of the scene. There is a script to create one!" % problem_desc[
            'sdf_file']

    target_obj_name = problem_desc['target_name']
    # placement_planner = pp_module.PlacementGoalPlanner(problem_desc['data_path'], env, scene_sdf)
    # placement_volume = (np.array([-0.35, 0.55, 0.66]), np.array([0.53, 0.9, 0.77]))  # on top of shelf
    # placement_volume = (np.array([-0.35, 0.55, 0.42]), np.array([0.53, 0.9, 0.77]))  # all of the shelf
    placement_volume = (np.array([-0.35, 0.55, 0.42]), np.array([0.53, 0.9, 0.64]))  # inside shelf
    # placement_volume = (np.array([-0.35, 0.4, 0.42]), np.array([0.53, 0.55, 0.64]))  # front of shelf
    # placement_volume = (np.array([-0.35, 0.55, 0.44]), np.array([0.53, 0.9, 0.49]))  # bottom inside shelf
    # placement_volume = (np.array([0.24, 0.58, 0.51]), np.array([0.29, 0.8, 0.55]))  # volume in small gap
    # placement_volume = (np.array(problem_desc['plcmnt_volume'][:3]), np.array(problem_desc['plcmnt_volume'][3:]))
    occ_target_volume = scene_occ.get_subset(placement_volume[0], placement_volume[1])
    # extract placement regions
    gpu_kit = plcmnt_regions_mod.PlanarRegionExtractor()
    surface_grid, labels, num_regions, regions = gpu_kit.extract_planar_regions(occ_target_volume, max_region_size=0.2)
    # extract placement orientations
    body = env.GetKinBody(target_obj_name)
    orientations = plcmnt_orientations_mod.compute_placement_orientations(body)
    # visualize placement regions
    env.SetViewer('qtcoin')
    handles = []
    # handles.append(draw_volume(env, placement_volume))
    handles.extend(plcmnt_regions_mod.visualize_plcmnt_regions(
        env, regions, height=occ_target_volume.get_cell_size(), level=1))
    print "Check the placement regions!"
    IPython.embed()
    handles = []
    # create arpo hierarchy
    robot = env.GetRobot(problem_desc['robot_name'])
    manips = robot.GetManipulators()
    hierarchy = arpo_placement_mod.ARPOHierarchy(manips, regions, orientations, 4)
    manip_data = {}
    for manip in manips:
        ik_solver = ik_module.IKSolver(manip, problem_desc['urdf_path'])
        # TODO have different grasp poses for each manipulator
        grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        manip_data[manip.GetName()] = arpo_placement_mod.ARPORobotBridge.ManipulatorData(
            manip, ik_solver, None, grasp_pose, problem_desc['grasp_config'])
    arpo_bridge = arpo_placement_mod.ARPORobotBridge(hierarchy, manip_data, body, None, surface_grid)
    random_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, arpo_bridge, arpo_bridge, arpo_bridge)
    IPython.embed()
