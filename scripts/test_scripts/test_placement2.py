import IPython
import sys
import os
import yaml
import rospy
import time
import argparse
import numpy as np
import openravepy as orpy
# import hfts_grasp_planner.placement
from hfts_grasp_planner.utils import is_dynamic_body, inverse_transform, get_manipulator_links
import hfts_grasp_planner.sdf.grid as grid_module
import hfts_grasp_planner.ik_solver as ik_module
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.robot as robot_sdf_module
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf_module
import hfts_grasp_planner.placement.arpo_placement.placement_orientations as plcmnt_orientations_mod
import hfts_grasp_planner.placement.arpo_placement.placement_regions as plcmnt_regions_mod
import hfts_grasp_planner.placement.arpo_placement.core as arpo_placement_mod
import hfts_grasp_planner.placement.goal_sampler.random_sampler as rnd_sampler_mod
import hfts_grasp_planner.placement.goal_sampler.mcts_sampler as mcts_sampler_mod
import hfts_grasp_planner.placement.anytime_planner as anytime_planner_mod
import hfts_grasp_planner.placement.reachability as rmap_mod
from hfts_grasp_planner.sdf.visualization import visualize_occupancy_grid


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
    for key in ['or_env', 'occ_file', 'sdf_file', 'urdf_file', 'data_path', 'gripper_file', 'grasp_file',
                'robot_occtree', 'reachability_path']:
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


def show_solution(sol, target_obj):
    robot = sol.manip.GetRobot()
    robot.SetDOFValues(sol.arm_config, sol.manip.GetArmIndices())
    target_obj.SetTransform(sol.obj_tf)


def show_solutions(solutions, target_obj, sleep_time=0.1):
    for sol in solutions:
        robot = sol.manip.GetRobot()
        before_config = robot.GetDOFValues()
        show_solution(sol, target_obj)
        time.sleep(sleep_time)
        robot.SetDOFValues(before_config)


def or_motion_planner(sol, manip_data, target_obj):
    robot = sol.manip.GetRobot()
    robot.SetActiveManipulator(sol.manip.GetName())
    # first attach target obj to manipulator
    manipd = manip_data[sol.manip.GetName()]
    target_obj.SetTransform(np.dot(sol.manip.GetEndEffectorTransform(), manipd.grasp_tf))
    robot.Grab(target_obj)
    # now plan
    manipprob = orpy.interfaces.BaseManipulation(robot)
    manipprob.MoveManipulator(goal=sol.arm_config)
    robot.WaitForController(0)
    # release object again
    robot.Release(target_obj)


def show_traj(robot, traj, goal, target_obj):
    # first attach target obj to manipulator
    target_obj.SetTransform(np.dot(goal.manip.GetEndEffectorTransform(), inverse_transform(goal.grasp_tf)))
    robot.SetActiveManipulator(goal.manip.GetName())
    robot.Grab(target_obj)
    # now plan
    robot.GetController().SetPath(traj)
    robot.WaitForController(0)
    # release object again
    robot.Release(target_obj)


def plan(planner, body, it):
    now = time.time()
    traj, goal = planner.plan(it, body)
    print "Planning took %fs" % (time.time() - now)
    return traj, goal


if __name__ == "__main__":
    # NOTE If the OpenRAVE viewer is created too early, nothing works! Collision checks may be incorrect!
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_desc', help="Path to a yaml file specifying what world, robot to use etc.", type=str)
    parser.add_argument('--debug', help="If provided, run in debug mode", action="store_true")
    parser.add_argument('--show_plcmnt_volume', help="If provided, visualize placement volume", action="store_true")
    parser.add_argument('--show_sdf_volume', help="If provided, visualize sdf volume", action="store_true")
    parser.add_argument('--show_sdf', help="If provided, visualize sdf", action="store_true")
    args = parser.parse_args()
    log_level = rospy.DEBUG if args.debug else rospy.INFO
    rospy.init_node("TestPlacement2", anonymous=True, log_level=log_level)
    with open(args.problem_desc, 'r') as f:
        problem_desc = yaml.load(f)
        resolve_paths(problem_desc, args.problem_desc)
        load_grasp(problem_desc)

    try:
        env = orpy.Environment()
        env.Load(problem_desc['or_env'])
        # reset object pose, if demanded
        # if 'initial_obj_pose' in problem_desc:
        #     tb = env.GetKinBody(problem_desc['target_name'])
        #     tf = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        #     tf[:3, 3] = problem_desc['initial_obj_pose'][:3]
        #     tb.SetTransform(tf)

        scene_occ = None
        try:
            scene_occ = grid_module.VoxelGrid.load(problem_desc['occ_file'])
        except IOError as e:
            rospy.logerr("Could not load %s. Please provide an occupancy grid of the scene." % problem_desc['occ_file'])
            rospy.logerr("There is a script to create one!")
            sys.exit(0)

        dynamic_bodies = [body for body in env.GetBodies() if is_dynamic_body(body)]
        scene_sdf = sdf_module.SceneSDF(env, [], excluded_bodies=dynamic_bodies)
        # scene_sdf = sdf_module.SceneSDF(env, [], excluded_bodies=[
                                        # problem_desc['robot_name'], problem_desc['target_name']])
        if os.path.exists(problem_desc['sdf_file']):
            now = time.time()
            scene_sdf.load(problem_desc['sdf_file'])
            rospy.logdebug("Loading scene sdf took %fs" % (time.time() - now))
        else:
            rospy.logerr("Could not load %s. Please provide a signed distance field of the scene." % problem_desc['sdf_file'])
            rospy.logerr("There is a script to create one!")
            sys.exit(0)

        target_obj_name = problem_desc['target_name']
        # placement_planner = pp_module.PlacementGoalPlanner(problem_desc['data_path'], env, scene_sdf)
        # placement_volume = (np.array([-0.35, 0.55, 0.6]), np.array([0.23, 0.8, 0.77]))  # on top of shelf
        # placement_volume = (np.array([-0.35, 0.55, 0.42]), np.array([0.53, 0.9, 0.77]))  # all of the shelf
        # placement_volume = (np.array([-0.35, 0.55, 0.42]), np.array([0.53, 0.9, 0.64]))  # inside shelf
        # placement_volume = (np.array([-0.35, 0.4, 0.42]), np.array([0.53, 0.55, 0.64]))  # front of shelf
        # placement_volume = (np.array([-0.35, 0.55, 0.44]), np.array([0.53, 0.9, 0.49]))  # bottom inside shelf
        # placement_volume = (np.array([0.24, 0.58, 0.51]), np.array([0.29, 0.8, 0.55]))  # volume in small gap
        placement_volume = (np.array(problem_desc['plcmnt_volume'][:3]), np.array(problem_desc['plcmnt_volume'][3:]))
        occ_target_volume = scene_occ.get_subset(placement_volume[0], placement_volume[1])
        # extract placement regions
        gpu_kit = plcmnt_regions_mod.PlanarRegionExtractor()
        surface_grid, labels, num_regions, regions = gpu_kit.extract_planar_regions(
            occ_target_volume, max_region_size=0.2)
        sufrace_distance_grid = plcmnt_regions_mod.PlanarRegionExtractor.compute_surface_distance_field(surface_grid)
        # extract placement orientations
        target_object = env.GetKinBody(target_obj_name)
        orientations = plcmnt_orientations_mod.compute_placement_orientations(target_object)
        # prepare robot data
        robot = env.GetRobot(problem_desc['robot_name'])
        # extract manipulators
        link_names = []
        manip_data = {}
        manips = robot.GetManipulators()
        for manip in manips:
            ik_solver = ik_module.IKSolver(manip, problem_desc['urdf_file'])
            # TODO have different grasp poses for each manipulator
            grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
            grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
            rmap = rmap_mod.ReachabilityMap(manip, ik_solver)
            try:
                filename = problem_desc["reachability_path"] + '/' + robot.GetName() + '_' + manip.GetName() + '.npy'
                rmap.load(filename)
            except IOError:
                rospy.logerr("Could not load reachability map for %s from file %s. Please provide one!" % (manip.GetName(), filename))
                sys.exit(1)
            manip_data[manip.GetName()] = arpo_placement_mod.ARPORobotBridge.ManipulatorData(
                manip, ik_solver, rmap, grasp_pose, problem_desc['grasp_config'])
            manip_links = [link.GetName() for link in get_manipulator_links(manip)]
            # remove base link - it does not move so
            manip_links.remove(manip.GetBase().GetName())
            link_names.extend(manip_links)
        # build robot_octree
        try:
            now = time.time()
            robot_octree = robot_sdf_module.RobotOccupancyOctree.load(base_file_name=problem_desc['robot_occtree'],
                                                                      robot=robot, link_names=link_names)
            rospy.logdebug("Loading robot octree took %fs" % (time.time() - now))
        except IOError:
            robot_octree = robot_sdf_module.RobotOccupancyOctree(
                problem_desc['parameters']['occ_tree_cell_size'], robot, link_names)
            robot_octree.save(problem_desc['robot_occtree'])
        robot_data = arpo_placement_mod.ARPORobotBridge.RobotData(robot, robot_octree, manip_data)
        # create object data
        obj_octree = kinbody_sdf_module.OccupancyOctree(
            problem_desc['parameters']['occ_tree_cell_size'], target_object.GetLinks()[0])
        object_data = arpo_placement_mod.ARPORobotBridge.ObjectData(target_object, obj_octree)
        # visualize placement regions
        env.SetViewer('qtcoin')  # WARNING: IK solvers also need to be created before setting the viewer
        handles = []
        # handles.append(draw_volume(env, placement_volume))
        # handles = visualize_occupancy_grid(env, surface_grid, color=np.array([1.0, 0.0, 0.0, 0.2]))
        handles.extend(plcmnt_regions_mod.visualize_plcmnt_regions(
            env, regions, height=occ_target_volume.get_cell_size(), level=1))
        print "Check the placement regions!"
        IPython.embed()
        if num_regions == 0:
            print "No placement regions found"
            sys.exit(0)
        handles = []
        # create arpo hierarchy
        hierarchy = arpo_placement_mod.ARPOHierarchy(manips, regions, orientations, 4)
        arpo_bridge = arpo_placement_mod.ARPORobotBridge(arpo_hierarchy=hierarchy, robot_data=robot_data,
                                                         object_data=object_data, objective_fn=None,
                                                         contact_point_distances=sufrace_distance_grid, scene_sdf=scene_sdf)
        # random_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, arpo_bridge, arpo_bridge, arpo_bridge, [
        #                                                         manip.GetName() for manip in manips])
        mcts_sampler = mcts_sampler_mod.MCTSPlacementSampler(hierarchy, arpo_bridge, arpo_bridge, arpo_bridge, [
                                                                manip.GetName() for manip in manips])

        motion_planner = anytime_planner_mod.AnyTimePlacementPlanner(mcts_sampler, manips)
        # traj, goal = plan(motion_planner, target_object, 10)
        solutions = mcts_sampler.sample(10, 100)
        IPython.embed()
    finally:
        orpy.RaveDestroy()
