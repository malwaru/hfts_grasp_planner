#! /usr/bin/python
import IPython
import sys
import os
import yaml
import rospy
import time
import argparse
import random
import itertools
import numpy as np
import cProfile
import openravepy as orpy
# import hfts_grasp_planner.placement
from hfts_grasp_planner.utils import (is_dynamic_body, inverse_transform, get_manipulator_links, set_grasp,
                                      set_body_color, set_body_alpha, get_tf_interpolation, vec_angle_diff)
import hfts_grasp_planner.sdf.grid as grid_module
import hfts_grasp_planner.ik_solver as ik_module
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.robot as robot_sdf_module
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf_module
import hfts_grasp_planner.placement.afr_placement.placement_orientations as plcmnt_orientations_mod
import hfts_grasp_planner.placement.afr_placement.placement_regions as plcmnt_regions_mod
import hfts_grasp_planner.placement.afr_placement.core as afr_placement_mod
import hfts_grasp_planner.placement.afr_placement.multi_grasp as afr_dmg_mod
import hfts_grasp_planner.placement.afr_placement.statsrecording as statsrecording
import hfts_grasp_planner.placement.goal_sampler.random_sampler as rnd_sampler_mod
import hfts_grasp_planner.placement.goal_sampler.simple_mcts_sampler as simple_mcts_sampler_mod
import hfts_grasp_planner.placement.anytime_planner as anytime_planner_mod
import hfts_grasp_planner.placement.clearance as clearance_mod
import hfts_grasp_planner.placement.objectives as objectives_mod
import hfts_grasp_planner.placement.so3hierarchy as so3hierarchy
from hfts_grasp_planner.placement.placement_planning import SE3Hierarchy
from hfts_grasp_planner.sdf.visualization import visualize_occupancy_grid


def draw_volume(env, volume):
    return env.drawbox(0.5 * (volume[0] + volume[1]), 0.5 * (volume[1] - volume[0]), np.array([0.3, 0.3, 0.3, 0.3]))


def resolve_paths(problem_desc, yaml_file):
    global_yaml = str(yaml_file)
    if not os.path.isabs(global_yaml):
        cwd = os.getcwd()
        global_yaml = cwd + '/' + global_yaml
    head, _ = os.path.split(global_yaml)
    for key in [
            'or_env', 'occ_file', 'sdf_file', 'urdf_file', 'target_obj_file', 'grasp_file', 'robot_occtree',
            'robot_occgrid', 'reachability_path', 'robot_ball_desc', 'dmg_file', 'gripper_information', 'gripper_file'
    ]:
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
        if grasp_id >= len(grasp_yaml):
            raise IOError("Invalid grasp id: " + str(grasp_id))
        problem_desc['grasp_pose'] = grasp_yaml[grasp_id]['grasp_pose']
        # grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        # grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        # grasp_pose = utils.inverse_transform(grasp_pose)
        # problem_desc['grasp_pose'][:3] = grasp_pose[:3, 3]
        # problem_desc['grasp_pose'][3:] = orpy.quatFromRotationMatrix(grasp_pose)
        problem_desc['grasp_config'] = grasp_yaml[grasp_id]['grasp_config']


def show_solution(sol, target_obj):
    robot = sol.manip.GetRobot()
    set_grasp(sol.manip, target_obj, inverse_transform(sol.grasp_tf), sol.grasp_config)
    robot.SetDOFValues(sol.arm_config, sol.manip.GetArmIndices())
    env = robot.GetEnv()
    if env.CheckCollision(robot) or robot.CheckSelfCollision():
        rospy.logerr("ERROR: The shown solution is in collision!")
        import IPython
        IPython.embed()


def show_solutions(solutions, target_obj, sleep_time=0.1):
    for sol in solutions:
        robot = sol.manip.GetRobot()
        before_config = robot.GetDOFValues()
        before_tf = target_obj.GetTransform()
        show_solution(sol, target_obj)
        time.sleep(sleep_time)
        robot.SetDOFValues(before_config)
        robot.Release(target_obj)
        target_obj.SetTransform(before_tf)


def tf_matrix_to_yaml_dict(m):
    """Transform a 4x4 transformation matrix to store as yaml.

    Args:
        m (numpy array of shape (4, 4)): The matrix to store
    Returns:
        dict: a dictionary that can be serialized in yaml
    """
    return {'pos': map(float, list(m[:3, 3])), 'quat': map(float, list(orpy.quatFromRotationMatrix(m[:3, :3])))}


def grasp_to_yaml_dict(grasp):
    """Return a dictionary representation of the given grasp for yaml serializing.

    Args:
        grasp (placement.afr_placement.multi_grasp.Grasp): The grasp to format as dictionary.
    Returns:
        dict: A dictionary storing all relevant information about the given grasp.
    """
    return {
        'id': grasp.gid,
        'cost': float(grasp.distance),
        'config': map(float, list(grasp.config)),
        'oTe': tf_matrix_to_yaml_dict(grasp.oTe)
    }


def goal_to_yaml_dict(goal):
    """Return a dictionary representation of the given goal to serialize in yaml.

    Args:
        goal (PlacementGoal): The goal to represent as a dict.
    Returns:
        dict: A dictionary representation of the given goal.
    """
    return {
        'arm_config': map(float, list(goal.arm_config)),
        'grasp_id': goal.grasp_id,
        'objective_value': float(goal.objective_value)
    }


def store_goals(goals, grasp_set, waypoints, filename, reverse_task=False):
    """Store the given placements in a file.

    Args:
        goals (list of PlacementGoal): The pgoals to store.
        grasp_set (DMGGraspSet): The superset of grasps.
        waypoints (list of np.array): A set of additional waypoints to store.
        filename (str): Filename to store placements in.
        reverse_task (bool): If true, indicates that the goals represent the reverse task, retrieving
            an object and not placing
    """
    if goals:
        # first determine what grasps we have
        grasp_ids = set([goal.grasp_id for goal in goals])
        # collect grasp information
        grasps = [grasp_set.get_grasp(gid) for gid in grasp_ids]
        # store everything in a dict and save to file
        output_dict = {}
        output_dict['manipulator'] = str(goals[0].manip.GetName())
        output_dict['grasps'] = [grasp_to_yaml_dict(grasp) for grasp in grasps]
        output_dict['goals'] = [goal_to_yaml_dict(goal) for goal in goals]
        output_dict['waypoints'] = [map(float, list(wp)) for wp in waypoints]
        output_dict['reverse_task'] = reverse_task
        with open(filename, 'w') as output_file:
            yaml.dump(output_dict, output_file)


def compute_orientations(num_samples, orientation_cone):
    """Compute orientations that are within the given orientation cone using rejection sampling.

    Args:
        num_samples (int): THe number of candidate samples to draw.
        orientation_cone (tuple (np.array, float)): Orientation cone in the form (axis, angle) that
            limits the permitted orientations of the z-axis in world frame.
    Returns:
        list of numpy.array: A list of quatnerions representing different orientations within
            the orientation cone.
    """
    orientations = []
    for key in so3hierarchy.get_key_generator(so3hierarchy.get_min_depth(num_samples)):
        quat = so3hierarchy.get_quaternion(key)
        mat = orpy.matrixFromQuat(quat)
        delta_angle = abs(vec_angle_diff(mat[:3, 2], orientation_cone[0]))
        if delta_angle < orientation_cone[1]:
            orientations.append(quat)
    if len(orientations) > num_samples:
        return random.sample(orientations, num_samples)
    return orientations


def sample_waypoints(manip, ik_solver, volume, orientation_cone, num_waypoints, pos_delta=0.1, num_rot_samples=200):
    """Sample a collection of arm configurations such that the end-effector is located
    in a given volume. None of the configurations are checked for collisions.

    Args:
        manip (orpy.Manipulator): Manipulator to sample waypoint configurations for.
        ik_solver (IkSolver): IkSolver for the given manipulator.
        volume (tuple of np.array): (min, max), where min and max are points in 3D spanning
            a volume to sample poses in.
        orientation_cone (tuple (np.array, float)): (axis, angle), where axis is the desired
            orientation of the eef's z axis in world frame and angle the maximal permitted angle between
            this axis and the eef's z axis at a sampled waypoint
        num_waypoints (int): The total number of waypoints to sample.
        pos_delta (float): The distance between position samples along each axis.
        num_rot_samples (int): The number of eef-orientations to sample.
    Returns:
        list of np arrays: Arm configurations with that place the end-effector within the given volume.
    """
    waypoints = []
    # compute valid orientations
    orientations = compute_orientations(num_rot_samples, orientation_cone)
    print "Found %i valid orientations" % len(orientations)
    if len(orientations) == 0:
        raise ValueError("Failed to find any orientations within the given orientation cone " + str(orientation_cone))
    # compute a grid of positions
    xs = np.arange(volume[0][0], volume[1][0], pos_delta)
    ys = np.arange(volume[0][1], volume[1][1], pos_delta)
    zs = np.arange(volume[0][2], volume[1][2], pos_delta)
    positions = itertools.product(xs, ys, zs)
    # print "Sampled %i valid positions" % len(positions)
    # run over poses and compute ik
    poses = itertools.product(positions, orientations)
    reachable_poses = []
    for pose in poses:
        m = orpy.matrixFromQuat(pose[1])
        m[:3, 3] = pose[0]
        sol = ik_solver.compute_ik(m)
        if sol is not None:
            waypoints.append(sol)
            reachable_poses.append(pose)
    # if this hasn't given us enough waypoints yet, compute new ik solutions for the reachable poses
    while len(waypoints) < num_waypoints:
        print "Found %i reachable poses, but have only %i/%i configurations, keep computing new ones" % (
            len(reachable_poses), len(waypoints), num_waypoints)
        for pose in reachable_poses:
            m = orpy.matrixFromQuat(pose[1])
            m[:3, 3] = pose[0]
            sol = ik_solver.compute_ik(m)
            if sol is not None:
                waypoints.append(sol)
    print "Found %i waypoints" % len(waypoints)
    return random.sample(waypoints, num_waypoints)


if __name__ == "__main__":
    # NOTE If the OpenRAVE viewer is created too early, nothing works! Collision checks may be incorrect!
    parser = argparse.ArgumentParser("Script to generate a collection of placements in a given scene.")
    parser.add_argument('problem_desc', help="Path to a yaml file specifying what world, robot to use etc.", type=str)
    parser.add_argument('num_goals', help="Number of goals to sample", type=int)
    parser.add_argument('output_path', help="Filename in which to store generated goals.", type=str)
    parser.add_argument('--sample_waypoints',
                        help="Sample n waypoint configurations and store them with the goals.",
                        type=int,
                        default=0)
    args = parser.parse_args()
    with open(args.problem_desc, 'r') as f:
        problem_desc = yaml.load(f)
        resolve_paths(problem_desc, args.problem_desc)
        load_grasp(problem_desc)
    try:
        env = orpy.Environment()
        env.Load(problem_desc['or_env'])
        # load target object
        btarget_found = env.Load(problem_desc['target_obj_file'])
        if not btarget_found:
            raise ValueError("Could not load target object. Aborting")
        # ensure the object has a useful name
        target_obj_name = "target_object"
        env.GetBodies()[-1].SetName(target_obj_name)
        # load occupancy grid
        scene_occ = None
        try:
            scene_occ = grid_module.VoxelGrid.load(problem_desc['occ_file'])
        except IOError as e:
            rospy.logerr("Could not load %s. Please provide an occupancy grid of the scene." % problem_desc['occ_file'])
            rospy.logerr("There is a script to create one!")
            sys.exit(0)
        placement_volume = (np.array(problem_desc['plcmnt_volume'][:3]), np.array(problem_desc['plcmnt_volume'][3:]))
        occ_target_volume = scene_occ.get_subset(placement_volume[0], placement_volume[1])
        # create scene sdf
        dynamic_bodies = [body for body in env.GetBodies() if is_dynamic_body(body)]
        scene_sdf = sdf_module.SceneSDF(env, [], excluded_bodies=dynamic_bodies)
        if os.path.exists(problem_desc['sdf_file']):
            now = time.time()
            scene_sdf.load(problem_desc['sdf_file'])
            rospy.logdebug("Loading scene sdf took %fs" % (time.time() - now))
        else:
            rospy.logerr("Could not load %s. Please provide a signed distance field of the scene." %
                         problem_desc['sdf_file'])
            rospy.logerr("There is a script to create one!")
            sys.exit(0)
        # extract placement orientations
        target_object = env.GetKinBody(target_obj_name)
        orientations = plcmnt_orientations_mod.compute_placement_orientations(target_object)
        # extract placement regions
        gpu_kit = plcmnt_regions_mod.PlanarRegionExtractor()
        surface_grid, labels, num_regions, regions = gpu_kit.extract_planar_regions(occ_target_volume,
                                                                                    max_region_size=0.3)
        if num_regions == 0:
            print "No placement regions found"
            sys.exit(0)
        # compute surface distance field
        obj_radius = np.linalg.norm(target_object.ComputeAABB().extents())
        global_region_info = plcmnt_regions_mod.PlanarRegionExtractor.compute_surface_distance_field(
            surface_grid, 2.0 * obj_radius)
        # prepare robot data
        robot = env.GetRobot(problem_desc['robot_name'])
        # set initial grasp and create grasp set
        # grasp_pose is oTe
        manip = robot.GetActiveManipulator()
        grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        set_grasp(manip, target_object, inverse_transform(grasp_pose), problem_desc['grasp_config'])
        gripper_info = afr_dmg_mod.load_gripper_info(problem_desc['gripper_information'], manip.GetName())
        grasp_set = afr_dmg_mod.DMGGraspSet(manip, target_object, problem_desc['target_obj_file'], gripper_info,
                                            problem_desc['dmg_file'])
        # prepare manipulator data
        link_names = []
        if 'manipulator' in problem_desc:
            robot.SetActiveManipulator(problem_desc['manipulator'])
        ik_solver = ik_module.IKSolver(manip, problem_desc['urdf_file'])
        manip_data = {}
        manip_data[manip.GetName()] = afr_dmg_mod.MultiGraspAFRRobotBridge.ManipulatorData(
            manip, ik_solver, grasp_set, gripper_info['gripper_file'])
        manip_links = [link.GetName() for link in get_manipulator_links(manip)]
        # remove base link - it does not move so
        manip_links.remove(manip.GetBase().GetName())
        link_names.extend(manip_links)
        # load robot ball approximation
        robot_ball_approx = robot_sdf_module.RobotBallApproximation(robot, problem_desc['robot_ball_desc'])
        # build robot_octree
        try:
            now = time.time()
            robot_occgrid = robot_sdf_module.RobotOccupancyGrid.load(base_file_name=problem_desc['robot_occgrid'],
                                                                     robot=robot,
                                                                     link_names=link_names)
            rospy.logdebug("Loading robot occgrid took %fs" % (time.time() - now))
        except IOError:
            now = time.time()
            robot_occgrid = robot_sdf_module.RobotOccupancyGrid(problem_desc['parameters']['occ_tree_cell_size'],
                                                                robot=robot,
                                                                link_names=link_names)
            robot_occgrid.save(problem_desc['robot_occgrid'])
            rospy.logdebug("Creating robot occgrid took %fs" % (time.time() - now))
        # prepare robot data
        urdf_content = None
        with open(problem_desc['urdf_file'], 'r') as urdf_file:
            urdf_content = urdf_file.read()
        robot_data = afr_placement_mod.AFRRobotBridge.RobotData(robot, robot_occgrid, manip_data, urdf_content,
                                                                robot_ball_approx)
        # create object data
        obj_occgrid = kinbody_sdf_module.RigidBodyOccupancyGrid(problem_desc['parameters']['occ_tree_cell_size'],
                                                                target_object.GetLinks()[0])
        obj_occgrid.setup_cuda_sdf_access(scene_sdf)
        object_data = afr_placement_mod.AFRRobotBridge.ObjectData(target_object, obj_occgrid)
        # create objective function
        now = time.time()
        if 'objective_fn' in problem_desc:
            if problem_desc['objective_fn'] == 'minimize_clearance':
                obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid, b_max=False)
            elif problem_desc['objective_fn'] == 'maximize_clearance':
                obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid, b_max=True)
            elif problem_desc['objective_fn'] == 'deep_shelf':
                obj_fn = objectives_mod.DeepShelfObjective(target_object, occ_target_volume, obj_occgrid, b_max=True)
        else:
            obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid, b_max=False)
        # create afr hierarchy
        hierarchy = afr_placement_mod.AFRHierarchy([manip], regions, orientations, so2_depth=4, so2_branching=4)
        # create afr bridge
        afr_bridge = afr_dmg_mod.MultiGraspAFRRobotBridge(afr_hierarchy=hierarchy,
                                                          robot_data=robot_data,
                                                          object_data=object_data,
                                                          objective_fn=obj_fn,
                                                          global_region_info=global_region_info,
                                                          scene_sdf=scene_sdf,
                                                          parameters=problem_desc['parameters'])
        # create goal sampler
        goal_sampler = simple_mcts_sampler_mod.SimpleMCTSPlacementSampler(hierarchy,
                                                                          afr_bridge,
                                                                          afr_bridge,
                                                                          afr_bridge, [manip.GetName()],
                                                                          debug_visualizer=None,
                                                                          parameters=problem_desc["parameters"])
        # sample goals
        solutions, num_solutions = goal_sampler.sample(args.num_goals, 1000000)
        if num_solutions < args.num_goals:
            print "Sorry, only found %i solutions (%i queried)" % (num_solutions, args.num_goals)
        # sample waypoints
        waypoints = []
        if args.sample_waypoints:
            print "Sampling %i waypoints" % args.sample_waypoints
            # TODO use different volume? Maybe enlarged? Bounding box of goals?
            waypoints = sample_waypoints(manip, ik_solver, placement_volume, (np.array([0, 1, 0]), 1.0),
                                         args.sample_waypoints)
        store_goals(solutions[manip.GetName()], grasp_set, waypoints, args.output_path)
        # IPython.embed()
    finally:
        orpy.RaveDestroy()
