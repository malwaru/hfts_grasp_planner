#! /usr/bin/python
import IPython
import sys
import os
import yaml
import rospy
import time
import argparse
import numpy as np
import cProfile
import openravepy as orpy
# import hfts_grasp_planner.placement
from hfts_grasp_planner.utils import (is_dynamic_body, inverse_transform, get_manipulator_links,
                                      set_grasp, set_body_color, set_body_alpha, get_tf_interpolation)
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
import hfts_grasp_planner.placement.goal_sampler.mcts_visualization as mcts_visualizer_mod
import hfts_grasp_planner.placement.anytime_planner as anytime_planner_mod
import hfts_grasp_planner.placement.clearance as clearance_mod
import hfts_grasp_planner.placement.objectives as objectives_mod
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
    for key in ['or_env', 'occ_file', 'sdf_file', 'urdf_file', 'target_obj_file', 'grasp_file',
                'robot_occtree', 'robot_occgrid', 'reachability_path', 'robot_ball_desc', 'dmg_file',
                'gripper_information']:
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


def camera_slide(viewer, goal_tf, duration, fps=25):
    num_steps = duration * fps
    for tf in get_tf_interpolation(viewer.GetCameraTransform(), goal_tf, num_steps):
        viewer.SetCamera(tf)
        time.sleep(1.0 / float(fps))


def camera_slide_list(viewer, tfs, duration, delays=None, fps=25):
    travel_dist = 0.0
    dist_segments = []
    last_tf = viewer.GetCameraTransform()
    for tf in tfs:
        dist_segments.append(np.linalg.norm(last_tf[:3, 3] - tf[:3, 3]))
        travel_dist += dist_segments[-1]
        last_tf = tf
    speed = travel_dist / duration
    if delays is None:
        delays = len(tfs) * [0.0]
    for tf, delay, seg in zip(tfs, delays, dist_segments):
        seg_duration = seg / speed
        camera_slide(viewer, tf, seg_duration, fps=fps)
        last_tf = viewer.GetCameraTransform()
        time.sleep(delay)


def show_ghost_traj(ghost, traj, goal, target_obj, vel_scale=0.4):
    dofs = goal.manip.GetArmDOF()
    ghost.SetActiveDOFs(goal.manip.GetArmIndices())
    ghost_traj = orpy.RaveCreateTrajectory(ghost.GetEnv(), '')
    ghost_traj.Init(ghost.GetActiveConfigurationSpecification('linear'))
    for widx in xrange(traj.GetNumWaypoints()):
        ghost_traj.Insert(widx, traj.GetWaypoint(widx)[:dofs])
    # time trajectory
    vel_limits = ghost.GetDOFVelocityLimits()
    ghost.SetDOFVelocityLimits(vel_scale * vel_limits)
    orpy.planningutils.RetimeTrajectory(ghost_traj, hastimestamps=False)
    ghost.SetDOFVelocityLimits(vel_limits)
    ghost_manip = ghost.GetManipulator(goal.manip.GetName())
    set_grasp(ghost_manip, target_obj, inverse_transform(goal.grasp_tf), goal.grasp_config)
    ghost.GetController().SetPath(ghost_traj)
    ghost.WaitForController(0)
    ghost.Release(target_obj)


def show_traj(robot, traj, goal, target_obj):
    # first attach target obj to manipulator
    set_grasp(goal.manip, target_obj, inverse_transform(goal.grasp_tf), goal.grasp_config)
    # now plan
    robot.GetController().SetPath(traj)
    robot.WaitForController(0)
    # release object again
    robot.Release(target_obj)


def show_all_trajs(ghost, robot, mplanner, ghost_obj, target_obj, vel_scale=0.4):
    start_config = robot.GetDOFValues()
    for i in range(len(motion_planner.solutions) - 1):
        traj, goal = motion_planner.get_solution(i)
        ghost.SetDOFValues(start_config)
        show_ghost_traj(ghost, traj, goal, ghost_obj, vel_scale=vel_scale)

    ghost.SetVisible(False)
    ghost_obj.SetVisible(False)
    traj, goal = motion_planner.get_solution(len(motion_planner.solutions) - 1)
    show_traj(robot, traj, goal, target_obj)
    time.sleep(0.2)
    ghost.SetDOFValues(start_config)
    robot.SetDOFValues(start_config)
    ghost.SetVisible(True)
    ghost_obj.SetVisible(True)


def side_transition(viewer, goal_tf, cabinet, duration, fps=30):
    num_steps = duration * fps
    alpha_vals = np.linspace(1.0, 0.2, num_steps)
    alpha_t = 0
    for tf in get_tf_interpolation(viewer.GetCameraTransform(), goal_tf, num_steps):
        set_body_alpha(cabinet, alpha_vals[alpha_t])
        viewer.SetCamera(tf)
        time.sleep(1.0 / float(fps))
        alpha_t += 1


def plan(planner, body, it):
    now = time.time()
    traj, goal = planner.plan(it, body)
    print "Planning took %fs" % (time.time() - now)
    return traj, goal


def plan_for_stats(num_iterations, offset, robot_data, object_data, scene_sdf, regions, orientations,
                   objective_fn, global_region_info, problem_desc):
    manips = [robot_data.robot.GetActiveManipulator()]
    parameters = problem_desc['parameters']
    for idx in xrange(num_iterations):
        for r in regions:
            r.clear_subregions()
        # create afr hierarchy and bridge
        hierarchy = afr_placement_mod.AFRHierarchy(manips, regions, orientations, so2_depth=4, so2_branching=4)
        afr_bridge = afr_dmg_mod.MultiGraspAFRRobotBridge(afr_hierarchy=hierarchy, robot_data=robot_data,
                                                          object_data=object_data, objective_fn=obj_fn,
                                                          global_region_info=global_region_info, scene_sdf=scene_sdf,
                                                          parameters=parameters)
        # create stats recorder
        goal_stats = statsrecording.GoalSamplingStatsRecorder(afr_bridge, afr_bridge, afr_bridge)
        planner_stats = statsrecording.PlacementMotionStatsRecorder()
        if parameters["sampler_type"] == "simple_mcts_sampler":
            goal_sampler = simple_mcts_sampler_mod.SimpleMCTSPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
                                                                              [manip.GetName() for manip in manips],
                                                                              parameters=parameters,
                                                                              stats_recorder=goal_stats)
        elif parameters["sampler_type"] == "random":
            goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge, [
                                                                  manip.GetName() for manip in manips], True, False,
                                                                  stats_recorder=goal_stats)
        elif parameters["sampler_type"] == "random_no_opt":
            goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge, [
                                                                  manip.GetName() for manip in manips], True, False,
                                                                  stats_recorder=goal_stats, b_local_opt=False)
        elif parameters["sampler_type"] == "random_proj":
            goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
                                                                  [manip.GetName() for manip in manips], True, True,
                                                                  stats_recorder=goal_stats)
        elif parameters["sampler_type"] == "random_afr":
            goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
                                                                  [manip.GetName()
                                                                   for manip in manips], True, True, 0.5,
                                                                  stats_recorder=goal_stats)
        elif parameters["sampler_type"] == "conservative_random":
            goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
                                                                  [manip.GetName()
                                                                   for manip in manips], False, True, 0.1,
                                                                  stats_recorder=goal_stats)
        elif parameters["sampler_type"] == "optimistic_random":
            goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
                                                                  [manip.GetName()
                                                                   for manip in manips], False, True, 0.9,
                                                                  stats_recorder=goal_stats)
        if parameters["motion_planner"] == "fake":
            planner = anytime_planner_mod.DummyPlanner(goal_sampler, num_goal_samples=parameters["num_goal_samples"],
                                                       num_goal_iterations=parameters["num_goal_iterations"],
                                                       stats_recorder=planner_stats)
        else:
            planner = anytime_planner_mod.MGAnytimePlacementPlanner(goal_sampler, [manip],
                                                                    mplanner="ParallelMGBiRRT",
                                                                    num_goal_samples=parameters["num_goal_samples"],
                                                                    num_goal_iterations=parameters["num_goal_iterations"],
                                                                    mp_timeout=parameters["mp_timeout"],
                                                                    stats_recorder=planner_stats)
        goal_stats.reset()
        planner_stats.reset()
        rospy.loginfo("Running trial %i/%i using %s" % (idx, num_iterations, type(goal_sampler).__name__))
        _, _ = planner.plan(problem_desc["time_limit"], object_data.kinbody)
        goal_stats.save_stats(problem_desc["goal_stats_file"] + "_" + str(idx + offset) + ".csv")
        planner_stats.save_stats(problem_desc["plan_stats_file"] + "_" + str(idx + offset) + ".csv")
        # free resources
        del planner
        del goal_sampler
        del planner_stats
        del goal_stats
        del afr_bridge
        del hierarchy


if __name__ == "__main__":
    # NOTE If the OpenRAVE viewer is created too early, nothing works! Collision checks may be incorrect!
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_desc', help="Path to a yaml file specifying what world, robot to use etc.", type=str)
    parser.add_argument('--debug', help="If provided, run in debug mode", action="store_true")
    parser.add_argument('--num_runs', help="Number of executions", type=int, default=1)
    parser.add_argument('--offset', help="Offset for file names when running multiple runs", type=int, default=0)
    parser.add_argument('--show_gui', help="Flag whether to show GUI or not", action="store_true")
    parser.add_argument('--show_plcmnt_volume', help="If provided, visualize placement volume", action="store_true")
    parser.add_argument('--show_sdf_volume', help="If provided, visualize sdf volume", action="store_true")
    parser.add_argument('--show_sdf', help="If provided, visualize sdf", action="store_true")
    args = parser.parse_args()
    log_level = rospy.DEBUG if args.debug else rospy.INFO
    rospy.init_node("TestDMGPlacement", anonymous=True, log_level=log_level)
    with open(args.problem_desc, 'r') as f:
        problem_desc = yaml.load(f)
        resolve_paths(problem_desc, args.problem_desc)
        load_grasp(problem_desc)
    try:
        env = orpy.Environment()
        env.Load(problem_desc['or_env'])
        if args.debug:
            orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
        # load target object
        btarget_found = env.Load(problem_desc['target_obj_file'])
        if not btarget_found:
            raise ValueError("Could not load target object. Aborting")
        # target_obj_name = problem_desc['target_name']
        target_obj_name = "target_object"
        # ensure the object has a useful name
        env.GetBodies()[-1].SetName(target_obj_name)
        scene_occ = None
        try:
            scene_occ = grid_module.VoxelGrid.load(problem_desc['occ_file'])
        except IOError as e:
            rospy.logerr("Could not load %s. Please provide an occupancy grid of the scene." % problem_desc['occ_file'])
            rospy.logerr("There is a script to create one!")
            sys.exit(0)

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

        placement_volume = (np.array(problem_desc['plcmnt_volume'][:3]), np.array(problem_desc['plcmnt_volume'][3:]))
        occ_target_volume = scene_occ.get_subset(placement_volume[0], placement_volume[1])
        # extract placement orientations
        target_object = env.GetKinBody(target_obj_name)
        orientations = plcmnt_orientations_mod.compute_placement_orientations(target_object)
        # extract placement regions
        gpu_kit = plcmnt_regions_mod.PlanarRegionExtractor()
        surface_grid, labels, num_regions, regions = gpu_kit.extract_planar_regions(
            occ_target_volume, max_region_size=0.3)
        if num_regions == 0:
            print "No placement regions found"
            sys.exit(0)
        obj_radius = np.linalg.norm(target_object.ComputeAABB().extents())
        global_region_info = plcmnt_regions_mod.PlanarRegionExtractor.compute_surface_distance_field(
            surface_grid, 2.0 * obj_radius)
        # prepare robot data
        robot = env.GetRobot(problem_desc['robot_name'])
        # extract manipulators
        link_names = []
        manip_data = {}
        if 'manipulator' in problem_desc:
            robot.SetActiveManipulator(problem_desc['manipulator'])
        manip = robot.GetActiveManipulator()
        ik_solver = ik_module.IKSolver(manip, problem_desc['urdf_file'])
        # set initial grasp (needed for grasp set))
        # grasp_pose is oTe
        grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        set_grasp(manip, target_object, inverse_transform(grasp_pose), problem_desc['grasp_config'])
        gripper_info = afr_dmg_mod.load_gripper_info(problem_desc['gripper_information'], manip.GetName())
        grasp_set = afr_dmg_mod.DMGGraspSet(manip, target_object,
                                            problem_desc['target_obj_file'],
                                            gripper_info,
                                            problem_desc['dmg_file'])
        manip_data[manip.GetName()] = afr_dmg_mod.MultiGraspAFRRobotBridge.ManipulatorData(
            manip, ik_solver, grasp_set, gripper_info['gripper_file'])
        manip_links = [link.GetName() for link in get_manipulator_links(manip)]
        # remove base link - it does not move so
        manip_links.remove(manip.GetBase().GetName())
        link_names.extend(manip_links)
        robot_ball_approx = robot_sdf_module.RobotBallApproximation(robot, problem_desc['robot_ball_desc'])
        # build robot_octree
        try:
            now = time.time()
            robot_occgrid = robot_sdf_module.RobotOccupancyGrid.load(base_file_name=problem_desc['robot_occgrid'],
                                                                     robot=robot, link_names=link_names)
            rospy.logdebug("Loading robot occgrid took %fs" % (time.time() - now))
        except IOError:
            now = time.time()
            robot_occgrid = robot_sdf_module.RobotOccupancyGrid(problem_desc['parameters']['occ_tree_cell_size'],
                                                                robot=robot, link_names=link_names)
            robot_occgrid.save(problem_desc['robot_occgrid'])
            rospy.logdebug("Creating robot occgrid took %fs" % (time.time() - now))
        urdf_content = None
        with open(problem_desc['urdf_file'], 'r') as urdf_file:
            urdf_content = urdf_file.read()
        robot_data = afr_placement_mod.AFRRobotBridge.RobotData(
            robot, robot_occgrid, manip_data, urdf_content, robot_ball_approx)
        # create object data
        obj_occgrid = kinbody_sdf_module.RigidBodyOccupancyGrid(problem_desc['parameters']['occ_tree_cell_size'],
                                                                target_object.GetLinks()[0])
        obj_occgrid.setup_cuda_sdf_access(scene_sdf)
        object_data = afr_placement_mod.AFRRobotBridge.ObjectData(target_object, obj_occgrid)
        # create objective function
        now = time.time()
        if 'objective_fn' in problem_desc:
            if problem_desc['objective_fn'] == 'minimize_clearance':
                obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid,
                                                          b_max=False)
            elif problem_desc['objective_fn'] == 'maximize_clearance':
                obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid,
                                                          b_max=True)
            elif problem_desc['objective_fn'] == 'deep_shelf':
                obj_fn = objectives_mod.DeepShelfObjective(target_object, occ_target_volume, obj_occgrid, b_max=False)
        else:
            obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid,
                                                      b_max=False)
        rospy.logdebug("Creation of objective function took %fs" % (time.time() - now))
        # print "Check the placement regions!"
        # IPython.embed()
        # handles = []
        if args.debug:
            mcts_visualizer = mcts_visualizer_mod.MCTSVisualizer(robot, target_object)
        else:
            mcts_visualizer = None
        if args.num_runs == 1:
            parameters = problem_desc["parameters"]
            # create afr hierarchy
            hierarchy = afr_placement_mod.AFRHierarchy([manip], regions, orientations, so2_depth=4, so2_branching=4)
            afr_bridge = afr_dmg_mod.MultiGraspAFRRobotBridge(afr_hierarchy=hierarchy, robot_data=robot_data,
                                                              object_data=object_data, objective_fn=obj_fn,
                                                              global_region_info=global_region_info,
                                                              scene_sdf=scene_sdf,
                                                              parameters=parameters)
            # visualize placement regions
            if args.show_gui:
                env.SetViewer('qtcoin')  # WARNING: IK solvers also need to be created before setting the viewer
                handles = []
                # handles.append(draw_volume(env, placement_volume))
                # handles = visualize_occupancy_grid(env, surface_grid, color=np.array([1.0, 0.0, 0.0, 0.2]))
                handles.extend(plcmnt_regions_mod.visualize_plcmnt_regions(
                    env, regions, height=occ_target_volume.get_cell_size(), level=2))
                # make target object green
                set_body_color(target_object, np.array([0.0, 0.5, 0.0]))
                # create a ghost robot (for video)
                # robot_ghost = orpy.RaveCreateRobot(env, '')
                # robot_ghost.Clone(robot, 0)
                # robot_ghost.SetName("robot_ghost")
                # set_body_alpha(robot_ghost, 0.3)
                # robot_ghost.Enable(False)
                # env.AddRobot(robot_ghost)
                # create a ghost target obj
                # target_ghost = orpy.RaveCreateKinBody(env, '')
                # target_ghost.Clone(target_object, 0)
                # target_ghost.SetName("ghost_traget_object")
                # set_body_alpha(target_ghost, 0.4)
                # target_ghost.Enable(False)
                # env.AddKinBody(target_ghost)
                # cam_tf = np.array([[-0.9999906, -0.00261649, -0.00345688,  0.00483596],
                #                    [0.00313553,  0.11418095, -0.99345502,  1.61701882],
                #                    [0.00299408, -0.99345652, -0.11417167,  0.52455425],
                #                    [0.,  0.,  0.,  1.]])
                # cam_tf = np.array([[ 0.99836227, -0.00712139,  0.05676323,  0.01027336],
                #                    [-0.04739557, -0.6586312 ,  0.75097177, -0.25150079],
                #                    [ 0.03203807, -0.7524322 , -0.65789007,  1.15394711],
                #                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                # viewer = env.GetViewer()
                # viewer.SetCamera(cam_tf)
            # goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge, [
            #                                                       manip.GetName() for manip in manips], True, False)
            goal_sampler = simple_mcts_sampler_mod.SimpleMCTSPlacementSampler(hierarchy, afr_bridge, afr_bridge,
                                                                              afr_bridge, [manip.GetName()],
                                                                              debug_visualizer=mcts_visualizer,
                                                                              parameters=problem_desc["parameters"])

            planner_stats = statsrecording.PlacementMotionStatsRecorder()
            motion_planner = anytime_planner_mod.MGAnytimePlacementPlanner(goal_sampler, [manip],
                                                                           mplanner="ParallelMGBiRRT",
                                                                           num_goal_samples=parameters["num_goal_samples"],
                                                                           num_goal_iterations=parameters["num_goal_iterations"],
                                                                           mp_timeout=0.0,
                                                                           stats_recorder=planner_stats)
            dummy_planner = anytime_planner_mod.DummyPlanner(goal_sampler, num_goal_samples=parameters["num_goal_samples"],
                                                             num_goal_iterations=parameters["num_goal_iterations"])
            real_time = time.time()
            clock_time = time.clock()
            # traj, goal = plan(motion_planner, target_object, 120)
            # solutions, num_solutions = goal_sampler.sample(1, 10)
            # objectives, solutions = dummy_planner.plan(20, target_object)
            # print objectives
            # probe = env.GetKinBody("probe")
            # prober = kinbody_sdf_module.RigidBodyOccupancyGrid(0.005, probe.GetLinks()[0])
            # rospy.loginfo("Starting cProfile")
            # cProfile.run("goal_sampler.sample(100, 100)", '/tmp/cprofile_placement')
            # solutions, num_solutions = goal_sampler.sample(100, 10)
            # print "Took %f realtime, %f clocktime" % (time.time() - real_time, time.clock() - clock_time)
            # cProfile.run("plan(motion_planner, target_object, 5)", '/tmp/cprofile_placement')
            # rospy.loginfo("cProfile complete")
            IPython.embed()
        else:
            print "Running %i iterations of the planning algorithm" % args.num_runs
            plan_for_stats(args.num_runs, args.offset, robot_data, object_data, scene_sdf, regions, orientations,
                           obj_fn, global_region_info, problem_desc)
            print "Finished %i iterations of the planning algorithm" % args.num_runs
    finally:
        orpy.RaveDestroy()
