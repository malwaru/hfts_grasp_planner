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
                                      set_grasp, set_body_color, set_body_alpha, get_tf_interpolation, get_tf_gripper)
import hfts_grasp_planner.sdf.grid as grid_module
import hfts_grasp_planner.ik_solver as ik_module
import hfts_grasp_planner.sdf.core as sdf_module
import hfts_grasp_planner.sdf.robot as robot_sdf_module
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf_module
import hfts_grasp_planner.placement.afr_placement.placement_orientations as plcmnt_orientations_mod
import hfts_grasp_planner.placement.afr_placement.placement_regions as plcmnt_regions_mod
import hfts_grasp_planner.placement.afr_placement.core_dmg as afr_placement_mod
import hfts_grasp_planner.placement.afr_placement.statsrecording as statsrecording
import hfts_grasp_planner.placement.goal_sampler.random_sampler as rnd_sampler_mod
import hfts_grasp_planner.placement.goal_sampler.mcts_sampler as mcts_sampler_mod
import hfts_grasp_planner.placement.goal_sampler.simple_mcts_sampler as simple_mcts_sampler_mod
import hfts_grasp_planner.placement.goal_sampler.mcts_visualization as mcts_visualizer_mod
import hfts_grasp_planner.placement.anytime_planner as anytime_planner_mod
# import hfts_grasp_planner.placement.reachability as rmap_mod
import hfts_grasp_planner.placement.clearance as clearance_mod
from hfts_grasp_planner.sdf.visualization import visualize_occupancy_grid
from hfts_grasp_planner.dmg.dmg_class import DexterousManipulationGraph as DMG
# from transformations import compose_matrix


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
    for key in ['or_env', 'occ_file', 'sdf_file', 'urdf_file', 'target_obj_file', 'gripper_file', 'grasp_file',
                'robot_occtree', 'robot_occgrid', 'reachability_path', 'robot_ball_desc',
                'target_object_stl', 'graph_file', 'nodes_position_file', 'nodes_component_file',
                'component_normal_file', 'nodes_angle_file', 'supervoxel_component_file']:
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
        # problem_desc['grasp_pose'] = grasp_yaml[grasp_id]['grasp_pose']

        # grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        # grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        # grasp_pose = utils.inverse_transform(grasp_pose)
        # problem_desc['grasp_pose'][:3] = grasp_pose[:3, 3]
        # problem_desc['grasp_pose'][3:] = orpy.quatFromRotationMatrix(grasp_pose)
        problem_desc['grasp_config'] = grasp_yaml[grasp_id]['grasp_config']
        problem_desc['dmg_node'] = grasp_yaml[grasp_id]['dmg_node']
        problem_desc['dmg_angle'] = grasp_yaml[grasp_id]['dmg_angle']
        problem_desc['angle_weight'] = grasp_yaml[grasp_id]['angle_weight']


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


def side_transition(goal_tf, cabinet, duration, fps=30):
    num_steps = duration * fps
    alpha_vals = np.linspace(1.0, 0.2, num_steps)
    alpha_t = 0
    for tf in get_tf_interpolation(viewer.GetCameraTransform(), goal_tf, num_steps):
        set_body_alpha(cabinet, alpha_vals[alpha_t])
        viewer.SetCamera(tf)
        time.sleep(1.0 / float(fps))
        alpha_t += 1


def plan(planner, body, it):
    # with body:
    now = time.time()
    traj, goal = planner.plan(it, body)
    print "Planning took %fs" % (time.time() - now)
    return traj, goal

def get_grasp(dmg_node, dmg_angle, robot, dmg):
    grasp_tf = get_tf_gripper(gripper=robot.GetJoint('gripper_r_joint'))
    object_tf = dmg.make_transform_matrix(dmg_node, dmg_angle)
    if object_tf is None:
        raise ValueError("The provided DMG node has no valid opposite node. Aborting")
    return inverse_transform(np.dot(grasp_tf, inverse_transform(object_tf)))
    # return np.dot(grasp_tf, inverse_transform(object_tf))


def plan_for_stats(num_iterations, offset, robot_data, object_data, scene_sdf, regions, orientations, 
                   objective_fn, global_region_info, problem_desc):
    manips = robot_data.robot.GetManipulators()
    parameters = problem_desc['parameters']
    for idx in xrange(num_iterations):
        for r in regions:
            r.clear_subregions()
        # create afr hierarchy and bridge
        hierarchy = afr_placement_mod.AFRHierarchy(manips, regions, orientations, so2_depth=4, so2_branching=4)
        afr_bridge = afr_placement_mod.AFRRobotBridge(afr_hierarchy=hierarchy, robot_data=robot_data,
                                                      object_data=object_data, objective_fn=obj_fn,
                                                      global_region_info=global_region_info, scene_sdf=scene_sdf,
                                                      parameters=parameters)
        # create stats recorder
        goal_stats = statsrecording.GoalSamplingStatsRecorder(afr_bridge, afr_bridge, afr_bridge)
        planner_stats = statsrecording.PlacementMotionStatsRecorder()
        if parameters["sampler_type"] == "mcts_sampler":
            goal_sampler = mcts_sampler_mod.MCTSPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge, 
                                                                 [manip.GetName() for manip in manips],
                                                                 parameters=parameters,
                                                                 stats_recorder=goal_stats)
        elif parameters["sampler_type"] == "simple_mcts_sampler":
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
                                                                  [manip.GetName() for manip in manips], True, True, 0.5,
                                                                  stats_recorder=goal_stats)
        elif parameters["sampler_type"] == "conservative_random":
            goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
                                                                  [manip.GetName() for manip in manips], False, True, 0.1,
                                                                  stats_recorder=goal_stats)
        elif parameters["sampler_type"] == "optimistic_random":
            goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
                                                                  [manip.GetName() for manip in manips], False, True, 0.9,
                                                                  stats_recorder=goal_stats)
        if parameters["motion_planner"] == "fake":
            planner = anytime_planner_mod.DummyPlanner(goal_sampler, num_goal_samples=parameters["num_goal_samples"],
                                                       num_goal_iterations=parameters["num_goal_iterations"],
                                                       stats_recorder=planner_stats)
        else:
            planner = anytime_planner_mod.AnyTimePlacementPlanner(goal_sampler, manips,
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
    rospy.init_node("TestPlacement2", anonymous=True, log_level=log_level)
    with open(args.problem_desc, 'r') as f:
        problem_desc = yaml.load(f)
        resolve_paths(problem_desc, args.problem_desc)
        load_grasp(problem_desc)

    try:

        # Load DMG files
        dmg = DMG()
        dmg.set_object_shape_file(problem_desc['target_object_stl'])
        dmg.read_graph(problem_desc['graph_file'])
        dmg.read_nodes(problem_desc['nodes_position_file'])
        dmg.read_node_to_component(problem_desc['nodes_component_file'])
        dmg.read_component_to_normal(problem_desc['component_normal_file'])
        dmg.read_node_to_angles(problem_desc['nodes_angle_file'])
        dmg.read_supervoxel_angle_to_angular_component(problem_desc['supervoxel_component_file'])

        env = orpy.Environment()
        env.Load(problem_desc['or_env'])
        # load floating gripper
        env.Add(env.ReadRobotURI(problem_desc['gripper_file']),True)
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

        # placement_planner = pp_module.PlacementGoalPlanner(problem_desc['data_path'], env, scene_sdf)
        # placement_volume = (np.array([-0.35, 0.55, 0.6]), np.array([0.23, 0.8, 0.77]))  # on top of shelf
        # placement_volume = (np.array([-0.35, 0.55, 0.42]), np.array([0.53, 0.9, 0.77]))  # all of the shelf
        # placement_volume = (np.array([-0.35, 0.55, 0.42]), np.array([0.53, 0.9, 0.64]))  # inside shelf
        # placement_volume = (np.array([-0.35, 0.4, 0.42]), np.array([0.53, 0.55, 0.64]))  # front of shelf
        # placement_volume = (np.array([-0.35, 0.55, 0.44]), np.array([0.53, 0.9, 0.49]))  # bottom inside shelf
        # placement_volume = (np.array([0.24, 0.58, 0.51]), np.array([0.29, 0.8, 0.55]))  # volume in small gap
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
        global_region_info = plcmnt_regions_mod.PlanarRegionExtractor.compute_surface_distance_field(surface_grid, 2.0 * obj_radius)
        # prepare robot data
        robot = env.GetRobot(problem_desc['robot_name'])
        # extract manipulators
        link_names = []
        manip_data = {}
        manips = robot.GetManipulators()

        # Just use one manipulator
        # To use both, comment the pop() statement
        manips.pop()

        # prepare floating gripper
        robot_gripper = env.GetRobot(problem_desc['gripper_name'])
        robot_gripper.SetJointValues([problem_desc['gripper_joint_value']], [0])

        # Get DMG Data
        initial_dmg_node = problem_desc["dmg_node"]
        initial_dmg_angle = problem_desc["dmg_angle"]
        
        for manip in manips:
            ik_solver = ik_module.IKSolver(manip, problem_desc['urdf_file'])
            
            # TODO have different grasp poses for each manipulator
            # grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
            # grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]

            # DMG node to grasp_pose
            dmg.set_current_node(initial_dmg_node)
            dmg.set_current_angle(initial_dmg_angle)
            grasp_pose = get_grasp(initial_dmg_node, initial_dmg_angle, robot, dmg)
            

            # grasp_pose = inverse_transform(grasp_pose)
            # rmap = rmap_mod.SimpleReachabilityMap(manip, ik_solver)
            # try:
            #     filename = problem_desc["reachability_path"] + '/' + robot.GetName() + '_' + manip.GetName() + '.npy'
            #     rmap.load(filename)
            # except IOError:
            #     rospy.logerr("Could not load reachability map for %s from file %s. Please provide one!" % (manip.GetName(), filename))
            #     sys.exit(1)
            manip_data[manip.GetName()] = afr_placement_mod.AFRRobotBridge.ManipulatorData(
                manip, ik_solver, None, grasp_pose, problem_desc['grasp_config'], robot_gripper)
            manip_links = [link.GetName() for link in get_manipulator_links(manip)]
            # remove base link - it does not move so
            manip_links.remove(manip.GetBase().GetName())
            link_names.extend(manip_links)
        robot_ball_approx = robot_sdf_module.RobotBallApproximation(robot, problem_desc['robot_ball_desc'])
        # build robot_octree
        try:
            now = time.time()
            # robot_octree = robot_sdf_module.RobotOccupancyOctree.load(base_file_name=problem_desc['robot_occtree'],
            #                                                           robot=robot, link_names=link_names)
            # rospy.logdebug("Loading robot octree took %fs" % (time.time() - now))
            robot_occgrid = robot_sdf_module.RobotOccupancyGrid.load(base_file_name=problem_desc['robot_occgrid'],
                                                                     robot=robot, link_names=link_names)
            rospy.logdebug("Loading robot occgrid took %fs" % (time.time() - now))
        except IOError:
            # robot_octree = robot_sdf_module.RobotOccupancyOctree(
            #     problem_desc['parameters']['occ_tree_cell_size'], robot, link_names)
            # robot_octree.save(problem_desc['robot_occtree'])
            now = time.time()
            robot_occgrid = robot_sdf_module.RobotOccupancyGrid(problem_desc['parameters']['occ_tree_cell_size'],
                                                                robot=robot, link_names=link_names)
            robot_occgrid.save(problem_desc['robot_occgrid'])
            rospy.logdebug("Creating robot occgrid took %fs" % (time.time() - now))
        urdf_content = None
        with open(problem_desc['urdf_file'], 'r') as urdf_file:
            urdf_content = urdf_file.read()
        robot_data = afr_placement_mod.AFRRobotBridge.RobotData(robot, robot_occgrid, manip_data, urdf_content, robot_ball_approx)
        # create object data
        # obj_octree = kinbody_sdf_module.OccupancyOctree(
        #     problem_desc['parameters']['occ_tree_cell_size'], target_object.GetLinks()[0])
        obj_occgrid = kinbody_sdf_module.RigidBodyOccupancyGrid(problem_desc['parameters']['occ_tree_cell_size'],
                                                                target_object.GetLinks()[0])
        obj_occgrid.setup_cuda_sdf_access(scene_sdf)

        #Get DMG Grasp Order
        # grasp_order = dmg.create_grasp_order(initial_dmg_node, initial_dmg_angle, max_depth=5)
        grasp_order = dmg.run_dijsktra(initial_dmg_node, initial_dmg_angle, angle_weight=problem_desc['angle_weight'])

        object_data = afr_placement_mod.AFRRobotBridge.ObjectData(target_object, obj_occgrid, dmg, grasp_order)
        # create objective function
        now = time.time()
        obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid,
                                                  b_max=problem_desc['maximize_clearance'])
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
            hierarchy = afr_placement_mod.AFRHierarchy(manips, regions, orientations, so2_depth=4, so2_branching=4)
            afr_bridge = afr_placement_mod.AFRRobotBridge(afr_hierarchy=hierarchy, robot_data=robot_data,
                                                          object_data=object_data, objective_fn=obj_fn,
                                                          global_region_info=global_region_info, scene_sdf=scene_sdf,
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
                cam_tf = np.array([[-0.9999906 , -0.00261649, -0.00345688,  0.00483596],
                                   [ 0.00313553,  0.11418095, -0.99345502,  1.61701882],
                                   [ 0.00299408, -0.99345652, -0.11417167,  0.52455425],
                                   [ 0.        ,  0.        ,  0.        ,  1.        ]])
                # cam_tf = np.array([[ 0.99836227, -0.00712139,  0.05676323,  0.01027336],
                #                    [-0.04739557, -0.6586312 ,  0.75097177, -0.25150079],
                #                    [ 0.03203807, -0.7524322 , -0.65789007,  1.15394711],
                #                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
                viewer = env.GetViewer()
                viewer.SetCamera(cam_tf)
                
            # goal_sampler = rnd_sampler_mod.RandomPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge, [
            #                                                       manip.GetName() for manip in manips], True, False)
            # goal_sampler = mcts_sampler_mod.MCTSPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
            #                                                      [manip.GetName() for manip in manips],
            #                                                      debug_visualizer=mcts_visualizer,
            #                                                      parameters=problem_desc["parameters"])
            goal_sampler = simple_mcts_sampler_mod.SimpleMCTSPlacementSampler(hierarchy, afr_bridge, afr_bridge, afr_bridge,
                                                                              [manip.GetName() for manip in manips],
                                                                              debug_visualizer=mcts_visualizer,
                                                                              parameters=problem_desc["parameters"])

            planner_stats = statsrecording.PlacementMotionStatsRecorder()
            motion_planner = anytime_planner_mod.AnyTimePlacementPlanner(goal_sampler, manips,
                                                                         num_goal_samples=parameters["num_goal_samples"],
                                                                         num_goal_iterations=parameters["num_goal_iterations"],
                                                                         mp_timeout=parameters["mp_timeout"],
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

            #Rizwan
            # from copy import deepcopy
            # import math
            # grasp_pose2 = deepcopy(grasp_pose)

            # Node (52,0)
            # T = np.array([ 0.015657 , -0.0210361, -0.013972 ])
            # T2 = np.array([ 0.0187194 , -0.0280397 ,  0.00791634])
            # zero_axis = np.array([0.999843 , 0.0, 0.0177472])
            # normal = np.array([-0.0176726, -0.0916115,  0.995638 ])
            # Tm = [ (T[0]+T2[0])/2.0, (T[1]+T2[1])/2.0, (T[2]+T2[2])/2.0 ]
            # R2 = 180.0 # Note: Original angle 0 

            # # Node (82,0)
            # # T = np.array([-0.0187902,  0.0390576, -0.0112461])
            # # T2 = np.array([-0.0205926,  0.0310959,  0.0121499])
            # # zero_axis = np.array([0.999843 , 0.0, 0.0177472])
            # # normal = np.array([-0.0176726, -0.0916115,  0.995638 ])
            # # Tm = [ (T[0]+T2[0])/2.0, (T[1]+T2[1])/2.0, (T[2]+T2[2])/2.0 ]
            # # R2 = 100.0-180.0 # Note: Original angle 100

            # #rotate the zero angle axis by the angle
            # theta = R2*math.pi/180.0
            # Rot = np.array([ [1,0,0], [0, math.cos(theta), -math.sin(theta) ], [0, math.sin(theta), math.cos(theta)] ])
            # #get the axis-angle matrix 
            # rrt = np.outer(normal, normal)
            # Sr = np.array([[0, -normal[2], normal[1]], [normal[2], 0, -normal[0]], [-normal[1], normal[0], 0]])
            # R = rrt + (np.identity(3) - rrt)*math.cos(theta) + Sr*math.sin(theta)
            # iR = np.array([ [1,0,0], [0,0,-1], [0,1,0] ] )
            # # R = np.dot(iR,R)
            
            # finger_axis = 1.0*np.dot(R, zero_axis)
            # finger_axis = finger_axis/np.linalg.norm(finger_axis)
            # M = np.array([normal,  zero_axis, np.cross(normal,zero_axis)])
            # M = np.dot(M, Rot)
            # R = np.dot(iR , M)

            # # A = compose_matrix(translate=[0,0,0], angles=R)
            # A = deepcopy(grasp_pose)
            # A[0][:3] = R[0]
            # A[1][:3] = R[1]
            # A[2][:3] = R[2]
            # # A[0][:3] = [1.0,0.0,0.0]
            # # A[1][:3] = [0.0,1.0,0.0]
            # # A[2][:3] = [0.0,0.0,1.0]
            # A[0][3] = Tm[0]
            # A[1][3] = Tm[1]
            # A[2][3] = Tm[2]
            # # A[0][3] = 0.0
            # # A[1][3] = 0.0
            # # A[2][3] = 0.0

            # gripper_r = robot.GetJoint('gripper_r_joint')
            # eTcl = gripper_r.GetInternalHierarchyLeftTransform()
            # eTcr = gripper_r.GetInternalHierarchyRightTransform()
            
            # # Center of finger transforms
            # eTcX = (eTcl[0][3] + eTcr[0][3])/2.0
            # eTcY = (eTcl[1][3] + eTcr[1][3])/2.0
            # eTcZ = (eTcl[2][3] + eTcr[2][3])/2.0

            # def angle_test(X):            
            #     # Finger offsets
            #     eTc = deepcopy(eTcr)
            #     eTc[0][3] = X[0]
            #     eTc[1][3] = X[1]
            #     eTc[2][3] = X[2]
            #     # eTc[0][3] = 0.0
            #     # eTc[1][3] = eTcY + 0.04
            #     # eTc[2][3] = eTcZ + 0.04
            #     eTc[0][:3] = [1,0,0]
            #     eTc[1][:3] = [0,1,0]
            #     eTc[2][:3] = [0,0,1]

            #     grasp_pose2 = np.dot(eTc, inverse_transform(A) )
            #     target_object.SetTransform(np.dot(manip.GetEndEffectorTransform(), grasp_pose2 ))

            # approx1 = [eTcX + 0.0, eTcY + 0.0065, eTcZ + 0.0837]
            # approx2 = [eTcX, eTcY, eTcZ]
            # angle_test(approx1)
            # target_object.SetTransform(np.dot(manip.GetEndEffectorTransform(), inverse_transform(A) ))

            # grasp_pose2[0][:3] = A[0][:3]
            # grasp_pose2[1][:3] = A[1][:3]
            # grasp_pose2[2][:3] = A[2][:3]
            # grasp_pose2[0][3] = T[0]
            # grasp_pose2[1][3] = T[1]
            # grasp_pose2[2][3] = T[2]
            # grasp_pose2[0] = [0.4719952, -0.5889378, -0.6560280, -0.1227483]
            # grasp_pose2[1] = [0.0307596, -0.7326781,  0.6798798, 0.02869511]
            # grasp_pose2[2] = [-0.8810643, -0.3410792, -0.3277051, -0.01849286]
            # inv_grasp = inverse_transform(grasp_pose2)

            # grasp_tf = get_tf_gripper(gripper=robot.GetJoint('gripper_r_joint'))
            # object_tf = dmg.make_transform_matrix(initial_dmg_node, initial_dmg_angle)
            # grasp_pose2 = np.dot(grasp_tf, inverse_transform(object_tf))
            # target_object.SetTransform(np.dot(manip.GetEndEffectorTransform(), grasp_pose2 ))

            # # robot_gripper.SetTransform(manip.GetEndEffectorTransform())
            # grasp_tf_gripper = get_tf_gripper(gripper=robot_gripper.GetJoints()[0])
            # target_pose = target_object.GetTransform()
            # object_tf2 = np.dot(target_pose, object_tf)
            # grasp_pose3 = np.dot(object_tf2, inverse_transform(grasp_tf_gripper))
            # robot_gripper.SetTransform(grasp_pose3)


            IPython.embed()
        else:
            plan_for_stats(args.num_runs, args.offset, robot_data, object_data, scene_sdf, regions, orientations,
                           obj_fn, global_region_info, problem_desc) 
    finally:
        orpy.RaveDestroy()
