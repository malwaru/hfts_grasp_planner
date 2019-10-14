import os
import yaml
import rospy
import time
import numpy as np
import openravepy as orpy
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

class PlacementPlanner(object):
    def __init__(self, problem_desc):
        self.env = orpy.Environment()
        self.env.Load(problem_desc['or_env'])
        # load target object
        btarget_found = self.env.Load(problem_desc['target_obj_file'])
        if not btarget_found:
            raise ValueError("Could not load target object. Aborting")
        # target_obj_name = problem_desc['target_name']
        target_obj_name = "target_object"
        # ensure the object has a useful name
        self.env.GetBodies()[-1].SetName(target_obj_name)
        self.scene_occ = None
        try:
            self.scene_occ = grid_module.VoxelGrid.load(problem_desc['occ_file'])
        except IOError as e:
            rospy.logerr("Could not load %s. Please provide an occupancy grid of the scene." % problem_desc['occ_file'])
            rospy.logerr("There is a script to create one!")
            raise RuntimeError("Occgrid not found")

        dynamic_bodies = [body for body in self.env.GetBodies() if is_dynamic_body(body)]
        self.scene_sdf = sdf_module.SceneSDF(self.env, [], excluded_bodies=dynamic_bodies)
        if os.path.exists(problem_desc['sdf_file']):
            now = time.time()
            self.scene_sdf.load(problem_desc['sdf_file'])
            rospy.logdebug("Loading scene sdf took %fs" % (time.time() - now))
        else:
            rospy.logerr("Could not load %s. Please provide a signed distance field of the scene." % problem_desc['sdf_file'])
            rospy.logerr("There is a script to create one!")
            raise RuntimeError("Could not load scene sdf")

        placement_volume = (np.array(problem_desc['plcmnt_volume'][:3]), np.array(problem_desc['plcmnt_volume'][3:]))
        occ_target_volume = self.scene_occ.get_subset(placement_volume[0], placement_volume[1])
        # extract placement orientations
        self.target_object = self.env.GetKinBody(target_obj_name)
        orientations = plcmnt_orientations_mod.compute_placement_orientations(self.target_object)
        # extract placement regions
        gpu_kit = plcmnt_regions_mod.PlanarRegionExtractor()
        surface_grid, _, num_regions, regions = gpu_kit.extract_planar_regions(
            occ_target_volume, max_region_size=0.3)
        if num_regions == 0:
            raise RuntimeError("No placement regions founds")
        obj_radius = np.linalg.norm(self.target_object.ComputeAABB().extents())
        self.global_region_info = plcmnt_regions_mod.PlanarRegionExtractor.compute_surface_distance_field(surface_grid, 2.0 * obj_radius)
        # prepare robot data
        self.robot = self.env.GetRobot(problem_desc['robot_name'])
        # extract manipulators
        link_names = []
        self.manip_data = {}
        if 'manipulator' in problem_desc:
            self.robot.SetActiveManipulator(problem_desc['manipulator'])
        self.manip = self.robot.GetActiveManipulator()
        self.ik_solver = ik_module.IKSolver(self.manip, problem_desc['urdf_file'])
        # set initial grasp (needed for grasp set))
        # grasp_pose is oTe
        grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        set_grasp(self.manip, self.target_object, inverse_transform(grasp_pose), problem_desc['grasp_config'])
        self.gripper_info = afr_dmg_mod.load_gripper_info(problem_desc['gripper_information'], self.manip.GetName())
        self.grasp_set = afr_dmg_mod.DMGGraspSet(self.manip, self.target_object,
                                                 problem_desc['target_obj_file'],
                                                 self.gripper_info,
                                                 problem_desc['dmg_file'])
        self.manip_data[self.manip.GetName()] = afr_dmg_mod.MultiGraspAFRRobotBridge.ManipulatorData(
            self.manip, self.ik_solver, self.grasp_set, self.gripper_info['gripper_file'])
        manip_links = [link.GetName() for link in get_manipulator_links(self.manip)]
        # remove base link - it does not move so
        manip_links.remove(self.manip.GetBase().GetName())
        link_names.extend(manip_links)
        robot_ball_approx = robot_sdf_module.RobotBallApproximation(self.robot, problem_desc['robot_ball_desc'])
        # build robot_octree
        try:
            now = time.time()
            robot_occgrid = robot_sdf_module.RobotOccupancyGrid.load(base_file_name=problem_desc['robot_occgrid'],
                                                                     robot=self.robot, link_names=link_names)
            rospy.logdebug("Loading robot occgrid took %fs" % (time.time() - now))
        except IOError:
            now = time.time()
            robot_occgrid = robot_sdf_module.RobotOccupancyGrid(problem_desc['parameters']['occ_tree_cell_size'],
                                                                robot=self.robot, link_names=link_names)
            robot_occgrid.save(problem_desc['robot_occgrid'])
            rospy.logdebug("Creating robot occgrid took %fs" % (time.time() - now))
        urdf_content = None
        with open(problem_desc['urdf_file'], 'r') as urdf_file:
            urdf_content = urdf_file.read()
        self.robot_data = afr_placement_mod.AFRRobotBridge.RobotData(
            self.robot, robot_occgrid, self.manip_data, urdf_content, robot_ball_approx)
        # create object data
        obj_occgrid = kinbody_sdf_module.RigidBodyOccupancyGrid(problem_desc['parameters']['occ_tree_cell_size'],
                                                                self.target_object.GetLinks()[0])
        obj_occgrid.setup_cuda_sdf_access(self.scene_sdf)
        self.object_data = afr_placement_mod.AFRRobotBridge.ObjectData(self.target_object, obj_occgrid)
        # create objective function
        now = time.time()
        if 'objective_fn' in problem_desc:
            if problem_desc['objective_fn'] == 'minimize_clearance':
                self.obj_fn = clearance_mod.PackingObjective(occ_target_volume, obj_occgrid)
            elif problem_desc['objective_fn'] == 'maximize_clearance':
                self.obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid)
            elif problem_desc['objective_fn'] == 'deep_shelf':
                self.obj_fn = objectives_mod.DeepShelfObjective(self.target_object, occ_target_volume, obj_occgrid)
        else:
            self.obj_fn = clearance_mod.ClearanceObjective(occ_target_volume, obj_occgrid)
        rospy.logdebug("Creation of objective function took %fs" % (time.time() - now))
        self.parameters = problem_desc["parameters"]
        # create afr hierarchy
        self.hierarchy = afr_placement_mod.AFRHierarchy([self.manip], regions, orientations, so2_depth=4, so2_branching=4)
        self.problem_desc = problem_desc

        # self.env.SetViewer('qtcoin')
        # handles = []
        # handles.extend(plcmnt_regions_mod.visualize_plcmnt_regions(
        #     self.env, regions, height=occ_target_volume.get_cell_size(), level=2))
        # import IPython
        # IPython.embed()

    def plan(self, timelimit, start_config):
        self.robot.SetDOFValues(start_config)
        # self.grasp_set = afr_dmg_mod.DMGGraspSet(self.manip, self.target_object,
        #                                          self.problem_desc['target_obj_file'],
        #                                          self.gripper_info,
        #                                          self.problem_desc['dmg_file'])
        # self.manip_data[self.manip.GetName()].grasp_set = self.grasp_set
        init_grasp = self.grasp_set.get_grasp(0)
        set_grasp(self.manip, self.target_object, init_grasp.eTo, init_grasp.config)
        self.afr_bridge = afr_dmg_mod.MultiGraspAFRRobotBridge(afr_hierarchy=self.hierarchy, robot_data=self.robot_data,
                                                               object_data=self.object_data, objective_fn=self.obj_fn,
                                                               global_region_info=self.global_region_info,
                                                               scene_sdf=self.scene_sdf,
                                                               parameters=self.parameters)
        self.goal_sampler = simple_mcts_sampler_mod.SimpleMCTSPlacementSampler(self.hierarchy, self.afr_bridge, self.afr_bridge,
                                                                               self.afr_bridge, [self.manip.GetName()],
                                                                               debug_visualizer=None,
                                                                               parameters=self.problem_desc["parameters"])
        self.motion_planner = anytime_planner_mod.MGAnytimePlacementPlanner(self.goal_sampler, [self.manip],
                                                                            mplanner="ParallelMGBiRRT",
                                                                            num_goal_samples=self.parameters["num_goal_samples"],
                                                                            num_goal_iterations=self.parameters["num_goal_iterations"],
                                                                            mp_timeout=0.0)
        traj, goal = self.motion_planner.plan(timelimit, self.target_object)
        return traj, goal
        
