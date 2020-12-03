#! /usr/bin/python
import IPython
import openravepy as orpy
import yaml
import os
import numpy as np
import argparse
import random
from hfts_grasp_planner.placement.afr_placement.multi_grasp import DMGGraspSet, load_gripper_info,\
     GripperCollisionChecker
from hfts_grasp_planner.placement.goal_sampler.interfaces import PlacementGoalSampler
import hfts_grasp_planner.utils as utils
import hfts_grasp_planner.ik_solver as ik_module

from generate_yumi_goals import resolve_paths, tf_matrix_to_yaml_dict, grasp_to_yaml_dict, goal_to_yaml_dict,\
     sample_waypoints, store_goals


class GraspSet(object):
    """ A simple GraspSet that provides the following functions, i.e. ducktyping parts of the
        afr_placement.multi_grasp.DMGGraspSet interface:
        get_num_grasps()
        get_grasp(int)
    """
    def __init__(self, dmg_grasp_sets=None, grasps=None):
        """Create a new grasp set either from a list of DMGGraspSets or a list of grasps.
        NOTE: You must provide at least one of the arguments dmg_grasp_sets or grasps.

        Args:
            dmg_grasp_sets (list(DMGGraspSet), optional): A list of individual DMGGraspSets that are to be merged
                into this single grasp set. Defaults to None.
            grasps (list(DMGGraspSet.Grasp), optional): A list of individual Grasps. Defaults to None.
        """
        self.grasps = []
        if dmg_grasp_sets is not None:
            for grasp_set in dmg_grasp_sets:
                offset = len(self.grasps)
                self.grasps.extend(
                    [GraspSet.copy_grasp(grasp_set.get_grasp(i), offset) for i in xrange(grasp_set.get_num_grasps())])
        if grasps is not None:
            self.grasps.extend([GraspSet.copy_grasp(g, offset) for g in grasps])
        if len(self.grasps) == 0:
            raise ValueError("No grasps given. You must either provide dmg_grasp_sets or grasps or both")
        self.grasp_id_to_index = {g.gid: i for i, g in enumerate(self.grasps)}

    def get_num_grasps(self):
        return len(self.grasps)

    def get_grasp(self, id_):
        if id_ in self.grasp_id_to_index:
            return self.grasps[self.grasp_id_to_index[id_]]
        return None

    def update_grasps(self, new_grasps):
        self.grasps = new_grasps
        self.grasp_id_to_index = {g.gid: i for i, g in enumerate(self.grasps)}

    @staticmethod
    def copy_grasp(g, id_offset):
        """Copy the given grasp and increase its id by the given offset.

        Args:
            g (DMGGraspSet.Grasp): The grasp to copy (shallow)
            id_offset (int): Offset to add to the grasp's id.

        Returns:
            DMGGraspSet.Grasp: A shallow copy of g with id g.gid + id_offset
        """
        return DMGGraspSet.Grasp(g.gid + id_offset, g.eTo, g.config, g.dmg_info, g.parent, g.distance)


def create_grasp_sets(problem_desc, manip, target_object):
    """Create grasp sets.

    Args:
        problem_desc (dict): Problem description that contains at least:
            grasp_file (str): path to a file containing different grasps
            dmg_file (str): path to a file containing the DMG of the target object
            kinbody_file (str): path to a file containing the kinbody description of the target object
            gripper_information (str): path to a file containing gripper information
        manip (OpenRAVE manipulator): The manipulator to compute the grasps for.
        target_object (OpenRAVE KinBody): The object to compute grasps for.
    Returns:
        GraspSet
    """
    with open(problem_desc['grasp_file'], 'r') as grasp_file:
        initial_grasps = yaml.load(grasp_file)
    gripper_info = load_gripper_info(problem_desc['gripper_information'], manip.GetName())
    dmg_sets = []
    with orpy.KinBodyStateSaver(target_object):
        with orpy.RobotStateSaver(manip.GetRobot()):
            for base_grasp in initial_grasps:
                grasp_pose = orpy.matrixFromQuat(base_grasp["grasp_pose"][3:])
                grasp_pose[:3, 3] = base_grasp["grasp_pose"][:3]
                grasp_config = np.array(base_grasp['grasp_config'])
                utils.set_grasp(manip, target_object, utils.inverse_transform(grasp_pose), grasp_config)
                dmg_sets.append(
                    DMGGraspSet(manip, target_object, problem_desc['target_obj_file'], gripper_info,
                                problem_desc['dmg_file']))
                robot.ReleaseAllGrabbed()
    return GraspSet(dmg_grasp_sets=dmg_sets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to generate a collection of grasp goals in a given scene.")
    parser.add_argument('problem_desc', help="Path to a yaml file specifying what world, robot to use etc.", type=str)
    parser.add_argument('num_goals', help="Number of goals to sample", type=int)
    parser.add_argument('output_path', help="Filename in which to store generated goals.", type=str)
    parser.add_argument('--sample_waypoints',
                        help="Sample n waypoint configurations and store them with the goals.",
                        type=int,
                        default=0)
    parser.add_argument('--show_viewer', help="Show viewer and launch IPython prompt", action='store_true')
    args = parser.parse_args()
    with open(args.problem_desc, 'r') as yamlfile:
        problem_desc = yaml.load(yamlfile)
        resolve_paths(problem_desc, args.problem_desc)
    # Load environment
    try:
        # initialize world
        env = orpy.Environment()
        env.Load(problem_desc['or_env'])
        robot = env.GetRobots()[0]
        if not env.Load(problem_desc['target_obj_file']):
            raise ValueError("Could not load target object from file '%s'. Aborting!" % problem_desc['target_obj_file'])
        # ensure the object has a useful name
        target_obj_name = "target_object"
        target_object = env.GetBodies()[-1]  # object that was loaded last
        target_object.SetName(target_obj_name)
        # set target object pose
        obj_tf = orpy.matrixFromQuat(problem_desc["object_pose"][3:])
        obj_tf[:3, 3] = problem_desc["object_pose"][:3]
        target_object.SetTransform(obj_tf)
        # ensure we have the correct active manipulator
        robot.SetActiveManipulator(problem_desc['manip_name'])
        manip = robot.GetActiveManipulator()
        # create ik solver
        ik_solver = ik_module.IKSolver(manip, problem_desc['urdf_file'])
        # create grasp sets
        grasp_set = create_grasp_sets(problem_desc, manip, target_object)
        # filter grasps based on collisions
        gripper_col_checker = GripperCollisionChecker(env, robot, problem_desc['gripper_file'], target_object)
        grasp_set.update_grasps([g for g in grasp_set.grasps if gripper_col_checker.is_gripper_col_free(g, obj_tf)])
        # now compute IK solutions, i.e. goals, for the given grasps
        if args.show_viewer:
            env.SetViewer('qtcoin')
            IPython.embed()
        goals = []
        print "Sampling goals"
        while len(goals) <= args.num_goals:
            grasp = random.choice(grasp_set.grasps)
            arm_config, col_free = ik_solver.compute_collision_free_ik(np.dot(obj_tf, grasp.oTe))
            if arm_config is not None and col_free:
                print "Found %i of %i goals" % (len(goals), args.num_goals)
                goal = PlacementGoalSampler.PlacementGoal(
                    manip,
                    arm_config,
                    obj_tf,
                    len(goals),
                    0.0,  # TODO cost of grasp somehow?
                    grasp.oTe,
                    grasp.config,
                    grasp.gid)
                goals.append(goal)
        # use store placements function from generate_yumi_goals to save the goals to disk
        # TODO sample waypoints
        if args.sample_waypoints:
            print "Sampling %i waypoints" % args.sample_waypoints
            # TODO use different volume? Maybe enlarged? Bounding box of goals?
            box_extents = problem_desc['approach_box_extents']
            approach_volume = (obj_tf[:3, 3] - box_extents, obj_tf[:3, 3] + box_extents)
            waypoints = sample_waypoints(manip,
                                         ik_solver,
                                         approach_volume, (np.array([0, 1, 0]), 1.0),
                                         args.sample_waypoints,
                                         pos_delta=0.02,
                                         num_rot_samples=10)
        store_goals(goals, grasp_set, waypoints, args.output_path, reverse_task=True)
        if args.show_viewer:
            IPython.embed()
    finally:
        orpy.RaveDestroy()
