import os
import numpy
import logging
import openravepy as orpy
from orsampler import RobotCSpaceSampler
from sampler import FreeSpaceProximitySampler
from rrt import DynamicPGoalProvider, RRT
import hfts_grasp_planner.utils as utils
import hfts_grasp_planner.placement.placement_planning as plcmnt_module
import hfts_grasp_planner.sdf.core as sdf_module
from hierarchy_visualization import FreeSpaceProximitySamplerVisualizer


class IntegratedPlacementPlanner(object):
    """
        A placement planner for planning object placements and approach motions.
        ----------
        Parameters
        ----------
        sdf_resolution, float - Resolution for the signed distance field in m (smaller better, but slower)
        dof_weights, None or list of d floats - weights for the configuration space distance function for the 
            d degrees of freedom of the arm that is planned for
        draw_search_tree, bool - If true, draws motion planning tree in viewer
    """

    def __init__(self, env_file, sdf_file, sdf_volume,
                 data_path, robot_name, manip_name, **kwargs):
        """
            Creates a new integrated placement planner.
            ---------
            Arguments
            ---------
            env_file, string - path to the environment to plan in.
            sdf_file, string - path to a signed distance field of the environment.
                If there is no sdf stored at this location, it is computed and saved there.
            sdf_volume, tuple of two numpy arrays of shape (3,) - min and max point defining the
                maximum placement volume used to compute the sdf in. It is only used if sdf_file does not exist.
            data_path, string - path to data folder in which additional data such as placement
                preferences for different objects are stored.
            robot_name, string - name of the robot to plan for.
            manip_name, manipulator_name - name of the manipulator to use.

            Additionally, you may specify any parameter described in the class description.
        """
        self._parameters = {
            'sdf_resolution': 0.005,
            'dof_weights': None,
            'draw_search_tree': False,
        }
        self.set_parameters(**kwargs)
        self._env = orpy.Environment()
        self._env.Load(env_file)
        self._robot = self._env.GetRobot(robot_name)
        if not self._robot:
            raise ValueError("Could not retrieve robot with name %s" % robot_name)
        self._manip = self._robot.GetManipulator(manip_name)
        if not self._manip:
            raise ValueError("Could not retrieve manipulator with name %s" % manip_name)
        self._robot.SetActiveDOFs(self._manip.GetArmIndices())
        dynamic_bodies = [body for body in self._env.GetBodies() if utils.is_dynamic_body(body)]
        # TODO instead of excluding all dynamic bodies we could also simply activate and deactivate bodies as we need them
        self._scene_sdf = sdf_module.SceneSDF(self._env, [], excluded_bodies=dynamic_bodies)
        if os.path.exists(sdf_file):
            self._scene_sdf.load(sdf_file)
        else:
            self._scene_sdf.create_sdf(
                sdf_volume, self._parameters['sdf_resolution'], self._parameters['sdf_resolution'])
            self._scene_sdf.save(sdf_file)
        # TODO do we need to let the plctm planner know about the manipulator?
        self._plcmt_planner = plcmnt_module.PlacementGoalPlanner(
            data_path, self._env, self._scene_sdf, robot_name, manip_name)
        self._c_sampler = RobotCSpaceSampler(self._env, self._robot, scaling_factors=self._parameters['dof_weights'])
        # TODO pass additional parameters
        # initialize optional members as None
        self._debug_tree_drawer = None
        # now set them based parameters
        self._setup_optional_members()

    def set_parameters(self, **kwargs):
        """
            Set the given parameters.
            ---------
            Arguments
            ---------
            Any parameter described in the class description.
        """
        for (key, value) in kwargs.iteritems():
            self._parameters[key] = value

    def plan_placement(self, target_obj, plcmnt_volume, grasp_tf, start_config, time_limit=60.0):
        """
            Plan a placement for the given object in the given volume starting from the current state of
            the OpenRAVE environment. It is assumed that the object is being grasped.
            ---------
            Arguments
            ---------
            target_obj, string - name of the object to place
            plcmnt_volume, tuple of two numpy arrays each of shape (3,) - placement volume in form (min_point, max_point)
            grasp_tf, numpy array of shape (4, 4) - pose of the grasped object relative to the end-effector
            start_config, numpy array of shape (d + d_h,) - start configuration of the hand-arm system
            time_limit, float - maximal number of seconds to plan
            -------
            Returns
            -------
            path, None or list of SampleData - if succesful, the approach path to a placement pose, else None
            pose, None or numpy array (4, 4) - if succesful, the target placement pose
        """
        self._setup_optional_members()
        self._plcmt_planner.set_placement_volume(plcmnt_volume)
        # self._plcmt_planner.set_grasp(grasp_tf, grasp_config)
        # self._plcmt_planner.set_object(target_obj)
        target_body = self._env.GetKinBody(target_obj)
        if not self._robot.IsGrabbing(target_body):
            self._robot.Grab(target_body)
        self._plcmt_planner.setup(target_obj, grasp_tf, start_config[self._manip.GetGripperIndices()])
        if self._debug_tree_drawer:
            self._debug_tree_drawer.clear()
            debug_fn = self._debug_tree_drawer.draw_trees
        else:
            def debug_fn(forward_tree, backward_trees):
                pass
        goal_sampler = FreeSpaceProximitySampler(self._plcmt_planner, self._c_sampler)
        motion_planner = RRT(DynamicPGoalProvider(), self._c_sampler, goal_sampler, logging.getLogger())
        start_config = self._robot.GetActiveDOFValues()
        path = motion_planner.proximity_birrt(start_config, time_limit=time_limit, debug_function=debug_fn)
        return path, None  # TODO return placement pose, too

    def _setup_optional_members(self):
        """
            Initialize optional members.
        """
        if self._parameters['draw_search_tree'] and not self._debug_tree_drawer:
            self._debug_tree_drawer = utils.OpenRAVEDrawer(self._env, self._robot, True)
