import abc
import rospy
import itertools
import functools
import numpy as np
import scipy.spatial
import hfts_grasp_planner.ik_solver as ik_module
import hfts_grasp_planner.utils as utils
import hfts_grasp_planner.placement.so3hierarchy as so3hierarchy
import hfts_grasp_planner.placement.reachability as reachability
import hfts_grasp_planner.external.transformations as transformations


class PlacementPredicate(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def set_target_object(self, obj_name):
        """
            Set the target object
            ---------
            Arguments
            ---------
            obj_name, string - name of the object
        """
        pass

    @abc.abstractmethod
    def is_placement(self, plcmnt_result):
        """
            Return whether the given placement_result is a valid placement. 
            ---------
            Arguments
            ---------
            plcmnt_result, placement result to evaluate
        """
        pass


class SimplePlacementPredicate(object):
    """
        A simple placement predicate that is based on projecting the vertices
        of a placement face onto the environment. If the projected points form
        a plane that is sufficiently even and spans a stable placement polygon,
        the predicate is fulfilled.
        TODO implement
    """

    def __init__(self, obj_fn):
        """
            Create a new SimplePlacementPredicate.
        """
        self.objective_fn = obj_fn
        self.parameters = {
            'max_falling_distance': 0.05,
            'max_misalignment_angle': 0.2,
            'max_slope_angle': 0.2,
            'min_chull_distance': -0.008,
        }

    def is_placement(self, plcmnt_result):
        # TODO could/should cache this value
        # TODO this is specific to the simple placement heuristic
        # TODO this should do more checks, physics simulation or falling model
        is_goal = False
        if plcmnt_result._valid and plcmnt_result.is_leaf():
            _, falling_distance, chull_distance, alpha, gamma = self.objective_fn(plcmnt_result.obj_pose, True)
            is_goal = falling_distance < self.parameters['max_falling_distance'] and \
                chull_distance < self.parameters['min_chull_distance'] and \
                alpha < self.parameters['max_slope_angle'] and \
                gamma < self.parameters['max_misalignment_angle']
            rospy.logdebug('Candidate goal: falling_distance %f, chull_distance %f, alpha %f, gamma %f' %
                           (falling_distance, chull_distance, alpha, gamma))
        return is_goal


class PhysicsBasedPlacementPredicate(object):
    """
        A placement predicate that applies a rigid body physics simulator
        to decide whether a given pose is suitable for a placement or not.
    """

    def __init__(self):
        """ 
        TODO implement me!
        """
        pass


class RobotInterface(object):
    """
        Interface for the full robot used to compute arm configurations for a placement pose.
    """

    def __init__(self, env, robot_name, rmap_file, manip_name=None, urdf_file_name=None):
        """
            Create a new RobotInterface.
            ---------
            Arguments
            ---------
            env, OpenRAVE Environment - environment containing the full planning scene including the
                robot. The environment is copied.
            robot_name, string - name of the robot to compute ik solutions for
            rmap_file, string - filename for reachability map
            manip_name, string (optional) - name of manipulator to compute ik solutions for. If not provided,
                the active manipulator is used.
            urdf_file_name, string (optional) - filename of the urdf to use
        """
        # clone the environment so we are sure it is always setup correctly
        # TODO either clone env again, or change API so it isn't misleadingly stating that we clone it
        # self._env = env.CloneSelf(orpy.CloningOptions.Bodies)
        self._env = env
        self._robot = self._env.GetRobot(robot_name)
        if not self._robot:
            raise ValueError("Could not find robot with name %s" % robot_name)
        if manip_name:
            # self._robot.SetActiveManipulator(manip_name)
            active_manip = self._robot.GetActiveManipulator()
            assert(active_manip.GetName() == manip_name)
        self._manip = self._robot.GetActiveManipulator()
        self._ik_solver = ik_module.IKSolver(self._manip, urdf_file_name)
        self._hand_config = None
        self._grasp_tf = None  # from obj frame to eef-frame
        self._inv_grasp_tf = None  # from eef frame to object frame
        self._arm_dofs = self._manip.GetArmIndices()
        self._hand_dofs = self._manip.GetGripperIndices()
        self._rmap = reachability.ReachabilityMap(self._manip, self._ik_solver)
        self._rmap.load(rmap_file)

    def set_grasp_info(self, grasp_tf, hand_config, obj_name):
        """
            Set information about the grasp the target object is grasped with.
            ---------
            Arguments
            ---------
            grasp_tf, numpy array of shape (4,4) - pose of object relative to end-effector (in eef frame)
            hand_config, numpy array of shape (d_h,) - grasp configuration of the hand
            obj_name, string - name of grasped object
        """
        self._grasp_tf = grasp_tf
        self._inv_grasp_tf = utils.inverse_transform(grasp_tf)
        self._hand_config = hand_config
        # with self._env:  # TODO this is only needed if have a cloned environment
        #     # first ungrab all grabbed objects
        #     self._robot.ReleaseAllGrabbed()
        #     body = self._env.GetKinBody(obj_name)
        #     if not body:
        #         raise ValueError("Could not find object with name %s in or environment" % obj_name)
        #     # place the body relative to the end-effector
        #     eef_tf = self._manip.GetEndEffectorTransform()
        #     obj_tf = np.dot(eef_tf, grasp_tf)
        #     body.SetTransform(obj_tf)
        #     # set hand configuration and grab the body
        #     self._robot.SetDOFValues(hand_config, self._hand_dofs)
        #     self._robot.Grab(body)

    def check_arm_ik(self, obj_pose, seed=None):
        """
            Check whether there is an inverse kinematics solution for the arm to place the set
            object at the given pose.
            ---------
            Arguments
            ---------
            obj_pose, numpy array of shape (4, 4) - pose of the object in world frame
            seed, numpy array of shape (d,) (optional) - seed arm configuration to use for computation
            -------
            Returns
            -------
            config, None or numpy array of shape (d,) - computed arm configuration or None, if no solution \
                exists.
            b_col_free, bool - True if the configuration is collision free, else False
        """
        # with self._env:
        # compute eef-pose from obj_pose
        eef_pose = np.dot(obj_pose, self._inv_grasp_tf)
        # Now find an ik solution for the target pose with the hand in the pre-grasp configuration
        return self._ik_solver.compute_collision_free_ik(eef_pose, seed)

    def get_reachability_map(self):
        """
            Return a reachability map for this robot.
        """
        return self._rmap


class DefaultLeafStage(object):
    """
        Default leaf stage for the placement planner.
    """

    def __init__(self, env, objective_fn, collision_cost, robot_interface=None):
        self.objective_fn = objective_fn
        self.collision_cost = collision_cost
        self.robot_interface = robot_interface
        self.plcmnt_predicate = SimplePlacementPredicate(objective_fn)
        self.env = env
        self.target_object = None
        self._parameters = {
            'rhobeg': 0.01,
            'max_iter': 100,
        }

    def post_optimize(self, plcmt_result):
        """
            Locally optimize the objective function in the domain of the plcmt_result's node
            using scikit's constrained optimization by linear approximation function.
            ---------
            Arguments
            ---------
            plcmt_result, PlacementGoalPlanner.PlacementResult - result to update with a locally optimized
                solution.
        """
        def to_matrix(x):
            quat = so3hierarchy.hopf_to_quaternion(x[3:])
            pose = transformations.quaternion_matrix(quat)
            pose[:3, 3] = x[:3]
            return pose

        def pose_wrapper_fn(fn, x, multiplier=1.0):
            # extract pose from x and pass it to fn
            val = multiplier * fn(to_matrix(x))
            assert(val != float('inf'))
            return val

        rospy.logdebug("Post optimizing from node " + str(plcmt_result.get_hashable_label()))
        # get the initial value
        x0 = plcmt_result._hierarchy_node.get_representative_value(rtype=1)
        # rhobeg = 0.001
        # rhobeg = np.min(partition_range / 2.0)
        search_space_bounds = plcmt_result._hierarchy_node._hierarchy.get_bounds()
        constraints = [
            {
                'type': 'ineq',
                'fun': functools.partial(pose_wrapper_fn, self.collision_cost),
            },
            {
                'type': 'ineq',
                'fun': lambda x: x - search_space_bounds[:, 0]
            },
            {
                'type': 'ineq',
                'fun': lambda x: search_space_bounds[:, 1] - x
            }
        ]
        options = {
            'maxiter': self._parameters["max_iter"],
            'rhobeg': self._parameters["rhobeg"],
            'disp': True,
        }
        opt_result = scipy.optimize.minimize(functools.partial(pose_wrapper_fn, self.objective_fn, multiplier=-1.0),
                                             x0, method='COBYLA', constraints=constraints, options=options)
        sol = opt_result.x
        plcmt_result.obj_pose = to_matrix(sol)
        # self.evaluate_result(plcmt_result)

    def evaluate_result(self, plcmt_result):
        """
            Evaluate the given result and set its validity and goal flags.
            If a robot interface is set, this will also set an arm configuration for the result, if possible.
        """
        if self.robot_interface:
            plcmt_result.configuration, plcmt_result._valid = self.robot_interface.check_arm_ik(
                plcmt_result.obj_pose)
        else:
            with self.target_object:
                self.target_object.SetTransform(plcmt_result.obj_pose)
                plcmt_result._valid = not self.env.CheckCollision(self.target_object)
        # compute whether it is a goal
        plcmt_result._bgoal = self.plcmnt_predicate.is_placement(plcmt_result)
        plcmt_result._was_evaluated = True

    def set_object(self, obj_name):
        """
            Set the target object.
            ---------
            Arguments
            ---------
            obj_name, string - name of the target object
        """
        self.target_object = self.env.GetKinBody(obj_name)

    def set_grasp_info(self, grasp_tf, grasp_config):
        """
            Set information about the grasp the target object is grasped with.
            NOTE: You need to set an object before, otherwise this will fail.
            ---------
            Arguments
            ---------
            grasp_tf, numpy array of shape (4,4) - pose of object relative to end-effector (in eef frame)
            hand_config, numpy array of shape (d_h,) - grasp configuration of the hand
            obj_name, string - name of grasped object
        """
        if self.robot_interface:
            if not self.target_object:
                raise ValueError("Setting grasp info before setting target object. Please set target object first")
            self.robot_interface.set_grasp_info(grasp_tf, grasp_config, self.target_object.GetName())
