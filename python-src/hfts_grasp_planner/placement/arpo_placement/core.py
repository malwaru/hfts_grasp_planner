import rospy
import scipy
import itertools
import numpy as np
import openravepy as orpy
from functools import partial
import trac_ik_python.trac_ik as trac_ik_module
import hfts_grasp_planner.external.transformations as tf_mod
import hfts_grasp_planner.utils as utils
import hfts_grasp_planner.ik_solver as ik_module
import hfts_grasp_planner.placement.so2hierarchy as so2hierarchy
import hfts_grasp_planner.placement.so3hierarchy as so3hierarchy
import hfts_grasp_planner.placement.goal_sampler.interfaces as placement_interfaces
"""
    This module defines the placement planning interfaces for an
    Arm-Region-PlacementOrientation(arpo)-hierarchy.
    An arpo-hierarchy allows to search for an object placement with a dual arm robot.
    On the root level,the hierarchy has as many branches as the robot has arms, i.e.
    it represents the decision which arm to use to place the object.
    On the second level, the hierarchy has as many branches as there are placement regions
    in the target volume.
    On the third level, it has as many branches as there are placement orientations for the object.
    Subsequent hierarchy levels arise from a subdivison of the SE(2) state space of the object
    in a particular region for the chosen placement orientation.
"""


class ARPOHierarchy(placement_interfaces.PlacementHierarchy):
    """
        Defines an ARPO hierarchy.
        A node in this hierarchy is identified by a key, which is a tuple of ints followed by two more tuples:
        (a, r, o, subregion_key, so2_key). The integers a, r, o define the arm, region and placement orientation.
        The elements subregion_key and so2_key are themselves tuple of ints representing 1. a subregion of the region r
        and 2. an interval of SO2. A key may be partially defined from left to right. Valid key formats are:
        (,) - root
        (a,) - chosen arm a, nothing else
        (a, r) - chosen arm a, region r, nothing else
        (a, r, o, (), ()) - chosen arm a, region r, orientation o, nothing else
        (a, r, o, subregion_key, so2_key) - subregion_key and so2_key can also be partially defined in the same way.

        Hierarchy layout:
        1. level: choice of arm
        2. level: choice of region
        3. level: choice placement orientation
        4. - n. level: choice of subregion and SO2 interval

        Subregions and SO2 interval:
        On level 4 and below, nodes represent a cartesian product of a region in R^2 and SO(2).
        The R^2 region is described by a placement region.
        A placement region describes the position of a contact point on the chull of an object.
        Each placement region can be divided into smaller subregions. This can be done until some minimum
        resolution is reached. This resolution is defined in the class of the placement region, and this class only
        queries for subregions as long as these exist.
        Similarly, the orientation of the object around the z-axis going through the aforementioned contact point
        is discretized hierarchically. On each level k >= 4 a node is assigned a subinterval of SO2, i.e. [0, 2pi].
        The branching factor at which the SO2 intervals are subdivided between each layer is passed as an argument.
        Since the size of placement regions can vary, and thus the depth of their hierarchy, you need to additionally
        specify the depth of the SO2 hierarchy. This implies that no interval in SO2 will be smaller than
        2pi / so2_branching^so2_depth. If a node has reached the bottom of the SO2 hierarchy, but there are more
        placement subregions, all its children share the same SO2 interval.
        Once a node has no subregions and is at the bottom of the SO2 hierarchy, it has no children anymore
        and the end of this composed hierarchy is reached.
        The above procedure guarantees that all leaves of this hierarchy are equal in placement region size
        and SO2 interval length.
    """

    def __init__(self, manipulators, regions, orientations, so2_depth, so2_branching=4):
        """
            Create a new ARPO hierarchy.
            ---------
            Arguments
            ---------
            manipulators, list of OpenRAVE manipulators - each manipulator represents one arm
            regions, list of PlacementRegions (see module placement_regions)
            orientations, list of PlacementOrientation (see module placement_orientations)
            so2_depth, int - maximum depth of SO2 hierarchy
            so2_branching, int - branching factor for SO2 on hierarchy levels below level 3.
        """
        self._manips = manipulators
        self._regions = regions
        self._orientations = orientations
        self._so2_branching = so2_branching
        self._so2_depth = so2_depth

    def get_child_key_gen(self, key):
        """
            Return a key-generator for the children of the given key.
            ---------
            Arguments
            ---------
            key, tuple - see class documentation for key description
            -------
            Returns
            -------
            generator of tuple of int - the generator produces children of the node with key key
                If there are no children, None is returned
        """
        if len(key) == 0:
            return ((i,) for i in xrange(len(self._manips)))
        elif len(key) == 1:
            return (key + (i,) for i in xrange(len(self._regions)))
        elif len(key) == 2:
            return (key + (i,) for i in xrange(len(self._orientations)))
        else:
            if len(key) == 3:
                subregion_key = ()
                so2_key = ()
            else:
                assert(len(key) == 5)
                # extract sub region key
                subregion_key = key[3]
                so2_key = key[4]
            subregion = self.get_placement_region((key[1], subregion_key))
            b_region_leaf = not subregion.has_subregions()
            b_so2_leaf = so2hierarchy.is_leaf(so2_key, self._so2_depth)
            if b_region_leaf and b_so2_leaf:
                return None
            if b_region_leaf:
                return (key[:3] + (subregion_key + (0,), o)
                        for o in so2hierarchy.get_key_gen(so2_key, self._so2_branching))
            if b_so2_leaf:
                return (key[:3] + (subregion_key + (r,), so2_key + (0,))
                        for r in xrange(subregion.get_num_subregions()))
            return (key[:3] + (subregion_key + (r,), o) for r in xrange(subregion.get_num_subregions())
                    for o in so2hierarchy.get_key_gen(so2_key, self._so2_branching))

    def get_random_child_key(self, key):
        """
            Return a random key of a child of the given key.
            ---------
            Arguments
            ---------
            key, tuple - see class documentation for key description
            -------
            Returns
            -------
            child key, tuple - a randomly generated key that describes a child of the input key.
                If there is no child defined, None is returned
        """
        if len(key) == 0:
            return (np.random.randint(0, len(self._manips)),)
        if len(key) == 1:
            return key + (np.random.randint(0, len(self._regions)),)
        if len(key) == 2:
            return key + (np.random.randint(0, len(self._orientations)),)
        # extract sub region key
        if len(key) == 5:
            subregion_key = key[3]
            so2_key = key[4]
        else:
            assert(len(key) == 3)
            subregion_key = ()
            so2_key = ()
        subregion = self.get_placement_region((key[1], subregion_key))
        b_region_leaf = not subregion.has_subregions()
        b_so2_leaf = so2hierarchy.is_leaf(so2_key, self._so2_depth)
        if b_region_leaf and b_so2_leaf:
            return None
        if b_region_leaf:
            return key[:3] + (subregion_key + (0,), so2_key + (np.random.randint(self._so2_branching),))
        if b_so2_leaf:
            return key[:3] + (subregion_key + (np.random.randint(subregion.get_num_subregions()),), so2_key + (0,))
        return key[:3] + (subregion_key + (np.random.randint(subregion.get_num_subregions()),), so2_key +
                          (np.random.randint(self._so2_branching),))

    def get_minimum_depth_for_construction(self):
        """
            Return the minimal depth, i.e. length of a key, for which it is possible
            to construct a solution.
        """
        return 3

    def is_leaf(self, key):
        """
            Return whether the given key corresponds to a leaf node.
        """
        if len(key) < 3:
            return False
        if len(key) == 3:
            subregion_key = ()
            so2_key = ()
        else:
            assert(len(key) == 5)
            # extract sub region key
            subregion_key = key[3]
            so2_key = key[4]
        subregion = self.get_placement_region((key[1], subregion_key))
        b_region_leaf = not subregion.has_subregions()
        b_so2_leaf = so2hierarchy.is_leaf(so2_key, self._so2_depth)
        return b_region_leaf and b_so2_leaf

    def get_num_children(self, key):
        """
            Return the total number of possible children for the given key.
        """
        if len(key) == 0:
            return len(self._manips)
        if len(key) == 1:
            return len(self._regions)
        if len(key) == 2:
            return len(self._orientations)
        # extract sub region key
        if len(key) == 5:
            subregion_key = key[3]
            so2_key = key[4]
        else:
            assert(len(key) == 3)
            subregion_key = ()
            so2_key = ()
        subregion = self.get_placement_region((key[1], subregion_key))
        b_region_leaf = not subregion.has_subregions()
        b_so2_leaf = so2hierarchy.is_leaf(so2_key, self._so2_depth)
        if b_region_leaf and b_so2_leaf:
            return 0
        if b_region_leaf:
            return self._so2_branching
        if b_so2_leaf:
            return subregion.get_num_subregions()
        return subregion.get_num_subregions() * self._so2_branching

    def get_placement_region(self, region_key):
        """
            Return the placement region with the specified region key.
            The key may either indentify a whole region or a subregion.
            ---------
            Arguments
            ---------
            region_key, int of tuple(int, tuple(int, ...)) -
                The region key can either be a single int or a tuple consisting of an
                int and another tuple of ints. In case it is a single int, it identifies
                a placement region. In case it is a tuple of the form (int, tuple(int, ...)),
                region_key[0] identifies the placement region and region_key[1] identifies the
                subregion of that region.
        """
        if type(region_key) is int:
            return self._regions[region_key]
        assert(type(region_key)) is tuple
        region = self._regions[region_key[0]]
        return region.get_subregion(region_key[1])

    def get_arpo_information(self, key):
        """
            Return the arm, region, placement orientation, subregion and so2 interval represented
            by the given key.
            ---------
            Arguments
            ---------
            key, tuple as described in class definition
        """
        if len(key) == 0:
            return None
        elif len(key) == 1:
            return (self._manips[key[0]],)
        elif len(key) == 2:
            return (self._manips[key[0]], self._regions[key[1]])
        elif len(key) == 3:
            return (self._manips[key[0]], self._regions[key[1]], self._orientations[key[2]])
        assert(len(key) == 5)
        subregion = self.get_placement_region((key[1], key[3]))
        so2region = so2hierarchy.get_interval(key[4][:self._so2_depth], self._so2_branching)
        return (self._manips[key[0]], self._regions[key[1]], self._orientations[key[2]], subregion, so2region)

    def get_all_manip_orientations(self):
        """
            Returns a generator of all possible combinations of manipulators and placement orientations.
            ---- Used by ARPORobotBridge to initialize IK solvers ----
            -------
            Returns
            -------
            generator for list of tuples ((mid, manip), (oid, po)), where (mid, manip) is the manipulator (manip)
                and its id, and (oid, po) the placement orientation (po) and its id (oid).
        """
        return itertools.product(enumerate(self._manips), enumerate(self._orientations))


class ARPORobotBridge(placement_interfaces.PlacementSolutionConstructor,
                      placement_interfaces.PlacementValidator,
                      placement_interfaces.PlacementObjective):
    """
        An ARPORobotBridge serves as the central interface for a placement planner
        operating on the ARPOHierarchy. The ARPORobotBridge fulfills multiple functionalities
        including a solution constructor, validator and objective function. The reason
        this class provides all these different functions together is that this way
        we can cache a lot at a single location.
        # TODO do we really need all this in one class?
    """
    class ObjectData(object):
        """
            Struct that stores object data.
            It stores a kinbody, i.e. the object, and a volumetric representation of it -
            an instance of sdf.kinbody.OccupancyTree or sdf.kinbody.RigidBodyOccupancyGrid
        """

        def __init__(self, kinbody, occtree):
            self.kinbody = kinbody
            self.volumetric_model = occtree

    class RobotData(object):
        """
            Struct that stores robot data
        """

        def __init__(self, robot, robot_occtree, manip_data, urdf_desc, ball_approx):
            """
                Create a new instance of robot data.
                ---------
                Arguments
                ---------
                robot - OpenRAVE robot
                robot_volumetric - volumetric model of the robot (either RobotOccupancyGrid or RobotOccupancyOctree)
                manip_data, dict of ManipulatorData - dict that maps manipulator names to ManipulatorData struct
                urdf_desc, string - URDF fliename of the robot (content, not the filename!)
                ball_approx, RobotBallApproximation
            """
            self.robot = robot
            self.volumetric_model = robot_occtree
            self.manip_data = manip_data
            self.urdf_desc = urdf_desc
            self.ball_approx = ball_approx

    class ManipulatorData(object):
        """
            Struct (named tuple) that stores manipulator data.
        """

        def __init__(self, manip, ik_solver, reachability_map, grasp_tf, grasp_config):
            """
                Create a new instance of manipulator data.
                ---------
                Arguments
                ---------
                manip - OpenRAVE manipulator
                ik_solver - ik_module.IKSolver, IK solver for end-effector poses
                reachability_map - ReachabilityMap for this manipulator
                grasp_tf - grasp transform (eef pose in object frame)
                grasp_config, numpy array (n_h,) - hand configuration for grasp
            """
            self.manip = manip
            self.manip_links = utils.get_manipulator_links(manip)
            self.grasp_tf = grasp_tf
            self.grasp_config = grasp_config
            self.inv_grasp_tf = utils.inverse_transform(self.grasp_tf)
            self.reachability_map = reachability_map
            self.ik_solver = ik_solver
            self.lower_limits, self.upper_limits = self.manip.GetRobot().GetDOFLimits(manip.GetArmIndices())

    class SolutionCacheEntry(object):
        """
            Cache all relevant information for a particular solution.
        """

        def __init__(self, key, region, plcmnt_orientation, so2_interval, solution):
            self.key = key  # arpo hierarchy key
            self.solution = solution  # PlacementGoal
            self.region = region  # PlacementRegion from key
            self.plcmnt_orientation = plcmnt_orientation  # PlacementOrientation from key
            self.so2_interval = so2_interval  # SO2 interval from key
            self.eef_tf = None  # store end-effector transform
            self.bkinematically_reachable = None  # store whether ik solutions exist
            self.bcollision_free = None  # store whether collision-free
            self.bbetter_objective = None  # store whether it has better objective than the current best
            self.bstable = None  # store whether pose is actually a stable placement
            self.objective_val = None  # store objective value
            # the following elements are used for gradient-based optimization
            self.jacobian = None  # Jacobian for active manipulator (and object reference pos)
            self.region_pose = None  # pose of the object relative to region frame
            self.region_state = None  # (x, y, theta) within region

    class ContactConstraint(object):
        """
            This constraint expresses that all contact points of a placement face need to
            be in contact with a support surface, i.e. within a placement region.
        """

        def __init__(self, global_region_info):
            """
                Create a new instance of contact constraint.
            """
            self._contact_point_distances = global_region_info[0]
            self._contact_point_gradients = global_region_info[1]
            self.eps = 0.005

        def check_contacts(self, cache_entry):
            """
                Check whether the solution stored in cache entry represents a stable placement.
                Returns True or False.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                ------------
                Side effects
                ------------
                sets the bstable flag in cache_entry
                -------
                Returns
                -------
                bstable, bool - whether the solution is stable
            """
            cache_entry.bstable = False
            obj_tf = cache_entry.solution.obj_tf
            po = cache_entry.plcmnt_orientation
            contact_points = np.dot(po.placement_face[1:], obj_tf[:3, :3].transpose()) + obj_tf[:3, 3]
            values = self._contact_point_distances.get_cell_values_pos(contact_points)
            none_values = values == None  # Ignore linter warning!
            if none_values.any():
                return False
            cache_entry.bstable = (values == 0.0).all()
            return cache_entry.bstable

        def get_relaxation(self, cache_entry):
            """
                Compute relaxation of contact constraint.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                -------
                Returns
                -------
                val, float - relaxation value
            """
            if cache_entry.solution.obj_tf is None:
                return 0.0
            obj_tf = cache_entry.solution.obj_tf
            po = cache_entry.plcmnt_orientation
            contact_points = np.dot(po.placement_face[1:], obj_tf[:3, :3].transpose()) + obj_tf[:3, 3]
            values = self._contact_point_distances.get_cell_values_pos(contact_points)
            none_values = values == None  # Ignore linter warning!
            if none_values.any():
                rospy.logwarn(
                    "Contact relaxation encountered None values. This should not happen! The contact point distance field is too small")
                return 0.0  # contact points are out of range, that means it's definitely a bad placement
            # TODO this may still fail if placement planes are not perfect planes...
            # TODO I.e. the height variance is larger than the cell size
            # assert((values != float('inf')).all())
            if (values == float('inf')).any():
                rospy.logerr("Invalid object pose detected - DEBUG!")
                import IPython
                IPython.embed()
            # get max distance. Clip it because the signed distance field isn't perfectly accurate
            max_distance = np.clip(np.max(values), 0.0, po.max_contact_pair_distance)
            return 1.0 - max_distance / po.max_contact_pair_distance

        def compute_cart_gradient(self, cache_entry, ref_pose):
            """
                Return the gradient w.r.t x, y, ez of distance to contacts.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                ref_pose, np.array of shape (4, 4) - pose of reference contact point
                TODO: this function assumes that the placement plane frame and the world frame are aligned
                -------
                Returns
                -------
                gradient, np.array of shape (3,) - gradient of contact constraint w.r.t. x, y, ez
            """
            # first extract contact points in local and global frame
            # exclude reference point itself
            local_contact_points = cache_entry.plcmnt_orientation.local_placement_face[2:]
            global_contact_points = np.matmul(local_contact_points, ref_pose[:3, :3].transpose()) + ref_pose[:3, 3]
            values = self._contact_point_distances.get_cell_values_pos(global_contact_points)
            # retrieve gradients w.r.t. x, y at contact points
            valid_flags, cart_gradients = self._contact_point_gradients.get_interpolated_vectors(global_contact_points)
            if not valid_flags.all():
                rospy.logerr(
                    "Extracting contact constraint gradients failed. This should not happen!" +
                    "The gradient field is too small or sth else is wrong. Try debugging:")
                import IPython
                IPython.embed()
                return np.zeros(3)
            # compute chomp's smooth distances
            _, cart_gradients = utils.chomps_distance(-values + self.eps, self.eps, -cart_gradients[:, :2])
            # next, we need to compute the gradient w.r.t. theta
            # filter zero gradients
            non_zero_idx = np.nonzero(cart_gradients)[0]  # rows with non zero gradients
            non_zero_grads, non_zero_pos = cart_gradients[non_zero_idx], local_contact_points[non_zero_idx]
            if non_zero_grads.shape[0] == 0:
                return np.zeros(3)
            # get object state
            x, y, theta = cache_entry.region_state
            # compute gradient w.r.t to state
            r = np.array([[-np.sin(theta), np.cos(theta)], [-np.cos(theta), -np.sin(theta)]])
            dxi_dtheta = np.matmul(non_zero_pos[:, :2], r)
            lcart_grads = np.empty((non_zero_grads.shape[0], 3))
            lcart_grads[:, :2] = non_zero_grads[:, :2]
            dthetas = np.sum(non_zero_grads[:, :2] * dxi_dtheta, axis=1)
            lcart_grads[:, 2] = dthetas
            return 1.0 / lcart_grads.shape[0] * np.sum(lcart_grads, axis=0)

            # compute transform from region base frame to current pose of reference point
            # tf_r_p = np.matmul(utils.inverse_transform(cache_entry.region.base_tf), ref_pose)
            # # TODO the following assumes that ez == theta, i.e. placement region and world frame are aligned
            # # a position in global_contact_points is computed as base_tf * tf_r_p * local_position
            # # thus if we rotate around the reference position tf_r_p[:3, 3] (i.e. change theta),
            # # the position changes. Assuming that base_tf's z axis is aligned with the world frame,
            # # we can compute the gradient w.r.t to theta/ez as base_tf * dR/dtheta * gradient_x,y
            # # compute dR/dtheta, since the rotation is only a rotation around the z axis, we can simply construct
            # # the derivative by swapping the x and y column and negating the x column
            # x_column = np.array(tf_r_p[:, 0])
            # y_column = np.array(tf_r_p[:, 1])
            # tf_r_p[:, 0] = y_column
            # tf_r_p[:, 1] = -x_column
            # # transformation matrix for gradient w.r.t ez
            # dtf = np.matmul(cache_entry.region.base_tf, tf_r_p)
            # dcontact_dtheta = np.matmul(local_contact_points, dtf[:3, :3].transpose())  # dworld_contact_points / dtheta
            # non_zero_grads = np.nonzero(cart_gradients)[0]
            # grad = np.empty(3)
            # grad[:2] = np.sum(cart_gradients, axis=0)  # x, y part of gradient
            # # theta part = sum(<dcontact_dtheta[i], cart_gradients[i]>) with cart_gradient[i] = dd(c)/dx, dy
            # # TODO drops z coordinate because we assume that frames are aligned
            # grad[2] = np.sum(dcontact_dtheta[:, :2] * cart_gradients)
            # return grad / non_zero_grads.shape[0]  # the cost is average distance

    class CollisionConstraint(object):
        """
            This constraint expresses that both the manipulator and the target object is
            not in collision with anything.
        """

        def __init__(self, object_data, robot_data, scene_sdf,
                     max_robot_intersection=0.1):
            """
                Create a new instance of a collision constraint.
                ---------
                Arguments
                ---------
                object_data, ARPORobotBridge.ObjectData - information about the object
                robot_data, ARPORobotBridge.RobotData - information about the robot
                scene_sdf, sdf.core.SceneSDF - signed distance field of the scene
                    to compute intersection, i.e. constraint relaxation
                max_robot_intersection, float - determines maximum percentage to which the robot
                    may be in collision so that the robot's contribution to the
                    contraint relaxation value is non-zero
            """
            self._target_obj = object_data.kinbody
            self._robot = robot_data.robot
            self._manip_data = robot_data.manip_data
            self._robot_volumetric = robot_data.volumetric_model
            self._robot_ball_approx = robot_data.ball_approx
            self._object_volumetric = object_data.volumetric_model
            self._scene_sdf = scene_sdf
            self._max_robot_intersection = max_robot_intersection

        def check_collision(self, cache_entry):
            """
                Check whether the solution stored in cache entry represents a collision-free placement.
                Returns True or False. TODO: implement relaxation
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                ------------
                Side effects
                ------------
                sets the bcollision_free flag in cache_entry
            """
            cache_entry.bcollision_free = False
            if cache_entry.solution.arm_config is None:
                return False
            manip = cache_entry.solution.manip
            manip_data = self._manip_data[manip.GetName()]
            robot = manip.GetRobot()
            with robot:
                with orpy.KinBodyStateSaver(self._target_obj):
                    # grab object (sets active manipulator for us)
                    utils.set_grasp(manip, self._target_obj,
                                    manip_data.inv_grasp_tf, manip_data.grasp_config)
                    robot.SetDOFValues(cache_entry.solution.arm_config, manip.GetArmIndices())
                    env = robot.GetEnv()
                    cache_entry.bcollision_free = not env.CheckCollision(robot) and not robot.CheckSelfCollision()
                    robot.Release(self._target_obj)
            return cache_entry.bcollision_free

        def get_relaxation(self, cache_entry):
            """
                Compute relaxation of collision constraint.
                # TODO
            """
            robot_intersection = 0.0
            arm_config = None
            if cache_entry.solution.arm_config is None:
                return 0.0
                # TODO compute intersection using approximate arm configuration
                robot_intersection = 0.0
            else:
                arm_config = cache_entry.solution.arm_config
            # first compute intersection for the arm
            if arm_config is not None:
                manip = cache_entry.solution.manip
                manip_data = self._manip_data[manip.GetName()]
                robot = manip.GetRobot()
                with robot:
                    robot.SetActiveDOFs(manip.GetArmIndices())
                    isec_values = self._robot_volumetric.compute_intersection(
                        robot.GetTransform(), arm_config, self._scene_sdf, links=manip_data.manip_links)
                    robot_intersection = isec_values[1]
            # next compute intersection for the object
            obj_tf = np.array(cache_entry.solution.obj_tf)
            # shift the obj tf a bit away from the contact region to ensure we do not count in contact on the surface
            obj_tf[:3, 3] += cache_entry.region.normal * self._object_volumetric.get_cell_size()
            isec_values = self._object_volumetric.compute_intersection(self._scene_sdf, obj_tf)
            # from this compute relaxation value that is in interval 0, 1
            object_intersection = isec_values[1]
            violation_term = np.clip((object_intersection + robot_intersection /
                                      self._max_robot_intersection)/2.0, 0.0, 1.0)
            return 1.0 - violation_term

        def get_cart_obj_collision_gradient(self, cache_entry):
            """
                Return the cartesian gradient of CHOMP's collision cost for the graped object.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                    Requirements: 
                        cache_entry.region_state is set
                -------
                Returns
                -------
                gradient, np array of shape (3,) - gradient of CHOMP's collision cost for the object
                    w.r.t x, y, theta(ez)
            """
            _, cart_grads, loc_positions = self._object_volumetric.compute_obstacle_cost(
                self._scene_sdf, bgradients=True)
            # translate local positions into positions relative to reference pose
            to_ref_pose = cache_entry.plcmnt_orientation.inv_reference_tf
            loc_positions = np.dot(loc_positions, to_ref_pose[:3, :3].T) + to_ref_pose[:3, 3]
            # filter zero gradients
            non_zero_idx = np.nonzero(cart_grads[:, :2])[0]
            non_zero_grads, non_zero_pos = cart_grads[non_zero_idx], loc_positions[non_zero_idx]
            if non_zero_grads.shape[0] == 0:
                return np.zeros(3)
            # get object state
            x, y, theta = cache_entry.region_state
            # compute gradient w.r.t to state
            r = np.array([[-np.sin(theta), np.cos(theta)], [-np.cos(theta), -np.sin(theta)]])
            dxi_dtheta = np.matmul(non_zero_pos[:, :2], r)
            lcart_grads = np.empty((non_zero_grads.shape[0], 3))
            lcart_grads[:, :2] = non_zero_grads[:, :2]
            lcart_grads[:, 2] = np.sum(non_zero_grads[:, :2] * dxi_dtheta, axis=1)
            return 1.0 / lcart_grads.shape[0] * np.sum(lcart_grads, axis=0)

        def get_obj_collision_gradient(self, cache_entry, config):
            """
                Return the gradient of CHOMP's collision cost applied to the grasped object w.r.t
                to the manipulator's configuration.
                *The gradient is computed such that it tries to avoid moving out of the placement region.
                NOTE: For this to work correctly, the active DOFs of the robot must be set before.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                config, np array of shape (q,) - robot configuration with q = active DOFs elements
                -------
                Returns
                -------
                gradient, np array of shape (q,) - gradient of CHOMP's collision cost for the object
            """
            manip = cache_entry.solution.manip
            # manip_data = self._manip_data[manip.GetName()]
            _, cart_grads, loc_positions = self._object_volumetric.compute_obstacle_cost(
                self._scene_sdf, bgradients=True)
            gradient = np.zeros(manip.GetArmDOF())
            # translate local positions into positions relative to reference pose
            to_ref_pose = cache_entry.plcmnt_orientation.inv_reference_tf
            loc_positions = np.dot(loc_positions, to_ref_pose[:3, :3].T) + to_ref_pose[:3, 3]
            # eef_index = manip.GetEndEffector().GetIndex()
            non_zero_idx = np.nonzero(cart_grads[:2])[0]
            non_zero_grads, non_zero_pos = cart_grads[non_zero_idx], loc_positions[non_zero_idx]
            trimmed_jac = np.array([cache_entry.jacobian[0], cache_entry.jacobian[1], cache_entry.jacobian[2]])
            r = np.array(cache_entry.relative_tf[:2, :2])
            x_column = np.array(r[:, 0])
            y_column = np.array(r[:, 1])
            r[:, 0] = y_column
            r[:, 1] = -x_column
            for cart_grad, lpos in itertools.izip(non_zero_grads, non_zero_pos):
                # pos = np.matmul(manip_data.inv_grasp_tf[:3, :3], lpos) + manip_data.inv_grasp_tf[:3, 3]
                # jacobian = self._robot.CalculateActiveJacobian(eef_index, pos)
                dpdq = trimmed_jac[:2] + np.matmul(np.matmul(r, lpos[:2]).reshape(2, 1),
                                                   trimmed_jac[2].reshape((1, trimmed_jac[2].shape[0])))
                gradient += np.matmul(cart_grad[:2], dpdq)
                # gradient += np.matmul(cart_grad, jacobian)  # it's jacobian.T * cart_grad
            # gradient = gradient * 1.0 / loc_positions.shape[0]
            return 1.0 / loc_positions.shape[0] * gradient

        def get_chomps_collision_gradient(self, cache_entry, config):
            """
                Return the gradient of CHOMP's collision cost.
                NOTE: For this to work correctly, the active DOFs of the robot must be set before.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                config, np array of shape (q,) - robot configuration with q = active DOFs elements
                -------
                Returns
                -------
                gradient, np array of shape (q,) - gradient of CHOMP's collision cost
            """
            manip = cache_entry.solution.manip
            manip_data = self._manip_data[manip.GetName()]
            gradient = np.zeros(manip.GetArmDOF())
            # _, gradient = self._robot_volumetric.compute_penetration_cost(
            #     self._scene_sdf, config, b_compute_gradient=True, links=manip_data.manip_links)
            # compute the gradient w.r.t to q for object
            # gradient += self.get_obj_collision_gradient(cache_entry, config)
            # get distances from ball approximation (much faster)
            _, gradient = self._robot_ball_approx.compute_penetration_cost(self._scene_sdf, manip_data.manip_links)
            return gradient

    class ReachabilityConstraint(object):
        """
            This constraint expresses that a solution needs to be kinematically reachable.
        """

        def __init__(self, robot_data, baggressive=False):
            """
                If baggressive is True, solutions for which no arm configuration is set, receive
                0.0 as relaxation value. In fact, if it is aggressive, the constraint relaxation
                is binary (either 1.0 or 0.0).
            """
            self._manip_data = robot_data.manip_data
            self._baggressive = baggressive

        def check_reachability(self, cache_entry):
            """
                Check whether the solution stored in cache entry is a kinematically reachable solution.
                Returns True or False.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                ------------
                Side effects
                ------------
                sets the bkinematically_reachable flag in cache_entry
                -------
                Returns
                -------
                bool whether solution is reachable or not
            """
            cache_entry.bkinematically_reachable = cache_entry.solution.arm_config is not None
            return cache_entry.bkinematically_reachable

        def get_relaxation(self, cache_entry):
            """
                Compute relaxation of reachability constraint.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                ------------
                Side effects
                ------------
                TODO: set approximate arm configuration
                -------
                Returns
                -------
                val, float - relaxation value in [0, 1]
            """
            if cache_entry.solution.arm_config is not None:
                return 1.0
            if self._baggressive:
                return 0.0
            # else compute heuristic value using reachability map
            manip = cache_entry.solution.manip
            manip_data = self._manip_data[manip.GetName()]
            pose = np.empty((1, 7))
            pose[0, :3] = cache_entry.eef_tf[:3, 3]
            pose[0, 3:] = orpy.quatFromRotationMatrix(cache_entry.eef_tf[:3, :3])
            _, nn_poses, configs = manip_data.reachability_map.query(pose)
            cart_dist = np.linalg.norm(pose[0, :3] - nn_poses[0, :3])
            quat_dist = so3hierarchy.quat_distance(pose[0, 3:], nn_poses[0, 3:])
            # cache_entry. TODO set arm config to configs[0]?
            # relate cartesian distance to the size of the placement region
            # normalize quaternion distance
            return 1.0 - 0.5 * np.clip(cart_dist / cache_entry.region.radius, 0.0, 1.0) - 0.5 * quat_dist / np.pi

        def get_pose_reachability_fn(self, cache_entry):
            """
                Return a function that maps a tuple (x, y, theta) to a reachability value in R.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                -------
                Returns
                -------
                a function fn that returns a reachability value r, i.e. fn(x) = r
                good_val, float - a value that if fn(x) <= good_val, x is very likely to be kinematically reachable
            """
            def reachability_fn(val, manip_data, region, po):
                local_pose = tf_mod.rotation_matrix(val[2], [0, 0, 1])
                local_pose[:2, 3] = val[:2]
                obj_tf = np.dot(region.base_tf, np.dot(local_pose, po.inv_reference_tf))
                # compute end-effector tf
                eef_tf = np.dot(obj_tf, manip_data.grasp_tf)
                pose = np.empty((1, 7))
                pose[0, :3] = eef_tf[:3, 3]
                pose[0, 3:] = orpy.quatFromRotationMatrix(eef_tf[:3, :3])
                # cart_distances, quat_distances = manip_data.reachability_map.query(pose)
                distances, _, _ = manip_data.reachability_map.query(pose)
                # return distances[0]
                # print "Reachability fn:", cart_distances[0] + 0.1 * quat_distances[0]
                # return cart_distances[0] + 0.1 * quat_distances[0]
                return distances[0]

            manip = cache_entry.solution.manip
            manip_data = self._manip_data[manip.GetName()]
            good_val = 0.5 * manip_data.reachability_map.get_dispersion()
            fn = partial(reachability_fn, manip_data=manip_data,
                         region=cache_entry.region, po=cache_entry.plcmnt_orientation)
            return fn, good_val

    # class ObjectiveImprovementConstraint(object):
    #     """
    #         This constraint expresses that a new solution needs to be better than a previously
    #         found one.
    #     """

    #     def __init__(self):
    #         """
    #             Construct new QualityImprovementConstraint
    #         """
    #         self._best_value = float('inf')

    #     def check_objective_improvement(self, cache_entry):
    #         """
    #             Check whether the solution stored in cache entry is has a better quality than the best
    #             reached so far.
    #             Returns True or False. TODO: implement relaxation
    #             ---------
    #             Arguments
    #             ---------
    #             cache_entry, SolutionCacheEntry
    #             ------------
    #             Side effects
    #             ------------
    #             sets the bbetter_objective flag in cache_entry
    #         """
    #         cache_entry.bbetter_objective = cache_entry.solution.objective_value < self._best_value
    #         return cache_entry.bbetter_objective

    #     def get_relaxation(self, cache_entry):
    #         """
    #             Compute relaxation of objctive constraint.
    #             #TODO
    #         """
    #         # TODO requires known objective values so far
    #         pass
    class JacobianOptimizer(object):
        """
            Optimization algorithm that follows the gradient of all contraints to locally search for
            feasible solutions.
        """

        def __init__(self, contact_constraint, collision_constraint, objective_constraint, robot_data):
            """
                Create a new JacobianOptimizer.
                ---------
                Arguments
                ---------
                contact_constraint, ContactConstraint
                collision_constraint, CollisionConstraint
                objective_constraint, ObjectiveConstraint TODO
            """
            self.contact_constraint = contact_constraint
            self.collision_constraint = collision_constraint
            self.objective_constraint = objective_constraint
            self.robot_data = robot_data
            self.manip_data = robot_data.manip_data
            self.epsilon = 0.01  # minimal magnitude of cspace gradient
            self.step_size = 0.01  # multiplier for update step
            self.max_iterations = 100  # maximal number of iterations
            self.damping_matrix = np.diag([0.9, 0.9, 1.0, 1.0, 1.0, 0.9])  # damping matrix for nullspace projection
            self.robot = robot_data.robot

        def optimize(self, cache_entry):
            """
                Attempt to make the arm configuration in cache_entry to feasible.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                ------------
                Side effects
                ------------
                cache_entry.solution.obj_tf, numpy array of shape (4, 4) - pose of the object is stored here
                cache_entry.eef_tf, numpy array of shape (4, 4) - pose of the end-effector is stored here
                cache_entry.solution.arm_config, numpy array of shape (n,) - arm configuration (n DOFs)
            """
            rospy.logdebug("Running JacobianOptimizer to make solution feasible")
            q_original = self.robot.GetActiveDOFValues()  # TODO remove
            manip_data = self.manip_data[cache_entry.solution.manip.GetName()]
            lower, upper = manip_data.lower_limits, manip_data.upper_limits
            manip = manip_data.manip
            # with self.robot:
            reference_pose = np.dot(manip_data.inv_grasp_tf, cache_entry.plcmnt_orientation.reference_tf)
            manip.SetLocalToolTransform(reference_pose)
            self.robot.SetActiveDOFs(manip.GetArmIndices())
            # init jacobian descent
            q_current = cache_entry.solution.arm_config
            q_best = np.array(q_current)
            # compute jacobian
            q_grad = self._compute_gradient(cache_entry, q_current, manip_data)
            if not q_grad is None:
                grad_norm = np.linalg.norm(q_grad)
            else:
                grad_norm = 0.0
            b_in_limits = (q_current >= lower).all() and (q_current <= upper).all()
            # iterate as long as the gradient is not zero and we are not beyond limits
            # while q_grad is not None and grad_norm > self.epsilon and b_in_limits:
            for i in xrange(self.max_iterations):
                if q_grad is None or grad_norm < self.epsilon or not b_in_limits:
                    break
                rospy.logdebug("Updating q_current %s in direction of gradient %s (magnitude %f)" %
                               (str(q_current), str(q_grad), grad_norm))
                q_current -= self.step_size * q_grad / grad_norm
                b_in_limits = (q_current >= lower).all() and (q_current <= upper).all()
                if b_in_limits:
                    # save q_current and compute next jacobian + gradient
                    q_best[:] = q_current
                    q_grad = self._compute_gradient(cache_entry, q_current, manip_data)
                    if q_grad is not None:
                        grad_norm = np.linalg.norm(q_grad)
                    else:
                        grad_norm = 0.0
                else:
                    rospy.logdebug("Jacobian descent has led to joint limit violation. Aborting")
                # iteration += 1
            rospy.logdebug("Jacobian descent finished after %i iterations" % i)
            cache_entry.solution.arm_config = q_best
            manip.SetLocalToolTransform(np.eye(4))
            self.robot.SetActiveDOFValues(q_original)  # TODO remove

        def _compute_gradient(self, cache_entry, q_current, manip_data):
            """
                Compute gradient for the current configuration.
                -------
                Returns
                -------
                qgrad, numpy array of shape (manip.GetArmDOF,) - gradient of po + collision constraints
                    w.r.t arm configurations. The returned gradient is None, if we either moved out
                    of range where we can compute it, or when we hit a singularity.
            """
            manip = manip_data.manip
            # with self.robot:
            self.robot.SetActiveDOFValues(q_current)
            # TODO delete
            crayola = self.robot.GetEnv().GetKinBody("crayola")
            crayola.SetTransform(
                np.matmul(cache_entry.solution.manip.GetEndEffector().GetTransform(), manip_data.inv_grasp_tf))
            # TODO until here
            # compute jacobian
            jacobian = np.empty((6, manip.GetArmDOF()))
            jacobian[:3] = manip.CalculateJacobian()
            jacobian[3:] = manip.CalculateAngularVelocityJacobian()
            cache_entry.jacobian = jacobian
            # compute pseudo inverse
            inv_jac, rank = utils.compute_pseudo_inverse_rank(jacobian)
            if rank < 6:  # if we are in a singularity, just return None
                # TODO instead of doing this, we could also only skip all Cartesian gradients
                rospy.logdebug("Jacobian descent failed: Ran into singularity.")
                return None
            # get pose of placement reference point
            ref_pose = manip.GetEndEffectorTransform()  # this takes the local tool transform into account
            # compute relative pose and state
            cache_entry.relative_tf = np.matmul(utils.inverse_transform(cache_entry.region.base_tf), ref_pose)
            ex, ey, ez = tf_mod.euler_from_matrix(cache_entry.relative_tf)
            theta = utils.normalize_radian(ez)
            rospy.logdebug("Current [x=%f, y=%f, z=%f, ex=%f, ey=%f, ez=%f, theta=%f]" %
                           (cache_entry.relative_tf[0, 3], cache_entry.relative_tf[1, 3], cache_entry.relative_tf[2, 3],
                            ex, ey, ez, theta))
            cache_entry.region_state = (cache_entry.relative_tf[0, 3], cache_entry.relative_tf[1, 3], theta)
            # Compute gradient w.r.t. constraints
            # ----- 1. Region constraint - reference point needs to be within the selected placement region
            region = cache_entry.region
            query_pos = ref_pose[:3, 3].reshape((1, 3))
            # values = region.aabb_distance_field.get_cell_values_pos(query_pos)
            b_valid, region_pos_grad = region.aabb_dist_gradient_field.get_interpolated_vectors(query_pos)
            if not b_valid[0]:
                rospy.logdebug("Jacobian descent failed: It went out of the placement region's aabb.")
                return None
            cart_grad_r = np.array([region_pos_grad[0, 0], region_pos_grad[0, 1], 0.0])
            # TODO could also make this distance smooth
            # _, cart_grad_r = utils.chomps_distance()
            rospy.logdebug("In-region constraint gradient is %s" % str(cart_grad_r))
            # ----- 2. Theta constraint - theta needs to be within the selected so2interval
            # We compute no gradient for this constraint, and instead simply abort if we violate it
            if utils.dist_in_range(theta, cache_entry.so2_interval) != 0.0:
                rospy.logdebug("Jacobian descent failed: Theta is out of so2 interval.")
                return None
            # ------ 3. Stability - all contact points need to be in a placement region (any)
            cart_grad_c = self.contact_constraint.compute_cart_gradient(cache_entry, ref_pose)
            rospy.logdebug("Contact constraint gradient is %s" % str(cart_grad_c))
            # ------ 4. Collision constraint - object must not be in collision
            cart_grad_col = self.collision_constraint.get_cart_obj_collision_gradient(cache_entry)
            rospy.logdebug("Object collisions constraint gradient is %s" % str(cart_grad_col))
            # ------ 5. Translate cartesian gradients into c-space gradients
            cart_grad = cart_grad_r + cart_grad_c + cart_grad_col
            rospy.logdebug("Resulting cartesian gradient (x, y, theta): %s" % str(cart_grad))
            extended_cart = np.zeros(6)
            extended_cart[:2] = cart_grad[:2]
            extended_cart[5] = cart_grad[2]
            qgrad = np.matmul(inv_jac, extended_cart)
            # sub_jacobian = np.array([jacobian[0], jacobian[1], jacobian[5]])
            # qgrad = np.matmul(cart_grad, sub_jacobian)
            # ------ 5. Collision constraint - arm must not be in collision
            col_grad = self.collision_constraint.get_chomps_collision_gradient(cache_entry, q_current)
            col_grad = np.matmul((np.eye(qgrad.shape[0]) - np.matmul(inv_jac,
                                                                     np.matmul(self.damping_matrix, jacobian))), qgrad)
            col_cart = np.matmul(np.matmul(self.damping_matrix, jacobian), col_grad)
            rospy.logdebug("Arm collision gradient: %s. Results in Cartesian motion: %s " % (str(col_grad), col_cart))
            qgrad += col_grad
            # remove any motion that changes the base orientation/z height of the object
            jacobian[[0, 1, 5], :] = 0.0  # motion in x, y, ez is allowed
            qgrad = np.matmul((np.eye(qgrad.shape[0]) - np.matmul(inv_jac, jacobian)), qgrad)
            return qgrad

    def __init__(self, arpo_hierarchy, robot_data, object_data,
                 objective_fn, global_region_info, scene_sdf):
        """
            Create a new ARPORobotBridge
            ---------
            Arguments
            ---------
            arpo_hierarchy, ARPOHierarchy - arpo hierarchy to create solutions for
            robot_data, RobotData - struct that stores robot information including ManipulatorData for each manipulator
            object_data, ObjectData - struct that stores object information
            objective_fn, ??? - TODO
            global_region_info, (VoxelGrid, VectorGrid) - stores for the full planning scene distances to where
                contact points may be located, as well as gradients
        """
        self._hierarchy = arpo_hierarchy
        self._robot_data = robot_data
        self._manip_data = robot_data.manip_data  # shortcut to manipulator data
        self._objective_fn = objective_fn
        self._object_data = object_data
        self._contact_constraint = ARPORobotBridge.ContactConstraint(global_region_info)
        self._collision_constraint = ARPORobotBridge.CollisionConstraint(object_data, robot_data, scene_sdf)
        self._reachability_constraint = ARPORobotBridge.ReachabilityConstraint(robot_data, True)
        # self._objective_constraint = ARPORobotBridge.ObjectiveImprovementConstraint() # TODO move this into goal sampler
        self._solutions_cache = []  # array of SolutionCacheEntry objects
        self._call_stats = np.array([0, 0, 0, 0])  # num sol constructions, is_valid, get_relaxation, evaluate
        self._plcmnt_ik_solvers = {}  # maps (manip_id, placement_orientation_id) to a trac_ik solver
        self._init_ik_solvers()
        self._jacobian_optimizer = ARPORobotBridge.JacobianOptimizer(self._contact_constraint, self._collision_constraint,
                                                                     None, self._robot_data)  # TODO objective constraint?

    def construct_solution(self, key, b_optimize_constraints=False, b_optimize_objective=False):
        """
            Construct a new PlacementSolution from a hierarchy key.
            ---------
            Arguments
            ---------
            key, object - a key object that identifies a node in a PlacementHierarchy
            boptimize_constraints, bool - if True, the solution constructor may put additional computational
                effort into computing a valid solution, e.g. some optimization of a constraint relaxation
            b_optimize_objective, bool - if True, the solution constructor may optimize an objective
                given the hierarchy key
            -------
            Returns
            -------
            PlacementSolution sol, a placement solution for the given key
        """
        if len(key) < self._hierarchy.get_minimum_depth_for_construction():
            raise ValueError("Could not construct solution for the given key: " + str(key) +
                             " This key is describing a node too high up in the hierarchy")
        arpo_info = self._hierarchy.get_arpo_information(key)
        assert(len(arpo_info) >= 3)
        manip = arpo_info[0]
        manip_data = self._manip_data[manip.GetName()]
        region = arpo_info[1]
        po = arpo_info[2]
        so2_interval = np.array([0, 2.0 * np.pi])
        if len(arpo_info) == 5:
            region = arpo_info[3]
            so2_interval = arpo_info[4]
        # construct a solution without valid values yet
        new_solution = placement_interfaces.PlacementGoalSampler.PlacementGoal(
            # TODO real objective value
            manip=manip, arm_config=None, obj_tf=None, key=len(self._solutions_cache), objective_value=0.0,
            grasp_tf=None, grasp_config=manip_data.grasp_config)
        # create a cache entry for this solution
        sol_cache_entry = ARPORobotBridge.SolutionCacheEntry(
            key=key, region=region, plcmnt_orientation=po, so2_interval=so2_interval, solution=new_solution)
        self._solutions_cache.append(sol_cache_entry)
        if b_optimize_constraints:
            # compute object pose and arm configuration jointly
            self._optimize_constraints(sol_cache_entry, b_optimize_objective)
        else:
            # compute object and end-effector pose
            self._compute_object_pose(sol_cache_entry)
            # compute arm configuration
            sol_cache_entry.solution.arm_config = manip_data.ik_solver.compute_ik(sol_cache_entry.eef_tf)
        self._call_stats[0] += 1
        return new_solution

    def can_construct_solution(self, key):
        """
            Return whether it is possible to construct a solution from the given (partially defined) key.
        """
        return len(key) >= self._hierarchy.get_minimum_depth_for_construction()

    def is_valid(self, solution):
        """
            Return whether the given PlacementSolution is valid.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
            -------
            Returns
            -------
            valid, bool
        """
        cache_entry = self._solutions_cache[solution.key]
        assert(cache_entry.solution == solution)
        self._call_stats[1] += 1
        # kinematic reachability?
        if not self._reachability_constraint.check_reachability(cache_entry):
            rospy.logdebug("Solution invalid because it's not reachable")
            return False
        # collision free?
        if not self._collision_constraint.check_collision(cache_entry):
            rospy.logdebug("Solution invalid because it's in collision")
            return False
        # next check whether the object pose is actually a stable placement
        if not self._contact_constraint.check_contacts(cache_entry):
            rospy.logdebug("Solution invalid because it's unstable")
            return False
        return True

    def get_constraint_relaxation(self, solution):
        """
            Return a value in the interval [0, 1], where 0 corresponds to invalid
            and 1 to valid. If a solution is somehow close to being valid, a value in the interval
            [0, 1] is returned. The larger the value, the closer it is to being a valid solution.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
            -------
            Returns
            -------
            val, float
        """
        cache_entry = self._solutions_cache[solution.key]
        assert(cache_entry.solution == solution)
        # self._object_data.kinbody.SetTransform(solution.obj_tf)  # TODO remove
        # if solution.arm_config is not None:  # TODO remove
        #     robot = solution.manip.GetRobot()  # TODO remove
        #     old_config = robot.GetDOFValues()  # TODO remove
        #     robot.SetDOFValues(solution.arm_config, solution.manip.GetArmIndices())  # TODO remove
        reachability_val = self._reachability_constraint.get_relaxation(cache_entry)
        contact_val = self._contact_constraint.get_relaxation(cache_entry)
        col_val = self._collision_constraint.get_relaxation(cache_entry)
        rospy.logdebug("Reachability value: %f, contact value: %f, collision value: %f" %
                       (reachability_val, contact_val, col_val))
        # if solution.arm_config is not None:  # TODO remove
        #     robot.SetDOFValues(old_config)  # TODO remove
        self._call_stats[2] += 1
        relaxation_val = (reachability_val + contact_val + col_val) / 3.0  # TODO could add weights here
        relaxation_val = relaxation_val**5
        solution.data = {'reachability': reachability_val,
                         "contact": contact_val, "col": col_val, "total": relaxation_val}
        return relaxation_val  # TODO could pass this value through a non-linear function here to make the reward sharper

    def evaluate(self, solution):
        # TODO implement me
        self._call_stats[3] += 1
        solution.objective_value = solution.obj_tf[1, 3]
        return solution.objective_value

    def get_num_construction_calls(self, b_reset=True):
        val = self._call_stats[0]
        if b_reset:
            self._call_stats[0] = 0
        return val

    def get_num_validity_calls(self, b_reset=True):
        val = self._call_stats[1]
        if b_reset:
            self._call_stats[1] = 0
        return val

    def get_num_relaxation_calls(self, b_reset=True):
        val = self._call_stats[2]
        if b_reset:
            self._call_stats[2] = 0
        return val

    def get_num_evaluate_calls(self, b_reset=True):
        val = self._call_stats[3]
        if b_reset:
            self._call_stats[3] = 0
        return val

    def _compute_object_pose(self, cache_entry):
        """
            Compute an object pose for the solution in cache_entry.
            ---------
            Arguments
            ---------
            cache_entry, SolutionCacheEntry - The following fields are required to be intialized:
                manip, region, plcmnt_orientation, so2_interval, solution
            ------------
            Side effects
            ------------
            cache_entry.solution.obj_tf, numpy array of shape (4,4) - pose of the object is stored here
            cache_entry.eef_tf, numpy array of shape (4, 4) - pose of the end-effector is stored here
        """
        manip_data = self._manip_data[cache_entry.solution.manip.GetName()]
        center_angle = (cache_entry.so2_interval[1] + cache_entry.so2_interval[0]) / 2.0  # rotation around local z axis
        # compute default object pose
        cache_entry.solution.obj_tf = np.dot(cache_entry.region.contact_tf, np.dot(tf_mod.rotation_matrix(
            center_angle, [0., 0., 1]), cache_entry.plcmnt_orientation.inv_reference_tf))
        # compute default end-effector tf
        cache_entry.eef_tf = np.dot(cache_entry.solution.obj_tf, manip_data.grasp_tf)

    def _optimize_constraints(self, cache_entry, b_optimize_obj):
        """
            Compute an object pose and arm configuration jointly by using an optimizer to minimize
            constraint violation.
            ---------
            Arguments
            ---------
            cache_entry, SolutionCacheEntry - The following fields are required to be intialized:
                manip, region, plcmnt_orientation, so2_interval, solution
            b_optimize_obj, bool - if True, also optimize objective
            ------------
            Side effects
            ------------
            cache_entry.solution.obj_tf, numpy array of shape (4, 4) - pose of the object is stored here
            cache_entry.eef_tf, numpy array of shape (4, 4) - pose of the end-effector is stored here
            cache_entry.solution.arm_config, numpy array of shape (n,) - arm configuration (n DOFs)
        """
        manip_data = self._manip_data[cache_entry.solution.manip.GetName()]
        # get ik solver
        plcmnt_ik_solver = self._get_plcmnt_ik_solver(cache_entry)
        # first compute an initial solution using trac_ik that is in the placement region
        center_tf = cache_entry.region.center_tf
        position_extents = (cache_entry.region.dimensions) / 2.0
        angle_range = cache_entry.so2_interval[1] - cache_entry.so2_interval[0]
        center_angle = (cache_entry.so2_interval[1] + cache_entry.so2_interval[0]) / 2.0
        # construct pose
        target_pose = np.dot(center_tf, tf_mod.rotation_matrix(center_angle, [0.0, 0.0, 1.0]))
        # quat = orpy.quatFromRotationMatrix(target_pose)
        # create a seed # TODO query reachability map! (for contact_tf)
        arm_config = plcmnt_ik_solver.compute_ik(target_pose, bx=position_extents[0], by=position_extents[1],
                                                 bz=position_extents[2]/4.0, brz=angle_range / 2.0)
        if arm_config is not None:
            cache_entry.solution.arm_config = arm_config
            rospy.logdebug("Trac-IK found an intial ik-solution for placement region.")
            self._jacobian_optimizer.optimize(cache_entry)
            # compute object_pose and end-effector transform TODO: do we need this here after the jacobian optimizer finished?
            with self._robot_data.robot:
                self._robot_data.robot.SetDOFValues(cache_entry.solution.arm_config, manip_data.manip.GetArmIndices())
                cache_entry.eef_tf = manip_data.manip.GetEndEffectorTransform()
                cache_entry.solution.obj_tf = np.dot(cache_entry.eef_tf, manip_data.inv_grasp_tf)
        else:
            rospy.logdebug("Trac-IK FAILED to compute initial solution. Setting default object tf.")
            self._compute_object_pose(cache_entry)

    def _get_plcmnt_ik_solver(self, cache_entry):
        """
            Return a trac_ik solver for the reference contact point given the manipulator
            and placement orientation.
            ---------
            Arguments
            ---------
            cache_entry, SolutionCacheEntry
            ------------
            Side effects
            ------------
            self._plcmnt_ik_solvers might be updated
            -------
            Returns
            -------
            Ik solver
        """
        mkey = (cache_entry.key[0], cache_entry.key[2])
        # if we already have an ik solver for the combination manip + placement orientation, return it
        if mkey in self._plcmnt_ik_solvers:
            return self._plcmnt_ik_solvers[mkey]
        else:
            raise RuntimeError("Could not find an ik solver for manipulator %s and placement orientation %s" %
                               (cache_entry.solution.manip.GetName(), mkey[1]))

    def _init_ik_solvers(self):
        for ((mid, manip), (oid, po)) in self._hierarchy.get_all_manip_orientations():
            mkey = (mid, oid)
            manip_data = self._manip_data[manip.GetName()]
            # make a custom ik solver for the placement point
            pose = np.dot(manip_data.inv_grasp_tf, po.reference_tf)
            manip.SetLocalToolTransform(pose)
            self._plcmnt_ik_solvers[mkey] = ik_module.IKSolver(
                manip, urdf_content=self._robot_data.urdf_desc, timeout=0.025)
            manip.SetLocalToolTransform(np.eye(4))

        # def optimize_constraints(self, solution):
        #     """
        #         Perform optimization on the constraint relaxation in order to make the given solution feasible.
        #         This method runs one of scipy's optimization algorithms to locally move the
        #         arm configuration in the given solution towards a feasible configuration.
        #         TODO:
        #         If there is no arm configuration, i.e. the object pose is not reachable, it first attempts to
        #         move the object towards a feasible pose.
        #     """
        #     # TODO implement optimzation!
        #     pass
        # constraints = [
        #     {
        #         'type': 'ineq',
        #         'fun': functools.partial(pose_wrapper_fn, self.collision_cost),
        #     },
        #     {
        #         'type': 'ineq',
        #         'fun': lambda x: x - search_space_bounds[:, 0]
        #     },
        #     {
        #         'type': 'ineq',
        #         'fun': lambda x: search_space_bounds[:, 1] - x
        #     }
        # ]
        # options = {
        #     'maxiter': self._parameters["max_iter"],
        #     'rhobeg': self._parameters["rhobeg"],
        #     'disp': True,
        # }
        # opt_result = scipy.optimize.minimize(functools.partial(pose_wrapper_fn, self.objective_fn, multiplier=-1.0),
        #                                      x0, method='COBYLA', constraints=constraints, options=options)
        # else:
        #     # TODO do nothing if start pose already reachable?
        #     # TODO just try simple gradient descent
        #     # grad_fn = self._reachability_constraint.get_pose_reachability_grad(cache_entry)
        #     # def region_distance(val):
        #     #     print "region_distance"
        #     #     pos = np.array([[val[0], val[1], cache_entry.region.cell_size / 2.0]])
        #     #     dist_val = cache_entry.region.local_distance_field.get_cell_values_pos(pos, b_global_frame=False)[0]
        #     #     print dist_val
        #     #     return dist_val

        #     reachability_fn, _ = self._reachability_constraint.get_pose_reachability_fn(cache_entry)
        #     # constraints = [
        #     #     {
        #     #         'type': 'eq',
        #     #         'fun': region_distance,
        #     #     },
        #     # ]
        #     # TODO delete
        #     default_obj_tf = np.dot(cache_entry.region.contact_tf, np.dot(tf_mod.rotation_matrix(
        #         center_angle, [0., 0., 1]), cache_entry.plcmnt_orientation.inv_reference_tf))
        #     self._object_data.kinbody.SetTransform(default_obj_tf)
        #     default_eef_tf = np.dot(default_obj_tf, manip_data.grasp_tf)
        #     # TODO remove
        #     handle = orpy.misc.DrawAxes(manip_data.manip.GetRobot().GetEnv(), default_eef_tf)
        #     # TODO until here
        #     start_point = np.empty(3)
        #     start_point[:2] = cache_entry.region.contact_xy
        #     start_point[2] = center_angle
        #     options = {
        #         'disp': True,
        #     }
        #     bounds = [(0.0, cache_entry.region.dimensions[0]),
        #               (0.0, cache_entry.region.dimensions[1]),
        #               (cache_entry.so2_interval[0], cache_entry.so2_interval[1])]
        #     result = scipy.optimize.fmin_tnc(reachability_fn, start_point, approx_grad=True, bounds=bounds)
        #     # result = scipy.optimize.minimize(reachability_fn, start_point, method='SLSQP', constraints=constraints,
        #                                     #  options=options, bounds=bounds)
        #     local_pose = tf_mod.rotation_matrix(result[0][2], [0, 0, 1])
        #     local_pose[:2, 3] = result[0][:2]
        #     cache_entry.solution.obj_tf = np.dot(cache_entry.region.contact_tf, np.dot(local_pose, cache_entry.plcmnt_orientation.inv_reference_tf))
        #     # compute end-effector tf
        #     cache_entry.eef_tf = np.dot(cache_entry.solution.obj_tf, manip_data.grasp_tf)
        # # TODO remove
        # self._object_data.kinbody.SetTransform(cache_entry.solution.obj_tf)
        # handle = orpy.misc.DrawAxes(manip_data.manip.GetRobot().GetEnv(), cache_entry.eef_tf)
        # print "Pose computed"

        ######### Untested code for scipy version 1.1.0  #########
        # lower_bounds = np.empty(3)
        # lower_bounds[:2] = 0.0
        # lower_bounds[2] = cache_entry.so2_interval[0]
        # upper_bounds = np.empty(3)
        # upper_bounds[:2] = cache_entry.region.dimensions
        # upper_bounds[2] = cache_entry.so2_interval[1]
        # bounds_constraint = scipy.optimize.Bounds(lower_bounds, upper_bounds, True)  # bounds should be enforced!

        # def region_distance(val):
        #     return cache_entry.local_distance_field.get_cell_values_pos(val.reshape((1, 3)))
        # in_region_constraint = scipy.optimize.NonlinearConstraint(region_distance, [0.0], [0.0])
        # reachability_fn, threshold = self._reachability_constraint.get_pose_reachability_fn(cache_entry)
        # start_point = np.empty(3)
        # start_point[:2] = cache_entry.contact_xy
        # start_point[2] = center_angle

        # def termination_criteria(val, state):
        #     return state.fun[0] <= threshold
        # result = scipy.optimize.minimize(reachability_fn, start_point, method='trust-constr',
        #                                  jac='2-point', bounds=bounds_constraint, constraints=in_region_constraint,
        #                                  callback=termination_criteria)

        ###################### Not working optimization code ###############################
        # def region_distance(val):
        #     print "region_distance"
        #     pos = np.array([[val[0], val[1], cache_entry.region.cell_size / 2.0]])
        #     dist_val = cache_entry.region.local_distance_field.get_cell_values_pos(pos, b_global_frame=False)[0]
        #     print dist_val
        #     return dist_val

        # reachability_fn, _ = self._reachability_constraint.get_pose_reachability_fn(cache_entry)
        # constraints = [
        #     {
        #         'type': 'eq',
        #         'fun': region_distance,
        #     },
        # ]
        # start_point = np.empty(3)
        # start_point[:2] = cache_entry.region.contact_xy
        # start_point[2] = center_angle
        # options = {
        #     'disp': True,
        # }
        # bounds = [(0.0, cache_entry.region.dimensions[0]),
        #           (0.0, cache_entry.region.dimensions[1]),
        #           (cache_entry.so2_interval[0], cache_entry.so2_interval[1])]
        # result = scipy.optimize.minimize(reachability_fn, start_point, method='SLSQP', constraints=constraints,
        #                                  options=options, bounds=bounds)
        # local_pose = tf_mod.rotation_matrix(result.x[2], [0, 0, 1])
        # local_pose[:2, 3] = result.x[:2]
        # cache_entry.solution.obj_tf = np.dot(cache_entry.region.base_tf, np.dot(
        #     local_pose, cache_entry.plcmnt_orientation.inv_reference_tf))
        # # compute end-effector tf
        # cache_entry.eef_tf = np.dot(cache_entry.solution.obj_tf, manip_data.grasp_tf)
