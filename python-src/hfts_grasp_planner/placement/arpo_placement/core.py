import numpy as np
import openravepy as orpy
import hfts_grasp_planner.utils as utils
import hfts_grasp_planner.placement.so2hierarchy as so2hierarchy
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
        Once a node has no subregions and is at the bottom of the SO2 hierarchy, it has no children anymore and the end of
        this composed hierarchy is reached.
        The above procedure guarantees that all leaves of this hierarchy are equal in placement region size and SO2 interval length.
    """

    def __init__(self, manipulators, regions, orientations, so2_depth, so2_branching=2):
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
                return (key[:3] + (subregion_key + (0,), so2_key + (o,)) for o in so2hierarchy.get_key_gen(key[4], self._so2_branching))
            if b_so2_leaf:
                return (key[:3] + (subregion_key + (r,), so2_key + (0,)) for r in xrange(subregion.get_num_subregions()))
            return (key[:3] + (subregion_key + (r,), (so2_key + (o,))) for r in xrange(subregion.get_num_subregions())
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
        return key[:3] + (subregion_key + (np.random.randint(subregion.get_num_subregions()),), so2_key + (np.random.randint(self._so2_branching),))

    def get_minimum_depth_for_construction(self):
        """
            Return the minimal depth, i.e. length of a key, for which it is possible
            to construct a solution.
        """
        return 3

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
    class ManipulatorData:
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
                ik_solver - IKSolver for this manipulator
                reachability_map - ReachabilityMap for this manipulator
                grasp_tf - grasp transform (eef pose in object frame)
                grasp_config, numpy array (n_h,) - hand configuration for grasp
            """
            self.manip = manip
            self.ik_solver = ik_solver
            self.reachability_map = reachability_map
            self.grasp_tf = grasp_tf
            self.grasp_config = grasp_config
            self.inv_grasp_tf = utils.inverse_transform(self.grasp_tf)

    class SolutionCacheEntry:
        """
            Cache all relevant information for a particular solution.
        """

        def __init__(self, key, solution, region, plcmnt_orientation, so2_interval):
            self.key = key  # arpo hierarchy key
            self.solution = solution  # PlacementGoal
            self.region = region  # PlacementRegion from key
            self.plcmnt_orientation = plcmnt_orientation  # PlacementOrientation from key
            self.so2_interval = so2_interval  # SO2 interval from key
            self.bkinematically_reachable = None  # store whether ik solutions exist
            self.bcollision_free = None  # store whether collision-free
            self.bbetter_objective = None  # store whether it has better objective than the current best
            self.bstable = None  # store whether pose is actually a stable placement
            self.objective_val = None  # store objective value

    def __init__(self, arpo_hierarchy, manipulator_data, target_obj, objective_fn, valid_contact_points):
        """
            Create a new ARPORobotBridge
            ---------
            Arguments
            ---------
            arpo_hierarchy, ARPOHierarchy - arpo hierarchy to create solutions for
            manipulator_data, dict - mapping from manipulator names to ManipulatorData
            target_obj, OpenRAVE kinbody - object that is to be placed
            objective_fn, ??? - TODO
            valid_contact_points, VoxelGrid with bool values - stores for the scene where object
                contact points may be located
        """
        self._hierarchy = arpo_hierarchy
        self._manip_data = manipulator_data
        self._objective_fn = objective_fn
        self._target_obj = target_obj
        self._valid_contact_points = valid_contact_points
        self._solutions_cache = []  # array of SolutionCacheEntry objects

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
        region = arpo_info[1]
        po = arpo_info[2]
        so2_interval = np.array([0, 2.0 * np.pi])
        if len(arpo_info) == 5:
            region = arpo_info[3]
            so2_interval = arpo_info[4]
        # compute object pose
        obj_tf = np.dot(region.contact_tf, po.inv_reference_tf)
        manip_data = self._manip_data[manip.GetName()]
        # TODO REMOVE
        import openravepy as orpy
        env = manip.GetRobot().GetEnv()
        body = env.GetKinBody('crayola')
        body.SetTransform(obj_tf)
        # end-effector tf
        eef_tf = np.dot(obj_tf, manip_data.grasp_tf)
        handle = orpy.misc.DrawAxes(env, eef_tf)  # TODO remove
        # TODO check reachability map on whether there is point in calling IK solver?
        # TODO seed?
        config = manip_data.ik_solver.compute_ik(eef_tf)
        # TODO optimize constraints, optimize objective
        new_solution = placement_interfaces.PlacementGoalSampler.PlacementGoal(
            # TODO real objective value
            manip=manip, arm_config=config, obj_tf=obj_tf, key=len(self._solutions_cache), objective_value=0.0,
            grasp_tf=manip_data.grasp_tf, grasp_config=manip_data.grasp_config)
        self._solutions_cache.append(ARPORobotBridge.SolutionCacheEntry(key, new_solution, region, po, so2_interval))
        return new_solution

    def is_valid(self, solution):
        """
            Return whether the given PlacementSolution is valid.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
        """
        solution_cache_entry = self._solutions_cache[solution.key]
        assert(solution_cache_entry.solution == solution)
        # kinematic reachability?
        config = solution.arm_config
        if config is None:
            solution_cache_entry.bkinematically_reachable = False
            return False
        # collision free?
        manip_data = self._manip_data[solution.manip.GetName()]
        robot = solution.manip.GetRobot()
        with robot:
            with orpy.KinBodyStateSaver(self._target_obj):
                # grab object (sets active manipulator for us)
                utils.set_grasp(manip_data.manip, self._target_obj, manip_data.inv_grasp_tf, manip_data.grasp_config)
                robot.SetDOFValues(config, manip_data.manip.GetArmIndices())
                env = robot.GetEnv()
                solution_cache_entry.bcollision_free = not env.CheckCollision(robot) and not robot.CheckSelfCollision()
                robot.Release(self._target_obj)
        if not solution_cache_entry.bcollision_free:
            return False
        # next check whether the object pose is actually a stable placement
        # solution_cache_entry.bstable = True
        solution_cache_entry.bstable = self._check_stability(solution_cache_entry)
        # TODO value must be better than some previous value
        return solution_cache_entry.bstable

    def get_constraint_relaxation(self, solution):
        # TODO implement me
        raise RuntimeError("Not yet implemented")

    def evaluate(self, solution):
        # TODO implement me
        solution.objective_value = solution.obj_tf[1, 3]
        return solution.objective_value

    def _check_stability(self, cache_entry):
        """
            Check whether the solution stored in cache entry represents a stable placement.
        """
        obj_tf = cache_entry.solution.obj_tf
        po = cache_entry.plcmnt_orientation
        contact_points = np.dot(po.placement_face[1:], obj_tf[:3, :3].transpose()) + obj_tf[:3, 3]
        # TODO delete this (debug output)
        # self._target_obj.SetTransform(obj_tf)
        # handles = []
        # or_env = self._target_obj.GetEnv()
        # _, grid_indices, valid_mask = self._valid_contact_points.map_to_grid_batch(contact_points, index_type=int)
        # if valid_mask.any():
        #     positions = self._valid_contact_points.get_cell_positions(grid_indices)
        #     extents = np.array(3 * [self._valid_contact_points.get_cell_size() / 2.0])
        #     for pos in positions:
        #         handles.append(or_env.drawbox(pos, extents, np.array([0, 1.0, 0.0, 0.3])))
        # return False
        # TODO until here
        values = self._valid_contact_points.get_cell_values_pos(contact_points)
        none_values = values == None
        if none_values.any():
            return False
        return values.all()  # for heuristic return ratio of true vs none
