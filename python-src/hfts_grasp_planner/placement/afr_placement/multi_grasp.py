import rospy
import scipy
import yaml
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
import hfts_grasp_planner.dmg.dmg_class as dmg_module
import hfts_grasp_planner.placement.afr_placement.core as afr_core

"""
    This module defines an AFRRobotBridge that considers multiple grasps on the target object.
    The considered grasps are obtained from the Dexterous Manipulation Graph of the object
    and a given initial grasp.
"""


class DMGGraspSet(object):
    """
        Stores a set of grasps that are reachable through in-hand manipulation from some initial grasp
        using the Dexterous Manipulation Graph.
    """
    class Grasp(object):
        def __init__(self, gid, tf, config, dmg_info):
            """
                Create a new grasp.
                ---------
                Arguments
                ---------
                gid, int - unique identifier of the grasp
                tf, np.array of shape (4, 4) - transformation eTo from object frame into eef-frame
                config, np.array of shape (q,) - gripper DOF configuration (it should be q = 1)
                dmg_info, tuple (node, angle, onode, oangle) - dmg information of this grasp
            """
            self.gid = gid
            self.eTo = tf
            self.oTe = utils.inverse_transform(tf)
            self.config = config
            self.dmg_info = dmg_info

    def __init__(self, manip, target_obj, gripper_file, target_obj_file, finger_info_file, dmg, rot_weight=0.1):
        """
            Create a new DMGGraspSet for the given manipulator on the given target object.
            The grasp set consists of the grasps that are reachable through in-hand manipulation
            starting from the current grasp. The current grasp is retrieved from the current relative transformation
            between the manipulator's end-effector and the target object.
            ---------
            Arguments
            ---------
            manip, OpenRAVE manipulator - the manipulator to create the set for
            target_obj, OpenRAVE Kinbody - the target object to create the set for
            gripper_file, string - path to a kinbody/robot xml file representing only the gripper
                (used internally for collision checks)
            target_obj_file, string - path to kinbody xml file representing the target object
            finger_info_file, string - path to finger information file
            dmg, DexterousManipulationGraph - the DMG created for the target object
            rot_weight, float - scaling factor of angles in radians for distance computation
        """
        self._manip = manip
        self._target_obj = target_obj
        self._dmg = dmg
        self._rot_weight = rot_weight
        # set up internal or environment
        self._my_env = orpy.Environment()
        self._my_env.Load(gripper_file)
        self._my_env.Load(target_obj_file)
        # load finger info
        finger_tfs = DMGGraspSet.load_finger_tf(finger_info_file)
        reference_link, rTf = finger_tfs[manip.GetName()]
        # obtain eTf
        wTr = manip.GetRobot().GetLink(reference_link).GetTransform()
        wTe = manip.GetEndEffectorTransform()
        self._eTf = np.dot(np.dot(utils.inverse_transform(wTe), wTr), rTf)
        # compute grasp set
        self._grasps = None
        self._compute_grasps()

    @staticmethod
    def load_finger_tf(filename):
        with open(filename, 'r') as info_file:
            finger_info = yaml.load(info_file)
        finger_tfs = {}
        for key, info_val in finger_info.iteritems():
            link_name = info_val['reference_link']
            rot = np.array(info_val['rotation'])
            trans = np.array(info_val['translation'])
            tf = orpy.matrixFromQuat(rot)
            tf[:3, 3] = trans
            finger_tfs[key] = (link_name, tf)
        return finger_tfs

    def _construct_grasp(self, node_a, angle_a, node_b, angle_b, gid):
        """
            Construct a grasp from the given DMG nodes and angles.
        """
        oTf = self._dmg.get_finger_tf(node_a, angle_a)
        eTo = np.dot(self._eTf, utils.inverse_transform(oTf))
        finger_dist = self._dmg.get_finger_distance(node_a, node_b)
        return DMGGraspSet.Grasp(gid, eTo, np.array([finger_dist / 2.0]), (node_a, angle_a, node_b, angle_b))

    def _get_adjacent_grasps(self, dmg_info):
        """
            Return a generator for adjacent grasps.
            ---------
            Arguments
            ---------
            dmg_info, tuple - (node_key, angle, onode_key, oangle)
            -------
            Returns
            -------
            list of dmg_info tuples of adjacent grasps
            list of edge costs
        """
        neighbor_grasps = []
        edge_costs = []
        node, angle, onode, oangle = dmg_info
        node_pos = self._dmg.get_position(node)
        ocomp = self._dmg.get_component(onode)
        neighbor_nodes = self._dmg.get_neighbors(node)
        for neigh in neighbor_nodes:
            # Two grasps are translational adjacent, if
            #  1. nodes are adjacent (iteration over neighbors solves this)
            #  2. angle is also valid in neighboring node
            #  3. opposite nodes are adjacent
            #  4. opposite angle is valid in opposite node neighbor
            # 2
            if not self._dmg.is_valid_angle(node, angle):
                continue
            # 3
            oneigh = self._dmg.get_opposite_node(neigh, comp=ocomp)
            if oneigh is None:
                continue
            if oneigh not in self._dmg.get_neighbors(onode):
                continue
            # 4
            if not self._dmg.is_valid_angle(oneigh, oangle):
                continue
            neighbor_grasps.append((neigh, angle, oneigh, oangle))
            edge_costs.append(np.linalg.norm(self._dmg.get_position(neigh) - node_pos))
        # Two grasps are rotationally adjacent, if
        #  1. angles are adjacent on grid
        #  2. opposite angle is within valid range of opposite node
        neighbor_angles = self._dmg.get_neighbor_angles(node, angle)
        for nangle in neighbor_angles:
            noangle = self._dmg.get_opposing_angle(node, nangle, onode)
            if self._dmg.is_valid_angle(onode, noangle):
                neighbor_grasps.append((node, nangle, oneigh, noangle))
                edge_costs.append(self._dmg.get_angular_resolution() / 180.0 * np.pi * self._rot_weight)
        return neighbor_grasps, edge_costs

    def _compute_grasps(self):
        """
            Explore the DMG to compute grasps than can be reached from the current grasp.
            Stores the resulting grasps in self._grasps
        """
        self._grasps = []
        # set initial grasp first. This creates two grasps, the observed one and the closest one in the DMG
        robot = self._manip.GetRobot()
        with robot:
            wTe = self._manip.GetEndEffectorTransform()
            wTo = self._target_obj.GetTransform()
            eTo = np.dot(utils.inverse_transform(wTe), wTo)
            config = robot.GetDOFValues(self._manip.GetGripperIndices())
            # add the observed initial grasp
            observed_initial_grasp = DMGGraspSet.Grasp(len(self._grasps), eTo, config, None)
            self._grasps.append(observed_initial_grasp)
            # now obtain the closest dmg grasp
            # for this, first compute local position of finger
            oTf = np.dot(utils.inverse_transform(eTo), self._eTf)
            start_node, start_angle, bvalid = self._dmg.get_closest_node_angle(oTf)
            if not bvalid:
                rospy.logwarn("Initial grasp is not within the valid angular range of the closest DMG node!")
            # TODO this may be specific to Yumi
            finger_dist = 2.0 * config[0]
            onode = self._dmg.get_opposite_node(start_node, finger_dist)
            oangle = self._dmg.get_opposing_angle(start_node, start_angle, onode)
            dmg_initial_grasp = self._construct_grasp(start_node, start_angle, onode, oangle, len(self._grasps))
            self._grasps.append(dmg_initial_grasp)
        # TODO explore all reachable grasps from here
        # TODO implement dijkstra
        open_list = []
        # TODO do collision checks here


class MultiGraspAFRRobotBridge(placement_interfaces.PlacementGoalConstructor,
                               placement_interfaces.PlacementValidator,
                               placement_interfaces.PlacementObjective):
    """
        A MultiGraspAFRRobotBridge serves as interface for a placement planner
        operating on the AFRHierarchy. In contrast to the AFRRobotBridge, the MultiGraspAFRRobotBridge
        considers multiple grasps when constructing placement solutions.
        ----------
        Parameters
        ----------
        relaxation_type, string - can either be 'binary', 'sub_binary', or 'continuous'
            Binary relaxation: No constraint relaxation at all, the function get_constraint_relaxation(sol)
                returns 0 if any constraint is violated by sol, else 1
            Sub-binary relaxation(default): For each constraint, a binary value is computed indicating whether the respective
                constraint is violated (0) or not (1). The overall relaxation is then a normalized weighted sum of
                these binary values, if the in-region constraint is fulfilled, else it is 0
            Continuous: For each constraint, a continuous relaxation function is used to compute to what degree the
                constraint is violated. The returned relaxation is the normalized weighted sum of these.
        joint_limit_margin, float - minimal distance to joint limits (must be >= 0.0)
    """
    class ManipulatorData(object):
        """
            Struct (named tuple) that stores manipulator data.
        """

        def __init__(self, manip, ik_solver, grasp_set):
            """
                Create a new instance of manipulator data.
                ---------
                Arguments
                ---------
                manip - OpenRAVE manipulator
                ik_solver - ik_module.IKSolver, IK solver for end-effector poses
                grasp_set - DMGGraspSet, DMG grasp set for this manipulator
            """
            self.manip = manip
            self.manip_links = utils.get_manipulator_links(manip)
            self.ik_solver = ik_solver
            self.lower_limits, self.upper_limits = self.manip.GetRobot().GetDOFLimits(manip.GetArmIndices())
            self.grasp_set = grasp_set

    # class SolutionCacheEntry(object):
    #     """
    #         Cache all relevant information for a particular solution.
    #     """

    #     def __init__(self, key, region, plcmnt_orientation, so2_interval, solution):
    #         self.key = key  # afr hierarchy key (the one it was created for)
    #         # leaf afr hierarchy key (the lowest element in the tree this solution could come from)
    #         self.leaf_key = None
    #         self.solution = solution  # PlacementGoal
    #         self.region = region  # PlacementRegion from key
    #         self.plcmnt_orientation = plcmnt_orientation  # PlacementOrientation from key
    #         self.so2_interval = so2_interval  # SO2 interval from key
    #         self.eef_tf = None  # store end-effector transform
    #         self.bkinematically_reachable = None  # store whether ik solutions exist
    #         self.barm_collision_free = None  # store whether the arm is collision-free
    #         self.bobj_collision_free = None  # store whether the object is collision-free
    #         self.bbetter_objective = None  # store whether it has better objective than the current best
    #         self.bstable = None  # store whether pose is actually a stable placement
    #         self.objective_val = None  # store objective value
    #         # the following elements are used for gradient-based optimization
    #         self.jacobian = None  # Jacobian for active manipulator (and object reference pos)
    #         self.region_pose = None  # pose of the object relative to region frame
    #         self.region_state = None  # (x, y, theta) within region

    #     def copy(self):
    #         new_entry = AFRRobotBridge.SolutionCacheEntry(self.key, self.region, self.plcmnt_orientation,
    #                                                       np.array(self.so2_interval), self.solution.copy())
    #         new_entry.eef_tf = None if self.eef_tf is None else np.array(new_entry.eef_tf)
    #         new_entry.bkinematically_reachable = self.bkinematically_reachable
    #         new_entry.barm_collision_free = self.barm_collision_free
    #         new_entry.bobj_collision_free = self.bobj_collision_free
    #         new_entry.bbetter_objective = self.bbetter_objective
    #         new_entry.bstable = self.bstable
    #         new_entry.objective_val = self.objective_val
    #         new_entry.jacobian = np.array(self.jacobian) if self.jacobian is not None else None
    #         new_entry.region_pose = np.array(self.region_pose) if self.region_pose is not None else None
    #         new_entry.region_state = self.region_state
    #         return new_entry
    class JacobianOptimizer(object):
        def __init__(self, contact_constraint, collision_constraint, obj_fn, robot_data, object_data,
                     grad_epsilon=0.01, step_size=0.01, max_iterations=100, joint_limit_margin=1e-4):
            """
                Create a new JacobianOptimizer.
                ---------
                Arguments
                ---------
                contact_constraint, ContactConstraint
                collision_constraint, CollisionConstraint
                obj_fn, ObjectiveFunction TODO
                robot_data, RobotData
                object_data, ObjectData
                grad_epsilon, float - minimal magnitude of cspace gradient
                step_size, float - multiplier for update step
                max_iterations, int - maximal number of iterations
                joint_limit_margin, float - minimal margin to joint limits (>= 0)
            """
            self.contact_constraint = contact_constraint
            self.collision_constraint = collision_constraint
            self.obj_fn = obj_fn
            self.robot_data = robot_data
            self.object_data = object_data
            self.manip_data = robot_data.manip_data
            self.robot = robot_data.robot
            self.grad_epsilon = grad_epsilon  # minimal magnitude of cspace gradient
            self.step_size = step_size  # multiplier for update step
            self.max_iterations = max_iterations  # maximal number of iterations
            self.damping_matrix = np.diag([0.9, 0.9, 1.0, 1.0, 1.0, 0.8])  # damping matrix for nullspace projection
            self.joint_limit_margin = joint_limit_margin  # minimal margin to joint limits
            self._last_obj_value = None  # objective value of previous iteration

        def locally_maximize(self, cache_entry):
            """
                Locally maximize the objective.
                ---------
                Arguments
                ---------
                cache_entry, SolutionCacheEntry
                ------------
                Side effects
                ------------
                cache_entry.solution.arm_config is set to the improved arm configuration
                cache_entry.solution.obj_tf is set to the improved object pose
                TODO
                -------
                Returns
                -------
                arm_configs, list of np.array - list of arm configurations that describe a path from
                    cache_entry.arm_config to arm_configs[-1], where arm_configs[-1] achieves the locally
                    maximal objective that can be reached by following the gradient of the objective
                    from cache_entry.solution.arm_config. The returned list does not contain the initial
                    cache_entry.solution.arm_config.
            """
            self._last_obj_value = cache_entry.solution.objective_value
            manip_data = self.manip_data[cache_entry.solution.manip.GetName()]
            lower, upper = manip_data.lower_limits + self.joint_limit_margin, manip_data.upper_limits - self.joint_limit_margin
            manip = manip_data.manip
            arm_configs = []
            with self.robot:
                with self.object_data.kinbody:
                    # TODO update to use grasp from solution
                    utils.set_grasp(manip, self.object_data.kinbody,
                                    manip_data.inv_grasp_tf, manip_data.grasp_config)
                    reference_pose = np.dot(manip_data.inv_grasp_tf, cache_entry.plcmnt_orientation.reference_tf)
                    manip.SetLocalToolTransform(reference_pose)
                    self.robot.SetActiveDOFs(manip.GetArmIndices())
                    # init jacobian descent
                    q_current = cache_entry.solution.arm_config
                    # iterate as long as the gradient is not None, zero or we are in collision or out of joint limits
                    # while q_grad is not None and grad_norm > self.epsilon and b_in_limits:
                    for _ in xrange(self.max_iterations):
                        in_limits = (q_current >= lower).all() and (q_current <= upper).all()
                        if in_limits:
                            q_grad = self._compute_gradient(cache_entry, q_current, manip_data)
                            grad_norm = np.linalg.norm(q_grad) if q_grad is not None else 0.0
                            if q_grad is None or grad_norm < self.grad_epsilon:
                                break
                            arm_configs.append(np.array(q_current))
                            q_current -= self.step_size * q_grad / grad_norm  # update q_current
                        else:
                            break
                    manip.SetLocalToolTransform(np.eye(4))
                    manip.GetRobot().Release(self.object_data.kinbody)
                    if len(arm_configs) > 1:  # first element is start configuration
                        self._set_cache_entry_values(cache_entry, arm_configs[-1], manip_data)
                        return arm_configs[1:]
                    # we failed in the first iteration, just set it back to what we started from
                    self._set_cache_entry_values(cache_entry, cache_entry.solution.arm_config, manip_data)
                    return []  # else return empty array

        def _set_cache_entry_values(self, cache_entry, q, manip_data):
            # TODO update using grasp from dmg
            manip = manip_data.manip
            self.robot.SetActiveDOFValues(q)
            cache_entry.solution.arm_config = q
            cache_entry.solution.obj_tf = np.matmul(manip.GetEndEffector().GetTransform(),
                                                    manip_data.inv_grasp_tf)
            ref_pose = manip.GetEndEffectorTransform()  # this takes the local tool transform into account
            # TODO we are not constraining the pose to stay in the region here, so technically we would need to compute
            # TODO which region we are actually in now (we might transition into a neighboring region)
            cache_entry.region_pose = np.matmul(utils.inverse_transform(cache_entry.region.base_tf), ref_pose)
            ex, ey, ez = tf_mod.euler_from_matrix(cache_entry.region_pose)
            theta = utils.normalize_radian(ez)
            cache_entry.region_state = (cache_entry.region_pose[0, 3], cache_entry.region_pose[1, 3], theta)
            cache_entry.solution.objective_value = self.obj_fn(cache_entry.solution.obj_tf)

        def _compute_gradient(self, cache_entry, q_current, manip_data):
            """
                Compute gradient for the current configuration.
                -------
                Returns
                -------
                qgrad, numpy array of shape (manip.GetArmDOF,) - gradient of po + collision constraints
                    w.r.t arm configurations. The returned gradient is None, if stability constraint
                    is violated, we descreased objective value, or we encountered a singularity.
            """
            manip = manip_data.manip
            self.robot.SetActiveDOFValues(q_current)
            # save last objective value from previous iteration
            self._last_obj_value = cache_entry.solution.objective_value
            # update cache_entry and solution to reflect q_current
            self._set_cache_entry_values(cache_entry, q_current, manip_data)
            # compute jacobian
            jacobian = np.empty((6, manip.GetArmDOF()))
            jacobian[:3] = manip.CalculateJacobian()
            jacobian[3:] = manip.CalculateAngularVelocityJacobian()
            cache_entry.jacobian = jacobian
            # compute pseudo inverse
            inv_jac, rank = utils.compute_pseudo_inverse_rank(jacobian)
            if rank < 6:  # if we are in a singularity, just return None
                return None
            # get pose of placement reference point
            ref_pose = manip.GetEndEffectorTransform()  # this takes the local tool transform into account
            # Compute gradient w.r.t. constraints
            # ------ 1. Stability - all contact points need to be in a placement region (any)
            value, cart_grad_c = self.contact_constraint.compute_cart_gradient(cache_entry, ref_pose)
            if value > 0.0:
                return None
            # ------ 2. Collision constraint - object must not be in collision
            # can not trust collision value properly, check for collision
            in_collision = self.robot.GetEnv().CheckCollision(self.robot) or self.robot.CheckSelfCollision()
            if in_collision:
                return None
            # _, cart_grad_col = self.collision_constraint.get_cart_obj_collision_gradient(cache_entry)
            # ------ 3. Objective Improvement constraint - objective must be an improvement
            if cache_entry.solution.objective_value < self._last_obj_value:
                return None
            cart_grad_xi = -self.obj_fn.get_gradient(cache_entry.solution.obj_tf, cache_entry.region_state,
                                                     cache_entry.plcmnt_orientation.inv_reference_tf)
            # ------ 4. Translate cartesian gradients into c-space gradients
            cart_grad = cart_grad_c + cart_grad_xi  # + cart_grad_col
            extended_cart = np.zeros(6)
            extended_cart[:2] = cart_grad[:2]
            extended_cart[5] = cart_grad[2]
            qgrad = np.matmul(inv_jac, extended_cart)
            # ------ 5. Arm collision constraint - arm must not be in collision
            _, col_grad = self.collision_constraint.get_chomps_collision_gradient(cache_entry, q_current)
            col_grad[:] = np.matmul((np.eye(col_grad.shape[0]) -
                                     np.matmul(inv_jac, np.matmul(self.damping_matrix, jacobian))), col_grad)
            qgrad += col_grad
            # remove any motion that changes the base orientation/z height of the object
            jacobian[[0, 1, 5], :] = 0.0  # motion in x, y, ez is allowed
            qgrad[:] = np.matmul((np.eye(qgrad.shape[0]) - np.matmul(inv_jac, jacobian)), qgrad)
            return qgrad

    def __init__(self, afr_hierarchy, robot_data, object_data,
                 objective_fn, global_region_info, scene_sdf,
                 parameters):
        """
            Create a new MultiGraspAFRRobotBridge
            ---------
            Arguments
            ---------
            afr_hierarchy, AFRHierarchy - afr hierarchy to create solutions for
            robot_data, RobotData - struct that stores robot information including ManipulatorData for each manipulator
            object_data, ObjectData - struct that stores object information
            objective_fn, ??? - TODO
            global_region_info, (VoxelGrid, VectorGrid) - stores for the full planning scene distances to where
                contact points may be located, as well as gradients
            parameters, dict - dictionary with parameters. See class description.
        """
        self._hierarchy = afr_hierarchy
        self._robot_data = robot_data
        self._manip_data = robot_data.manip_data  # shortcut to manipulator data
        self._objective_fn = objective_fn
        self._object_data = object_data
        self._contact_constraint = afr_core.AFRRobotBridge.ContactConstraint(global_region_info)
        self._collision_constraint = afr_core.AFRRobotBridge.CollisionConstraint(object_data, robot_data, scene_sdf)
        self._reachability_constraint = afr_core.AFRRobotBridge.ReachabilityConstraint(robot_data, True)
        self._objective_constraint = afr_core.AFRRobotBridge.ObjectiveImprovementConstraint(
            objective_fn, parameters['eps_xi'])
        self._solutions_cache = []  # array of SolutionCacheEntry objects
        self._call_stats = np.array([0, 0, 0, 0])  # num sol constructions, is_valid, get_relaxation, evaluate
        self._plcmnt_ik_solvers = {}  # maps (manip_id, placement_orientation_id) to a trac_ik solver
        self._init_ik_solvers()
        # TODO update jacobian optimizer as needed
        self._jacobian_optimizer = MultiGraspAFRRobotBridge.JacobianOptimizer(self._contact_constraint,
                                                                              self._collision_constraint,
                                                                              objective_fn, self._robot_data,
                                                                              self._object_data,
                                                                              joint_limit_margin=parameters["joint_limit_margin"])
        self._parameters = parameters
        self._use_jacobian_proj = parameters["proj_type"] == "jac"

    def construct_solution(self, key, b_optimize_constraints=False):
        """
            Construct a new PlacementSolution from a hierarchy key.
            ---------
            Arguments
            ---------
            key, object - a key object that identifies a node in a PlacementHierarchy
            b_optimize_constraints, bool - if True, the solution constructor may put additional computational
                effort into computing a valid solution, e.g. some optimization of a constraint relaxation
            -------
            Returns
            -------
            PlacementSolution sol, a placement solution for the given key
        """
        if len(key) < self._hierarchy.get_minimum_depth_for_construction():
            raise ValueError("Could not construct solution for the given key: " + str(key) +
                             " This key is describing a node too high up in the hierarchy")
        afr_info = self._hierarchy.get_afr_information(key)
        assert(len(afr_info) >= 3)
        manip = afr_info[0]
        manip_data = self._manip_data[manip.GetName()]
        po = afr_info[1]
        region = afr_info[2]
        so2_interval = np.array([0, 2.0 * np.pi])
        if len(afr_info) == 5:
            region = afr_info[3]
            so2_interval = afr_info[4]
        # construct a solution without valid values yet
        new_solution = placement_interfaces.PlacementGoalSampler.PlacementGoal(
            manip=manip, arm_config=None, obj_tf=None, key=len(self._solutions_cache), objective_value=None,
            grasp_tf=manip_data.grasp_tf, grasp_config=manip_data.grasp_config, grasp_id=0)
        # create a cache entry for this solution
        sol_cache_entry = afr_core.AFRRobotBridge.SolutionCacheEntry(
            key=key, region=region, plcmnt_orientation=po, so2_interval=so2_interval, solution=new_solution)
        self._solutions_cache.append(sol_cache_entry)
        if b_optimize_constraints:
            # TODO use grasp cache in this case
            pass
        else:
            # TODO in this case simply sample a random grasp all the time
            # compute object and end-effector pose
            self._compute_object_pose(sol_cache_entry, b_random=True)
            # compute arm configuration
            sol_cache_entry.solution.arm_config = manip_data.ik_solver.compute_ik(sol_cache_entry.eef_tf,
                                                                                  joint_limit_margin=self._parameters['joint_limit_margin'])
        self._call_stats[0] += 1
        new_solution.sample_num = self._call_stats[0]
        return new_solution

    def can_construct_solution(self, key):
        """
            Return whether it is possible to construct a solution from the given (partially defined) key.
        """
        return len(key) >= self._hierarchy.get_minimum_depth_for_construction()

    def get_leaf_key(self, solution):
        """
            Return the key of the deepest hierarchy node (i.e. the leaf) that the given solution
            can belong to.
            ---------
            Arguments
            ---------
            solution, PlacementGoal - a solution constructed by this goal constructor.
            -------
            Returns
            -------
            key, object - a key object that identifies a node in a PlacementHierarchy
        """
        cache_entry = self._solutions_cache[solution.key]
        base_key = cache_entry.key
        reference_point_pose = np.dot(solution.obj_tf, cache_entry.plcmnt_orientation.reference_tf)
        cache_entry.leaf_key = self._hierarchy.get_leaf_key(
            base_key, reference_point_pose[:3, 3], cache_entry.region_state[2])
        if cache_entry.leaf_key is None:
            rospy.logwarn("Could not compute leaf key for solution. This means that the solution is out of bounds.")
            cache_entry.leaf_key = base_key
        return cache_entry.leaf_key

    def locally_improve(self, solution):
        """
            Search for a new placement that maximizes the objective locally around solution such that
            there exists a simple collision-free path from solution to the new solution.
            By simple collision-free path it is meant that this function is only using
            a local path planner rather than a global path planner (such as straight line motions).
            ---------
            Arguments
            ---------
            solution, PlacementGoal - a valid PlacementGoal
            -------
            Returns
            -------
            new_solution, PlacementGoal - the newly reached goal
            approach_path, list of np.array - arm configurations describing a path from solution to new_solution
        """
        cache_entry = self._solutions_cache[solution.key]
        cache_entry = cache_entry.copy()
        arm_configs = self._jacobian_optimizer.locally_maximize(cache_entry)
        if len(arm_configs) > 0:
            # The local optimizer was succesful at improving the solution a bit
            # add new solution to cache
            cache_entry.solution.key = len(self._solutions_cache)
            self._solutions_cache.append(cache_entry)
            # figure out what part of the hierarchy the new solution lies in
            reference_point_pose = np.dot(cache_entry.solution.obj_tf, cache_entry.plcmnt_orientation.reference_tf)
            # the jacobian optimizer may have moved the solution to a different region
            # TODO this should maybe be done within jacobian optimizer
            region_id = self._hierarchy.get_region(reference_point_pose[:3, 3])
            cache_entry.key = (cache_entry.key[0], cache_entry.key[1], region_id)
            if region_id is None:  # TODO this is a bug
                return None, []
            return cache_entry.solution, arm_configs
        return None, []

    def set_minimal_objective(self, val):
        """
            Sets the minimal objective that a placement needs to achieve in order to be considered valid.
            ---------
            Arguments
            ---------
            val, float - minimal objective value
        """
        self._objective_constraint.best_value = val

    def is_valid(self, solution, b_improve_objective, b_lazy=True):
        """
            Return whether the given PlacementSolution is valid.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
            b_improve_objective, bool - If True, the solution has to be better than the current minimal objective.
            b_lazy, bool - If True, only checks for validity until one constraint is violated and returns,
                else all constraints are evaluated (and saved in cache_entry)
            -------
            Returns
            -------
            valid, bool
        """
        cache_entry = self._solutions_cache[solution.key]
        assert(cache_entry.solution == solution)
        self._call_stats[1] += 1
        # kinematic reachability?
        if not self._reachability_constraint.check_reachability(cache_entry) and b_lazy:
            # rospy.logdebug("Solution invalid because it's not reachable")
            return False
        # collision free?
        if not self._collision_constraint.check_collision(cache_entry) and b_lazy:
            # rospy.logdebug("Solution invalid because it's in collision")
            return False
        # next check whether the object pose is actually a stable placement
        if not self._contact_constraint.check_contacts(cache_entry) and b_lazy:
            # rospy.logdebug("Solution invalid because it's unstable")
            return False
        # finally check whether the objective is an improvement
        if b_improve_objective:
            if cache_entry.objective_val is None:  # objective has not been evaluated yet
                self.evaluate(solution)
            return self._objective_constraint.check_objective_improvement(cache_entry)
        return True

    def get_constraint_relaxation(self, solution, b_incl_obj=False, b_obj_normalizer=False):
        """
            Return a relaxation value between [0, 1] that is 0
            if the solution is invalid and goes towards 1 the closer the solution is to
            something valid.
            The constraint relexation may include the objective-improvement constraint, or not.
            This is determined by setting b_incl_obj. If it is True, the returned relaxation
            includes it, else not. In any case, to ensure the returned value lies within [0, 1], it is internally
            normalized. If b_incl_obj=False, by setting b_obj_norrmalizer the normalizer can be forced
            to be the same as if b_incl_obj was True. Note that this implies that returned values are in some range [0, c]
            with c < 1.
            ---------
            Arguments
            ---------
            solution, PlacementGoal - solution to evaluate
            -------
            Returns
            -------
            val, float - relaxation value in [0, 1], or [0, c] with c < 1 if b_incl_obj=False, and b_obj_normalizer=True
        """
        # first compute normalizer
        normalizer = self._parameters["weight_arm_col"] + \
            self._parameters["weight_obj_col"] + self._parameters["weight_contact"]
        if b_incl_obj or b_obj_normalizer:
            normalizer += self._parameters["weight_objective"]
        # check if have binary relaxation
        if self._parameters["relaxation_type"] == "binary":
            if not b_obj_normalizer or b_incl_obj:
                return float(self.is_valid(solution, b_incl_obj))
            else:
                val = float(self.is_valid(solution, False))
                return val * (normalizer - self._parameters["weight_objective"])
        # sub-binary or continuous
        cache_entry = self._solutions_cache[solution.key]
        assert(cache_entry.solution == solution)
        self._call_stats[2] += 1
        self.is_valid(solution, b_incl_obj, b_lazy=False)  # ensure all validity flags are set
        val = 0.0
        if not cache_entry.bkinematically_reachable:  # not a useful solution without an arm configuration
            return 0.0
        # compute binary or continuous sub relaxatoins
        if self._parameters["relaxation_type"] == "sub-binary":
            val += self._parameters["weight_arm_col"] * float(cache_entry.barm_collision_free)
            val += self._parameters["weight_obj_col"] * float(cache_entry.bobj_collision_free)
            val += self._parameters["weight_contact"] * float(cache_entry.bstable)
            if b_incl_obj:
                valid_sol = float(cache_entry.bobj_collision_free) * float(cache_entry.bstable)
                val += valid_sol * self._parameters["weight_objective"] * float(cache_entry.bbetter_objective)
        else:  # compute continuous relaxation
            assert(self._parameters["relaxation_type"] == "continuous")
            contact_val = self._contact_constraint.get_relaxation(cache_entry)
            arm_col_val, obj_col_val = self._collision_constraint.get_relaxation(cache_entry)
            # rospy.logdebug("contact value: %f, arm-collision value: %f, obj_collision value: %f" %
            #                (contact_val, arm_col_val, obj_col_val))
            val += self._parameters["weight_arm_col"] * arm_col_val
            val += self._parameters["weight_obj_col"] * obj_col_val
            val += self._parameters["weight_contact"] * contact_val
            solution.data = {'arm_col': arm_col_val, "obj_col": obj_col_val, "contact": contact_val, "total": val}
            if b_incl_obj:
                valid_sol = float(cache_entry.bobj_collision_free) * float(cache_entry.bstable)
                val += valid_sol * self._parameters["weight_objective"] * \
                    self._objective_constraint.get_relaxation(cache_entry)
        return val / normalizer

    def get_constraint_weights(self):
        return np.array((self._parameters["weight_arm_col"], self._parameters["weight_obj_col"],
                         self._parameters["weight_contact"], self._parameters["weight_objective"]))

    def evaluate(self, solution):
        """
            Evaluate the given solution.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
                solution.obj_tf must not be None
            ------------
            Side effects
            ------------
            solution.objective_value will be set to the solution's objective
            -------
            Returns
            -------
            objective_value, float
        """
        cache_entry = self._solutions_cache[solution.key]
        self._call_stats[3] += 1
        if cache_entry.objective_val is None:
            cache_entry.objective_val = self._objective_fn(solution.obj_tf)
        solution.objective_value = cache_entry.objective_val
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

    def _compute_object_pose(self, cache_entry, b_random=False):
        """
            Compute an object pose for the solution in cache_entry.
            ---------
            Arguments
            ---------
            cache_entry, SolutionCacheEntry - The following fields are required to be intialized:
                manip, region, plcmnt_orientation, so2_interval, solution
            b_random, bool - if True, randomly sample object pose from region and so2-interval, else
                select determenistic representative
            ------------
            Side effects
            ------------
            cache_entry.solution.obj_tf, numpy array of shape (4,4) - pose of the object is stored here
            cache_entry.eef_tf, numpy array of shape (4, 4) - pose of the end-effector is stored here
        """
        manip_data = self._manip_data[cache_entry.solution.manip.GetName()]
        if b_random:
            angle = so2hierarchy.sample(cache_entry.so2_interval)
            # compute region pose
            cache_entry.region_pose = np.dot(cache_entry.region.sample(b_local=True),
                                             tf_mod.rotation_matrix(angle, [0., 0., 1]))
            contact_tf = np.dot(cache_entry.region.base_tf, cache_entry.region_pose)
        else:
            angle = (cache_entry.so2_interval[1] + cache_entry.so2_interval[0]) / 2.0  # rotation around local z axis
            contact_tf = np.dot(cache_entry.region.contact_tf, tf_mod.rotation_matrix(angle, [0., 0., 1]))
            cache_entry.region_pose = tf_mod.rotation_matrix(angle, [0., 0., 1])
            cache_entry.region_pose[:2, 3] = cache_entry.region.contact_xy
            cache_entry.region_pose[3, 3] = cache_entry.region.cell_size * 0.5
        # compute object pose
        cache_entry.solution.obj_tf = np.dot(contact_tf, cache_entry.plcmnt_orientation.inv_reference_tf)
        # compute end-effector tf
        cache_entry.eef_tf = np.dot(cache_entry.solution.obj_tf, manip_data.grasp_tf)
        # set region state
        cache_entry.region_state = (cache_entry.region_pose[0, 3], cache_entry.region_pose[1, 3], angle)
