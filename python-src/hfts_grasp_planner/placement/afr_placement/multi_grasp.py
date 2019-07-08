import rospy
import os
import random
import scipy
import yaml
import itertools
import heapq
from ordered_set import OrderedSet
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


def load_gripper_info(filename, manip_name):
    """
        Load gripper information for the given manipulator.
        ---------
        Arguments
        ---------
        filename, string - path to file containing gripper information
        manip_name, string - name of the manipulator
        -------
        Returns
        -------
        dict with items:
            gripper_file: <path_to_gripper.robot.xml>
            fingertip_links: list of stings (fingertip names)
            contact_tf: (ref_name, tf), where ref_name is the name of a link
                and tf is the transformation matrix of the finger contact point
                in that link's frame
    """
    basepath = os.path.dirname(filename) + '/'
    with open(filename, 'r') as info_file:
        yaml_dict = yaml.load(info_file)
    gripper_info = {}
    info_val = yaml_dict[manip_name]
    gripper_info['fingertip_links'] = info_val['fingertip_links']
    gripper_info['gripper_file'] = basepath + info_val['gripper_file']
    tf_dict = info_val['contact_tf']
    link_name = tf_dict['reference_link']
    rot = np.array(tf_dict['rotation'])
    trans = np.array(tf_dict['translation'])
    tf = orpy.matrixFromQuat(rot)
    tf[:3, 3] = trans
    gripper_info['contact_tf'] = (link_name, tf)
    return gripper_info


class DMGGraspSet(object):
    """
        Stores a set of grasps that are reachable through in-hand manipulation from some initial grasp
        using the Dexterous Manipulation Graph.
    """
    class Grasp(object):
        def __init__(self, gid, tf, config, dmg_info, parent, distance):
            """
                Create a new grasp.
                ---------
                Arguments
                ---------
                gid, int - unique identifier of the grasp
                tf, np.array of shape (4, 4) - transformation eTo from object frame into eef-frame
                config, np.array of shape (q,) - gripper DOF configuration (it should be q = 1)
                dmg_info, tuple (node, angle, onode, oangle) - dmg information of this grasp
                parent, Grasp - parent grasp on path from initial grasp to this grasp (maybe None for root)
                distance, float - distance to initial grasp
            """
            self.gid = gid
            self.eTo = tf
            self.oTe = utils.inverse_transform(tf)
            self.config = config
            self.dmg_info = dmg_info
            self.parent = parent
            self.distance = distance

    def __init__(self, manip, target_obj, target_obj_file, gripper_info, dmg_info_file, rot_weight=0.1):
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
            target_obj_file, string - path to kinbody xml file representing the target object
            gripper_info, string or dict - either the path to a yaml file containing gripper information
                or the dict returned by load_gripper_info(...) defined at the top of this module.
            dmg_info_file, string - path to yaml file containing information to load dmg
            rot_weight, float - scaling factor of angles in radians for distance computation
        """
        self.dmg = dmg_module.DexterousManipulationGraph.loadFromYaml(dmg_info_file)
        self._rot_weight = rot_weight
        if type(gripper_info) is str:
            gripper_info = load_gripper_info(gripper_info, manip.GetName())
        else:
            assert(type(gripper_info) is dict)
        # set up internal or environment
        self._my_env = orpy.Environment()
        self._my_env.Load(gripper_info['gripper_file'])
        self._my_env.Load(target_obj_file)
        for body in self._my_env.GetBodies():
            body.SetTransform(np.eye(4))
        # store finger info
        self._my_gripper = self._my_env.GetRobots()[0]
        self._fingertip_names = gripper_info["fingertip_links"]
        reference_link, self._rTf = gripper_info["contact_tf"]
        self._reference_finger = self._my_gripper.GetLink(reference_link)
        assert(self._reference_finger is not None)
        # compute grasp set
        self._grasps = None  # int id to grasp
        wTe = manip.GetEndEffectorTransform()
        wTo = target_obj.GetTransform()
        robot = manip.GetRobot()
        config = robot.GetDOFValues(manip.GetGripperIndices())
        self._compute_grasps(wTe, wTo, config)

    def __del__(self):
        self._my_gripper = None
        self._my_env.Destroy()

    def _compute_eTf(self, config):
        """
            Compute the transform eTf for the given finger configuration.
        """
        # obtain eTf
        with self._my_gripper:
            self._my_gripper.SetDOFValues(config)
            wTr = self._reference_finger.GetTransform()
            wTe = self._my_gripper.GetTransform()
            return np.dot(np.dot(utils.inverse_transform(wTe), wTr), self._rTf)

    def _check_grasp_validity(self, grasp):
        # with self._my_gripper:
        self._my_gripper.SetTransform(grasp.oTe)
        min_limits, max_limits = self._my_gripper.GetDOFLimits()
        bgreater = grasp.config >= min_limits
        bsmaller = grasp.config <= max_limits
        bin_limits = np.logical_and.reduce(bgreater) and np.logical_and.reduce(bsmaller)
        if not bin_limits:
            return False
        # TODO open the gripper a bit
        self._my_gripper.SetDOFValues(grasp.config)
        # TODO for more complex grippers we would also need to do self-collision checks
        collisions = [self._my_env.CheckCollision(
            link) for link in self._my_gripper.GetLinks() if link.GetName() not in self._fingertip_names]
        return not reduce(lambda a, b: a and b, collisions, True)

    def _construct_grasp(self, dmg_info, gid, parent, distance):
        """
            Construct a grasp from the given DMG node info.
            ---------
            Arguments
            ---------
            dmg_info, tuple - (node_a, angle_a, node_b, angle_b)
            gid, int - id for the grasp
            parent, Grasp - parent grasp
            distance, float - distance to initial grasp
        """
        node_a, angle_a, node_b, _ = dmg_info
        finger_dist = self.dmg.get_finger_distance(node_a, node_b)
        config = np.array([finger_dist / 2.0])
        eTf = self._compute_eTf(config)
        oTf = self.dmg.get_finger_tf(node_a, angle_a)
        eTo = np.dot(eTf, utils.inverse_transform(oTf))
        return DMGGraspSet.Grasp(gid, eTo, config, dmg_info, parent, distance)

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
        node_pos = self.dmg.get_position(node)
        ocomp = self.dmg.get_component(onode)
        neighbor_nodes = self.dmg.get_neighbors(node)
        for neigh in neighbor_nodes:
            # Two grasps are translational adjacent, if
            #  1. nodes are adjacent (iteration over neighbors solves this)
            #  2. angle is also valid in neighboring node
            #  3. opposite nodes are adjacent
            #  4. opposite angle is valid in opposite node neighbor
            # 2
            if not self.dmg.is_valid_angle(neigh, angle):
                continue
            # 3
            oneigh = self.dmg.get_opposite_node(neigh, angle, comp=ocomp)
            if oneigh is None:
                continue
            if oneigh not in self.dmg.get_neighbors(onode):
                continue
            # 4
            if not self.dmg.is_valid_angle(oneigh, oangle):
                continue
            neighbor_grasps.append((neigh, angle, oneigh, oangle))
            edge_costs.append(np.linalg.norm(self.dmg.get_position(neigh) - node_pos))
        # Two grasps are rotationally adjacent, if
        #  1. angles are adjacent on grid
        #  2. opposite angle is within valid range of opposite node
        neighbor_angles = self.dmg.get_neighbor_angles(node, angle)
        for nangle in neighbor_angles:
            noangle = self.dmg.get_opposing_angle(node, nangle, onode)
            if self.dmg.is_valid_angle(onode, noangle):
                neighbor_grasps.append((node, nangle, onode, noangle))
                edge_costs.append(self.dmg.get_angular_resolution() / 180.0 * np.pi * self._rot_weight)
        return neighbor_grasps, edge_costs

    def _compute_grasps(self, wTe, wTo, config):
        """
            Explore the DMG to compute grasps than can be reached from the current grasp.
            Stores the resulting grasps in self._grasps
        """
        # self._my_env.SetViewer('qtcoin')
        self._grasps = []
        # set initial grasp first. This creates two grasps, the observed one and the closest one in the DMG
        eTo = np.dot(utils.inverse_transform(wTe), wTo)
        # add the observed initial grasp
        observed_initial_grasp = DMGGraspSet.Grasp(len(self._grasps), eTo, config, None, None, 0.0)
        self._grasps.append(observed_initial_grasp)
        # now obtain the dmg info of the closest grasp in the dmg
        # for this, first compute local position of finger
        # TODO the closest DMG grasp may be invalid, in that case we should extend our search for more neighboring grasps
        eTf = self._compute_eTf(config)
        oTf = np.dot(utils.inverse_transform(eTo), eTf)
        start_node, start_angle, bvalid = self.dmg.get_closest_node_angle(oTf)
        if not bvalid:
            rospy.logwarn("Initial grasp is not within the valid angular range of the closest DMG node!")
        finger_dist = 2.0 * config[0]  # TODO this may be specific to Yumi
        onode = self.dmg.get_opposite_node(start_node, start_angle, finger_dist)
        if onode is None:
            raise ValueError(
                "Could not retrieve initial DMG node for initial grasp. There is no opposing valid node.")
        oangle = self.dmg.get_opposing_angle(start_node, start_angle, onode)
        initial_dmg_info = (start_node, start_angle, onode, oangle)
        # create priority list storing tuples (distance, tie_breaker, dmg_info, parent)
        # distance is the minimal distance to the start node
        tb = 0  # tie breaker for heapq comparisons. See https://docs.python.org/2/library/heapq.html#priority-queue-implementation-notes
        open_list = []
        # TODO should the distance of the initial dmg grasp be sth larger than 0.0? how to compute that?
        heapq.heappush(open_list, (0.0, tb, initial_dmg_info, observed_initial_grasp))
        tb += 1
        # store which nodes we have already visited
        closed_set = set()
        while open_list:
            d, _, dmg_info, parent = heapq.heappop(open_list)
            # since we don't have an addressable pq, we may have duplicates in the open_list
            # hence, skip if we already closed this grasp
            if dmg_info[:2] in closed_set:
                continue
            # else add it as a new grasp
            new_grasp = self._construct_grasp(dmg_info, len(self._grasps), parent, d)
            # if it is valid
            if not self._check_grasp_validity(new_grasp):
                continue
            self._grasps.append(new_grasp)
            closed_set.add(dmg_info[:2])
            neighbors, costs = self._get_adjacent_grasps(dmg_info)
            for ndmg_info, cost in itertools.izip(neighbors, costs):
                if ndmg_info[:2] not in closed_set:
                    heapq.heappush(open_list, (d + cost, tb, ndmg_info, new_grasp))
                    tb += 1

    def get_num_grasps(self):
        """
            Return the total number of grasps stored in this set.
        """
        return len(self._grasps)

    def get_grasp(self, gid):
        """
            Return the grasp with the given id.
            Returns None if there is no such grasp.
            ---------
            Arguments
            ---------
            gid, int - id of the gripper
            -------
            Returns
            -------
            grasp, Grasp or None
        """
        if gid >= len(self._grasps):
            return None
        return self._grasps[gid]

    def sample_grasp(self, blacklist=None):
        """
            Sample a random grasp uniformly. Optionally, provide a blacklist of grasps
            not to sample.
            ---------
            Arguments
            ---------
            blacklist, set of ints - grasps to not sample
            -------
            Returns
            -------
            g, Grasp - grasp, NOTE: you shouldn't modify the returned struct
                None if blacklist blocks all possible grasps
        """
        random_idx = random.randint(0, len(self._grasps) - 1)
        if blacklist is None:
            return self._grasps[random_idx]
        if len(blacklist) == len(self._grasps):
            return None
        running_idx = random_idx
        while running_idx in blacklist and running_idx < len(self._grasps):
            running_idx += 1
        if running_idx == len(self._grasps):
            # start the same from the beginning
            running_idx = 0
            while running_idx in blacklist and running_idx < random_idx:
                running_idx += 1
        assert(running_idx not in blacklist)
        return self._grasps[running_idx]

    def return_inhand_path(self, gid):
        """
            Return the in-hand path from the grasp with given id to the initial DMG grasp (id=1).
            ---------
            Arguments
            ---------
            gid, int - id of the grasp
            -------
            Returns
            -------
            grasps, list of Grasps - the grasps that the path transitions through (length n + 1)
            translations, list of np.arrays of shape (3,) describing translational pushes (length n)
            rotations, list of floats (radians) describing rotational pushes (length n)
        """
        if gid == 0 or gid == 1 or gid > len(self._grasps):
            return [], []
        translations = []
        rotations = []
        current_grasp = self._grasps[gid]
        grasps = [current_grasp]
        current_pos = self.dmg.get_position(current_grasp.dmg_info[0])
        current_angle = current_grasp.dmg_info[1]
        parent_grasp = current_grasp.parent
        while parent_grasp.gid != 0:  # do this until the parent grasp is the initial one
            parent_pos = self.dmg.get_position(parent_grasp.dmg_info[0])
            parent_angle = parent_grasp.dmg_info[1]
            delta_pos = current_pos - parent_pos
            delta_angle = self.dmg.get_delta_angle(parent_angle, current_angle)
            translations.append(delta_pos)
            rotations.append(delta_angle / 180.0 * np.pi)
            grasps.append(parent_grasp)
            current_grasp = parent_grasp
            parent_grasp = parent_grasp.parent
            current_pos = parent_pos
            current_angle = parent_angle
        translations.reverse()
        rotations.reverse()
        grasps.reverse()
        return grasps, translations, rotations

    def return_pusher_path(self, gid):
        """
            Return a list of pushes that transfer the object from the initial grasp to
            the grasp with id gid.
            ---------
            Arguments
            ---------
            gid, int - id of the grasp to reach
            -------
            Returns
            -------
            gras_path, a list of Grasps that the pushing path transitions through
            pushes, a list of tuples (start, end, rot_center), 
                where start and end are np.array of shape (4, 4) representing the initial and final pose of the pusher
                rot_center is either None or a np.array of shape (3,) denoting the center of rotation for a rotational push
        """
        if gid > len(self._grasps) or len(self._grasps) <= 1:
            return None
        # get initial grasp on dmg first
        start_grasp = self._grasps[1]
        grasp_path, translations, rotations = self.return_inhand_path(gid)
        finger_positions = np.empty((2, 3))
        finger_positions[0] = self.dmg.get_position(start_grasp.dmg_info[0])
        finger_positions[1] = self.dmg.get_position(start_grasp.dmg_info[2])
        return grasp_path, self.dmg.convert_path((translations, rotations), start_grasp.eTo, finger_positions)


class HierarchicalGraspCache(object):
    """
        Cache for grasp feasibility that uses the AFR hierarchy for caching.
    """

    def __init__(self, hierarchy):
        self._hierarchy = hierarchy
        self._cache = {}

    def get_cached_grasps(self, key):
        """
            Return cached grasps for the given afr key.
            The returned set is ordered by proximity within the hierarchy.
            The further in the sequence a grasp is, the further away it was sampled in the hierarchy.
            Complexity:
            O(len(key) * #grasps)
            ---------
            Arguments
            ---------
            key, tuple - key of a placement
            -------
            Returns
            -------
            OrderedSet of ints - grasp ids that were found valid in this branch of the hierarchy before.
        """
        grasps_to_return = OrderedSet()
        while len(key) > 1:  # we do not cross the manipulator border (i.e. each manipulator has separate grasps)
            if key in self._cache:
                # add all grasps from this level of the hierarchy
                for g in self._cache[key]:
                    grasps_to_return.add(g)
            key = self._hierarchy.get_parent_key(key)
        return grasps_to_return

    def add_grasp_to_cache(self, key, gid):
        """
            Adds the given grasp id to the cache at the given key.
            ---------
            Arguments
            ---------
            key, tuple - key of placement
            gid, int - id of a grasp that was found valid for this placement
        """
        while len(key) > 1:
            if key in self._cache:
                old_set = self._cache[key]
                if gid in old_set:
                    # don't need to do anything in this case
                    return
                old_set.add(gid)
            else:
                self._cache[key] = OrderedSet([gid])
            key = self._hierarchy.get_parent_key(key)


class GlobalGraspCache(object):
    """
        Simple cache that just stores ids of valid grasps globally, independent on where they were found valid.
        See HierarchicalGraspCache for documentation of individual functions.
    """

    def __init__(self):
        self._cache = set()

    def get_cached_grasps(self, key):
        return self._cache

    def add_grasp_to_cache(self, key, gid):
        self._cache.add(gid)


class GripperCollisionChecker(object):
    """
        Helper class to encapsulate collision-checker function for a floating gripper.
    """

    def __init__(self, env, robot, gripper_path, target_obj):
        self._env = env.CloneSelf(orpy.CloningOptions.Bodies)
        self._robot = self._env.GetRobot(robot.GetName())
        self._target_obj = self._env.GetKinBody(target_obj.GetName())
        # TODO what if the environment already contains the gripper?
        bsuccess = self._env.Load(gripper_path)
        assert(bsuccess)
        self._gripper = self._env.GetRobots()[-1]
        self._robot.Enable(False)
        self._target_obj.Enable(False)
        # self._env.SetViewer("qtcoin")

    def __del__(self):
        self._gripper = None
        self._env.Destroy()

    def is_gripper_col_free(self, grasp, obj_tf):
        with self._env:
            self._gripper.SetTransform(np.dot(obj_tf, grasp.oTe))
            self._gripper.SetDOFValues(grasp.config)
            bcollision = self._env.CheckCollision(self._gripper)
            return not bcollision


class DummyGraspProvider(object):
    """
        A dummy grasp provider that always returns the initial grasp.
    """

    def __init__(self, grasp_set):
        self._grasp_set = grasp_set

    def select_grasp(self, key, obj_tf):
        return self._grasp_set.get_grasp(0)

    def report_grasp_validity(self, gid, key, bvalid):
        pass


class NaiveGraspProvider(object):
    """
        A naiive grasp provider simply uniformly samples a new grasp.
    """

    def __init__(self, grasp_set):
        self._grasp_set = grasp_set

    def select_grasp(self, key, obj_tf):
        return self._grasp_set.sample_grasp()

    def report_grasp_validity(self, gid, key, bvalid):
        """
            Report the validity of grasp gid for placement leaf with id gid.
            ---------
            Arguments
            ---------
            gid, int - id of the grasp
            key, tuple - key identifiying the afr leaf
            bvalid, bool - bool indicating whether the grasp was valid or not
        """
        pass


class CacheGraspProvider(object):
    """
        A grasp provider that constructs a cache of feasible grasps,
        and samples this cache to provide grasps.
        Cache types can be 'global' or 'hierarchical'.
    """

    def __init__(self, grasp_set, cache_type, hierarchy, col_checker, pNext=0.1):
        if cache_type == "global":
            self._cache = GlobalGraspCache()
        elif cache_type == 'hierarchical':
            self._cache = HierarchicalGraspCache(hierarchy)
        else:
            raise ValueError("Invalid choice of cache type: %s" % str(cache_type))
        self._grasp_set = grasp_set
        self._col_checker = col_checker
        self._pNext = pNext

    def select_grasp(self, key, obj_tf):
        # first get cached grasps and check whether any appears feasible for this placement
        cached_grasp_ids = self._cache.get_cached_grasps(key)
        if len(cached_grasp_ids) > 0:
            for gid in cached_grasp_ids:
                grasp = self._grasp_set.get_grasp(gid)
                if self._col_checker.is_gripper_col_free(grasp, obj_tf):
                    # roll a die to decide whether we return this or not
                    pdie = np.random.rand()
                    if pdie > self._pNext:
                        return grasp
            tried_grasps = cached_grasp_ids
        else:
            # if we have no cached grasps yet, try the initial one
            initial_grasp = self._grasp_set.get_grasp(0)
            if self._col_checker.is_gripper_col_free(initial_grasp, obj_tf):
                # roll a die to decide whether we return this grasp or not
                if np.random.rand() > self._pNext:
                    return initial_grasp
            tried_grasps = [0]
        # else return a random grasp that is not the initial grasp or one of the cached ones
        if len(tried_grasps) < self._grasp_set.get_num_grasps():
            return self._grasp_set.sample_grasp(blacklist=tried_grasps)
        # else choose a random grasp
        return self._grasp_set.get_grasp(random.choice(tried_grasps))

    def report_grasp_validity(self, gid, key, bvalid):
        if bvalid:
            self._cache.add_grasp_to_cache(key, gid)


class BanditGraspProvider(object):
    """
        A grasp provider that uses multi-armed bandits to select a grasp to try.
    """

    def __init__(self, grasp_set, hierarchy, col_checker, lamb=0.5, gamma=0.8, c=1.41):
        self._hierarchy = hierarchy
        self._col_checker = col_checker
        self._grasp_set = grasp_set
        self._hierarchy = hierarchy
        # _stores for each leaf how many times it has been visited
        self._leaf_visits = {}
        # stores for each face how many times it has been visited, and how many times a grasp has failed
        self._face_visits = {}
        # stores for each element in the hierarchy a dictionary that maps grasp id
        # to tuple (#success, #sampled)
        self._success_rates = {}
        self._gamma = gamma
        self._c = c
        self._lambda = lamb

    def select_grasp(self, key, obj_tf):
        priors, num_face_visits, num_grasp_failures = self._collect_priors(key)
        # return the initial grasp if we have never tried one for this face
        if num_face_visits == 0:
            return self._grasp_set.get_grasp(0)
        # else evaluate our grasp choices
        if key in self._success_rates:
            local_success_rates = self._success_rates[key]
        else:
            local_success_rates = {}
        num_leaf_visits = self._leaf_visits[key] if key in self._leaf_visits else 1
        # what are our unique choices?
        grasp_choices = OrderedSet(priors.keys()) | OrderedSet(local_success_rates.keys())
        # filter grasp_choices based on gripper collision
        filtered_candidates = filter(lambda gid: self._col_checker.is_gripper_col_free(
            self._grasp_set.get_grasp(gid), obj_tf), grasp_choices)
        grasp_choices = OrderedSet(filtered_candidates)
        # how many do we have left?
        num_choices = len(grasp_choices)
        if num_choices == 0:
            # if have no viable options, sample a new one
            grasp = self._grasp_set.sample_grasp(grasp_choices)
            # if there is no viable grasp at all, return initial grasp
            if grasp is None:
                return self._grasp_set.get_grasp[0]
            return grasp
        # else put priors for each grasp_choice in a numpy array
        nppriors = np.array([priors[g] if g in priors else 0.0 for g in grasp_choices])
        # same for local rates
        npsuccesses = np.array(
            [local_success_rates[g][0] if g in local_success_rates else 0 for g in grasp_choices], dtype=np.float)
        # visit counts
        npvisits = np.array(
            [local_success_rates[g][1] if g in local_success_rates else 0 for g in grasp_choices], dtype=np.float)
        prior_weights = np.power(self._lambda, npvisits)
        local_rates = np.zeros(num_choices)
        visited_bandits = npvisits.nonzero()[0]
        local_rates[visited_bandits] = npsuccesses[visited_bandits] / npvisits[visited_bandits]
        exploration_vals = np.sqrt(2.0 * np.log(num_leaf_visits) / (npvisits + 1.0))
        # compute ucb scores
        ucb_scores = prior_weights * nppriors + (1.0 - prior_weights) * local_rates + self._c * exploration_vals
        # choose the grasp that maximizes ucb score
        maximizing_bandit = np.argmax(ucb_scores)
        chosen_grasp = grasp_choices[maximizing_bandit]
        grasp_ucb_score = ucb_scores[maximizing_bandit]
        # before returning this grasp, consider sampling a new one
        # TODO given that we filtered the grasps above, it doesn't seem sensible to have a sample-new-grasp arm with
        # TODO with such high score (considering that priors are usually rather small)
        if num_grasp_failures > 0 and len(grasp_choices) < self._grasp_set.get_num_grasps():
            # TODO one can construct scenarios where a kinematically reachable grasp without collisions
            # TODO can not be reached with a motion plan. In this case, we would need to try different grasps.
            # TODO Accordingly, the new_grasp_score should also be able to grow even if all grasps are always
            # TODO collision-free and reachable.
            new_grasp_score = self._c * np.sqrt(np.log(num_grasp_failures) / num_choices)
            if new_grasp_score > grasp_ucb_score:
                return self._grasp_set.sample_grasp(grasp_choices)
        return self._grasp_set.get_grasp(chosen_grasp)

    def report_grasp_validity(self, gid, key, bvalid):
        assert(self._hierarchy.is_leaf(key))
        # first count a face visit
        face_key = key[:2]
        if face_key not in self._face_visits:
            self._face_visits[face_key] = [1, int(not bvalid)]
        else:
            self._face_visits[face_key][0] += 1
            self._face_visits[face_key][1] += int(not bvalid)
        # second increase leaf visits
        if key in self._leaf_visits:
            self._leaf_visits[key] += 1
        else:
            self._leaf_visits[key] = 1
        # next adjust success rates
        while len(key) > 1:
            if key in self._success_rates:
                rates = self._success_rates[key]
            else:
                rates = {}
                self._success_rates[key] = rates
            if gid in rates:
                success, visits = rates[gid]
            else:
                success, visits = 0, 0
            rates[gid] = (success + bvalid, visits + 1)
            key = self._hierarchy.get_parent_key(key)

    def _collect_priors(self, key):
        """
            Ascend the hierarchy from key and collect priors on grasp success.
            ---------
            Arguments
            ---------
            key, tuple - afr key, must be referring to a leaf
            -------
            Returns
            -------
            priors, dict - mapping grasp id to success belief (float in range [0, 1])
            num_face_samples, int - number of times this face has been sampled for placement
            num_failed_grasps, int - number of times a grasp sampled for this face has been the reason for failure
        """
        priors = {}
        face_key = key[:2]
        num_face_visits, num_failed_grasps = self._face_visits[face_key] if face_key in self._face_visits else (0, 0)
        key = self._hierarchy.get_parent_key(key)
        discount = self._gamma
        while len(key) > 1:
            if key in self._success_rates:
                for gid, (success, visits) in self._success_rates[key].iteritems():
                    if gid not in priors:
                        priors[gid] = discount * float(success) / float(visits)
            key = self._hierarchy.get_parent_key(key)
            discount *= self._gamma
        return priors, num_face_visits, num_failed_grasps


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
        grasp_selector_type, string - 'naive', 'cache_global', 'cache_hierarchy', 'bandit'
            defining what grasp selector type to use.
            naive: Simple uniform sampler
            cache_global: Cache sampler that uses a global cache
            cache_hierarchy: Cache sample that uses a hierarchy-based cache
            bandit: Multi-armed bandit sampling using a hierarchy-based success rate as prior
    """
    class ManipulatorData(object):
        """
            Struct (named tuple) that stores manipulator data.
        """

        def __init__(self, manip, ik_solver, grasp_set, gripper_file):
            """
                Create a new instance of manipulator data.
                ---------
                Arguments
                ---------
                manip - OpenRAVE manipulator
                ik_solver - ik_module.IKSolver, IK solver for end-effector poses
                grasp_set - DMGGraspSet, DMG grasp set for this manipulator
                gripper_file, string - path to a robot xml representing only the gripper
            """
            self.manip = manip
            self.manip_links = utils.get_manipulator_links(manip)
            self.ik_solver = ik_solver
            self.lower_limits, self.upper_limits = self.manip.GetRobot().GetDOFLimits(manip.GetArmIndices())
            self.grasp_set = grasp_set
            self.gripper_path = gripper_file

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
                cache_entry.solution.objective_value is set to the new objective value
                cache_entry.eef_tf is set to the updated end-effector pose
                cache_entry.region_pose is set to the updated pose (TODO what if we leave the region?)
                cache_entry.region_state is set to the updated state
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
                    inv_grasp_tf = utils.inverse_transform(cache_entry.solution.grasp_tf)
                    utils.set_grasp(manip, self.object_data.kinbody,
                                    inv_grasp_tf,
                                    cache_entry.solution.grasp_config)
                    reference_pose = np.dot(inv_grasp_tf, cache_entry.plcmnt_orientation.reference_tf)
                    manip.SetLocalToolTransform(reference_pose)
                    self.robot.SetActiveDOFs(manip.GetArmIndices())
                    # init jacobian descent
                    q_current = cache_entry.solution.arm_config
                    # iterate as long as the gradient is not None, zero or we are in collision or out of joint limits
                    # while q_grad is not None and grad_norm > self.epsilon and b_in_limits:
                    for it in xrange(self.max_iterations):
                        in_limits = (q_current >= lower).all() and (q_current <= upper).all()
                        if in_limits:
                            q_grad = self._compute_gradient(cache_entry, q_current, manip_data, it != 0)
                            grad_norm = np.linalg.norm(q_grad) if q_grad is not None else 0.0
                            if q_grad is None or grad_norm < self.grad_epsilon:
                                break
                            arm_configs.append(np.array(q_current))
                            q_current -= self.step_size * q_grad / grad_norm  # update q_current
                        else:
                            break
                    if len(arm_configs) > 1:  # first element is start configuration
                        self._set_cache_entry_values(cache_entry, arm_configs[-1], manip_data)
                        manip.SetLocalToolTransform(np.eye(4))
                        manip.GetRobot().Release(self.object_data.kinbody)
                        return arm_configs[1:]
                    # we failed in the first iteration, just set it back to what we started from
                    self._set_cache_entry_values(cache_entry, cache_entry.solution.arm_config, manip_data)
                    manip.SetLocalToolTransform(np.eye(4))
                    manip.GetRobot().Release(self.object_data.kinbody)
                    return []  # else return empty array

        def _set_cache_entry_values(self, cache_entry, q, manip_data):
            """
                Sets cache entry values for the given q.
                Assumes that the manipulator has a local tool transform set to the reference pose of
                the placement face.
            """
            inv_grasp_tf = utils.inverse_transform(cache_entry.solution.grasp_tf)
            manip = manip_data.manip
            self.robot.SetActiveDOFValues(q)
            cache_entry.solution.arm_config = q
            cache_entry.eef_tf = manip.GetEndEffector().GetTransform()  # pose of the link exluding local transform!!!
            cache_entry.solution.obj_tf = np.matmul(cache_entry.eef_tf, inv_grasp_tf)
            ref_pose = manip.GetEndEffectorTransform()  # this takes the local tool transform into account
            # TODO we are not constraining the pose to stay in the region here, so technically we would need to compute
            # TODO which region we are actually in now (we might transition into a neighboring region)
            cache_entry.region_pose = np.matmul(utils.inverse_transform(cache_entry.region.base_tf), ref_pose)
            _, _, ez = tf_mod.euler_from_matrix(cache_entry.region_pose)
            theta = utils.normalize_radian(ez)
            cache_entry.region_state = (cache_entry.region_pose[0, 3], cache_entry.region_pose[1, 3], theta)
            cache_entry.solution.objective_value = self.obj_fn(cache_entry.solution.obj_tf)

        def _compute_gradient(self, cache_entry, q_current, manip_data, bimprove_objective=True):
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
            # TODO bimprove_objective is a hack for the case that the initial objective value is slightly smaller than
            # TODO it is when we evaluate the objective here. Why can this happen? (Imprecise arm configuration?)
            if cache_entry.solution.objective_value < self._last_obj_value and bimprove_objective:
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
            # _, col_grad = self.collision_constraint.get_chomps_collision_gradient(cache_entry, q_current)
            # col_grad[:] = np.matmul((np.eye(col_grad.shape[0]) -
            #                          np.matmul(inv_jac, np.matmul(self.damping_matrix, jacobian))), col_grad)
            # qgrad += col_grad
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
        # TODO update jacobian optimizer as needed
        self._jacobian_optimizer = MultiGraspAFRRobotBridge.JacobianOptimizer(self._contact_constraint,
                                                                              self._collision_constraint,
                                                                              objective_fn, self._robot_data,
                                                                              self._object_data,
                                                                              joint_limit_margin=parameters["joint_limit_margin"])
        self._parameters = parameters
        self._use_jacobian_proj = parameters["proj_type"] == "jac"
        # load the grippers into the world for grasp collision checking
        self._gripper_col_checkers = {}
        env = self._robot_data.robot.GetEnv()
        for manip_name, manip_data in self._manip_data.iteritems():
            self._gripper_col_checkers[manip_name] = GripperCollisionChecker(
                env, self._robot_data.robot, manip_data.gripper_path, self._object_data.kinbody)

        # create grasp selectors as demanded
        self._grasp_selectors = {}
        if 'grasp_selector_type' not in self._parameters:
            self._parameters['grasp_selector_type'] = 'cache_hierarchy'
        if self._parameters['grasp_selector_type'] == 'naive':
            self._grasp_selectors = {mn: NaiveGraspProvider(md.grasp_set) for mn, md in self._manip_data.iteritems()}
        elif self._parameters['grasp_selector_type'] == 'cache_global':
            self._grasp_selectors = {mn: CacheGraspProvider(
                md.grasp_set, "global", self._hierarchy, self._gripper_col_checkers[mn]) for mn, md in self._manip_data.iteritems()}
        elif self._parameters['grasp_selector_type'] == 'cache_hierarchy':
            self._grasp_selectors = {mn: CacheGraspProvider(
                md.grasp_set, "hierarchical", self._hierarchy, self._gripper_col_checkers[mn]) for mn, md in self._manip_data.iteritems()}
        elif self._parameters['grasp_selector_type'] == 'bandit':
            self._grasp_selectors = {mn: BanditGraspProvider(
                md.grasp_set, self._hierarchy, self._gripper_col_checkers[mn]) for mn, md in self._manip_data.iteritems()}
        elif self._parameters['grasp_selector_type'] == 'dummy':
            self._grasp_selectors = {mn: DummyGraspProvider(md.grasp_set) for mn, md in self._manip_data.iteritems()}
        else:
            raise ValueError("Unknown grasp selector type %s." % self._parameters['grasp_selector_type'])

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
        manip_name = manip.GetName()
        manip_data = self._manip_data[manip_name]
        po = afr_info[1]
        region = afr_info[2]
        so2_interval = np.array([0, 2.0 * np.pi])
        if len(afr_info) == 5:
            region = afr_info[3]
            so2_interval = afr_info[4]
        # construct a solution without valid values yet
        new_solution = placement_interfaces.PlacementGoalSampler.PlacementGoal(
            manip=manip, arm_config=None, obj_tf=None, key=len(self._solutions_cache), objective_value=None,
            grasp_tf=None, grasp_config=None, grasp_id=None)
        # create a cache entry for this solution
        sol_cache_entry = afr_core.AFRRobotBridge.SolutionCacheEntry(
            key=key, region=region, plcmnt_orientation=po, so2_interval=so2_interval, solution=new_solution)
        self._solutions_cache.append(sol_cache_entry)
        if b_optimize_constraints:
            raise RuntimeError("Not implemented")
        else:
            # compute object pose for this hierarchy element (we randomly sample withing the range)
            angle = so2hierarchy.sample(sol_cache_entry.so2_interval)
            # compute region pose
            sol_cache_entry.region_pose = np.dot(sol_cache_entry.region.sample(b_local=True),
                                                 tf_mod.rotation_matrix(angle, [0., 0., 1]))
            contact_tf = np.dot(sol_cache_entry.region.base_tf, sol_cache_entry.region_pose)
            new_solution.obj_tf = np.dot(contact_tf, sol_cache_entry.plcmnt_orientation.inv_reference_tf)
            # set region state
            sol_cache_entry.region_state = (sol_cache_entry.region_pose[0, 3], sol_cache_entry.region_pose[1, 3], angle)
            # get a grasp from the grasp selector
            grasp_to_try = self._grasp_selectors[manip_name].select_grasp(key, new_solution.obj_tf)
            # update grasp_tf, grasp_config, grasp_id and eef
            new_solution.grasp_tf = grasp_to_try.oTe
            new_solution.grasp_config = grasp_to_try.config
            new_solution.grasp_id = grasp_to_try.gid
            sol_cache_entry.eef_tf = np.dot(new_solution.obj_tf, grasp_to_try.oTe)
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
            # The local optimizer was successful at improving the solution a bit
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
            # update grasp cache
            cache_entry.key = self.get_leaf_key(cache_entry.solution)
            afr_info = self._hierarchy.get_afr_information(cache_entry.key)
            manip_name = afr_info[0].GetName()
            self._grasp_selectors[manip_name].report_grasp_validity(
                solution.grasp_id, cache_entry.key, True)
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

    def is_valid(self, solution, b_improve_objective):
        """
            Return whether the given PlacementSolution is valid.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
            b_improve_objective, bool - If True, the solution has to be better than the current minimal objective.
            -------
            Returns
            -------
            valid, bool
        """
        cache_entry = self._solutions_cache[solution.key]
        assert(cache_entry.solution == solution)
        afr_info = self._hierarchy.get_afr_information(cache_entry.key)
        manip_name = afr_info[0].GetName()
        self._call_stats[1] += 1
        bgrasp_needs_feedback = False
        # kinematic reachability?
        if cache_entry.bkinematically_reachable is None:
            self._reachability_constraint.check_reachability(cache_entry)
            bgrasp_needs_feedback = True
        # collision free?
        if cache_entry.barm_collision_free is None or cache_entry.bobj_collision_free is None:
            self._collision_constraint.check_collision(cache_entry)
            bgrasp_needs_feedback = True
        # let the grasp selector know about the grasp's validity for this placement, if this is the first time we evaluate it
        if bgrasp_needs_feedback:
            bgrasp_valid = cache_entry.bkinematically_reachable and cache_entry.barm_collision_free
            self._grasp_selectors[manip_name].report_grasp_validity(solution.grasp_id, cache_entry.key, bgrasp_valid)
        # next check whether the object pose is actually a stable placement
        if cache_entry.bstable is None:
            self._contact_constraint.check_contacts(cache_entry)
        # finally check whether the objective is an improvement
        if b_improve_objective and cache_entry.bbetter_objective is None:
            if cache_entry.objective_val is None:  # objective has not been evaluated yet
                self.evaluate(solution)
            self._objective_constraint.check_objective_improvement(cache_entry)
        return cache_entry.bkinematically_reachable and cache_entry.barm_collision_free\
            and cache_entry.bobj_collision_free and cache_entry.bstable \
            and (cache_entry.bbetter_objective or not b_improve_objective)

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
        # TODO do we want to compute this differently for multi-grasp?
        # first compute normalizer
        normalizer = self._parameters["weight_arm_col"] + \
            self._parameters["weight_obj_col"] + self._parameters["weight_contact"] + \
            self._parameters["weight_reachable"]
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
        self.is_valid(solution, b_incl_obj)  # ensure all validity flags are set
        val = 0.0
        # compute binary or continuous sub relaxatoins
        if self._parameters["relaxation_type"] == "sub-binary":
            val += self._parameters["weight_arm_col"] * float(cache_entry.barm_collision_free)
            val += self._parameters["weight_obj_col"] * float(cache_entry.bobj_collision_free)
            val += self._parameters["weight_contact"] * float(cache_entry.bstable)
            val += self._parameters["weight_reachable"] * float(cache_entry.bkinematically_reachable)
            if b_incl_obj:
                valid_sol = float(cache_entry.bobj_collision_free) * float(cache_entry.bstable)
                val += valid_sol * self._parameters["weight_objective"] * float(cache_entry.bbetter_objective)
        else:  # compute continuous relaxation
            raise ValueError("Continuous relaxation not supported")
            # assert(self._parameters["relaxation_type"] == "continuous")
            # contact_val = self._contact_constraint.get_relaxation(cache_entry)
            # arm_col_val, obj_col_val = self._collision_constraint.get_relaxation(cache_entry)
            # # rospy.logdebug("contact value: %f, arm-collision value: %f, obj_collision value: %f" %
            # #                (contact_val, arm_col_val, obj_col_val))
            # val += self._parameters["weight_arm_col"] * arm_col_val
            # val += self._parameters["weight_obj_col"] * obj_col_val
            # val += self._parameters["weight_contact"] * contact_val
            # solution.data = {'arm_col': arm_col_val, "obj_col": obj_col_val, "contact": contact_val, "total": val}
            # if b_incl_obj:
            #     valid_sol = float(cache_entry.bobj_collision_free) * float(cache_entry.bstable)
            #     val += valid_sol * self._parameters["weight_objective"] * \
            #         self._objective_constraint.get_relaxation(cache_entry)
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
