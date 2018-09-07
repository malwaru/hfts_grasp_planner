#!/usr/bin/env python
""" This module contains a general hierarchically organized goal region sampler. """

import ast
import rospy
import copy
import math
import numpy
import random
import abc
from rtree import index
from hfts_grasp_planner.rrt import SampleData
from hfts_grasp_planner.sdf.robot import RobotOccupancyOctree

NUMERICAL_EPSILON = 0.00001


class CSpaceSampler:
    """
        Interface for configuration space sampler.
    """

    def __init__(self):
        pass

    def sample(self):
        pass

    def sample_gaussian_neighborhood(self, config, variance):
        pass

    def is_valid(self, qSample):
        pass

    def get_sampling_step(self):
        pass

    def get_space_dimension(self):
        return 0

    def get_upper_bounds(self):
        pass

    def get_lower_bounds(self):
        pass

    def get_scaling_factors(self):
        return self.get_space_dimension() * [1]

    def distance(self, config_a, config_b):
        total_sum = 0.0
        scaling_factors = self.get_scaling_factors()
        for i in range(len(config_a)):
            total_sum += scaling_factors[i] * math.pow(config_a[i] - config_b[i], 2)
        return math.sqrt(total_sum)

    def configs_are_equal(self, config_a, config_b):
        dist = self.distance(config_a, config_b)
        return dist < NUMERICAL_EPSILON

    def interpolate(self, start_sample, end_sample, projection_function=lambda x, y: y):
        """
        Samples cspace linearly from the startSample to endSample until either
        a collision is detected or endSample is reached. All intermediate sampled configurations
        are returned in a list as SampleData.
        If a projectionFunction is specified, each sampled configuration is projected using this
        projection function. This allows to interpolate within a constraint manifold, i.e. some subspace of
        the configuration space. Additionally to the criterias above, the method also terminates when
        no more progress is made towards endSample.
        @param start_sample        The SampleData to start from.
        @param end_sample          The SampleData to sample to.
        @param projection_function (Optional) A projection function on a contraint manifold.
        @return A tuple (bSuccess, samples), where bSuccess is True if a connection to endSample was found;
                samples is a list of all intermediate sampled configurations [startSample, ..., lastSampled].
        """
        waypoints = [start_sample]
        config_sample = start_sample.get_configuration()
        pre_config_sample = start_sample.get_configuration()
        dist_to_target = self.distance(end_sample.get_configuration(), config_sample)
        while True:
            pre_dist_to_target = dist_to_target
            dist_to_target = self.distance(end_sample.get_configuration(), config_sample)
            if self.configs_are_equal(config_sample, end_sample.get_configuration()):
                # We reached our target. Since we want to keep data stored in the target, simply replace
                # the last waypoint with the instance endSample
                waypoints.pop()
                waypoints.append(end_sample)
                return True, waypoints
            elif dist_to_target > pre_dist_to_target:
                # The last sample we added, took us further away from the target, then the previous one.
                # Hence remove the last sample and return.
                waypoints.pop()
                return False, waypoints
            # We are still making progress, so sample a new sample
            # To prevent numerical issues, we move at least NUMERICAL_EPSILON
            step = min(self.get_sampling_step(), max(dist_to_target, NUMERICAL_EPSILON))
            config_sample = config_sample + step * (end_sample.get_configuration() - config_sample) / dist_to_target
            # Project the sample to the constraint manifold
            config_sample = projection_function(pre_config_sample, config_sample)
            if config_sample is not None and self.is_valid(config_sample):
                # We have a new valid sample, so add it to the waypoints list
                waypoints.append(SampleData(numpy.copy(config_sample)))
                pre_config_sample = numpy.copy(config_sample)
            else:
                # We ran into an obstacle - we won t get further, so just return what we have so far
                return False, waypoints


class GoalHierarchy(object):
    __metaclass__ = abc.ABCMeta

    class GoalHierarchyNode(object):
        __metaclass__ = abc.ABCMeta
        """
            Represents a node of a GoalHierarchy.
        """

        def __init__(self, configuration, cache_id=-1):
            self.configuration = configuration
            self.cache_id = cache_id

        def get_configuration(self):
            return self.configuration

        def to_sample_data(self):
            return SampleData(self.configuration, self.get_additional_data(),
                              self.get_data_copy_fn(), id_num=self.cache_id)

        @abc.abstractmethod
        def is_valid(self):
            """
                Return whether this node is valid. That means it does not violate any constraints
                and at least one associated configuration is collision-free.
            """
            pass

        @abc.abstractmethod
        def is_goal(self):
            """
                Return whether this node represents a goal. 
                In addition to being valid, a goal must represent a configuration that is 
                fulfilling all goal criteria.
            """
            pass

        @abc.abstractmethod
        def hierarchy_info_str(self):
            """
                Return a string representation of the hierarchy information for debugging and printing.
            """
            pass

        @abc.abstractmethod
        def get_num_possible_children(self):
            """
                Return the number of possible children of this node.
            """
            pass

        @abc.abstractmethod
        def get_num_possible_leaves(self):
            """
                Return the number of possible leaves in the subranch rooted at this node.
            """
            pass

        @abc.abstractmethod
        def get_hashable_label(self):
            """
                Return a unique and hashable identifier for this node.
            """
            pass

        @abc.abstractmethod
        def get_label(self):
            """
                Return a unique identifier for this node. This identifier does not need to be
                hashable.
            """
            pass

        @abc.abstractmethod
        def get_depth(self):
            """
                Return the depth of this node.
            """
            pass

        @abc.abstractmethod
        def get_additional_data(self):
            """
                Return optional additional data associated with this node.
                This may be any python object that stores additional information that an external
                caller may be interested in, such as the hand configuration of a grasp, or a hand-opening
                policy for dropping an object, etc.
            """
            pass

        def get_data_copy_fn(self):
            """
                Return a function that when passed an object returned by self.get_additional_data() produces
                a copy of that object. Defaults to copy.deepcopy
            """
            return copy.deepcopy

        @abc.abstractmethod
        def is_extendible(self):
            """
                Return whether this node is extendible, i.e. it has children.
            """
            pass

        def is_leaf(self):
            """
                Return whether this node is a leaf. This is equivalent to not self.is_extendible()
            """
            return not self.is_extendible()

        def __repr__(self):
            return self.__str__()

        def __str__(self):
            return "{SamplingResult:[Config=" + str(self.configuration) + "; Info=" + self.hierarchy_info_str() + "]}"

    @abc.abstractmethod
    def sample(self, depth_limit, post_opt=True):
        """
            Sample a goal from the root level.
            ---------
            Arguments
            ---------
            depth_limit, int - maximal number of levels to descend (needs to be at least 1)
            post_opt, bool - flag indicating whether (computationally) expensive post optimization
                at a leaf node is allowed to be performed.
            ---------
            Returns
            ---------
            result, GoalHierarchyNode - a newly sampled goal hierarchy node
        """
        pass

    @abc.abstractmethod
    def sample_warm_start(self, hierarchy_node, depth_limit, label_cache=None, post_opt=False):
        """
            Sample a goal from the given node on.
            ---------
            Arguments
            ---------
            hierarchy_node, GoalHierarchyNode - node to start sampling from, may be root node.
            depth_limit, int - maximal number of levels to descend (needs to be at least 1)
            label_cache, ???? - ???? # TODO document
            post_opt, bool - flag indicating whether (computationally) expensive post optimization 
                at a leaf node is allowed to be performed.
            ---------
            Returns
            ---------
            result, GoalHierarchyNode - a newly sampled goal hierarchy node
        """
        pass

    @abc.abstractmethod
    def set_max_iter(self, iterations):
        """
            Set the maximum number of iterations to search for a new hierarhcy node on a
            single level.
            ---------
            Arguments
            ---------
            iterations, int - number of iterations
        """
        pass

    @abc.abstractmethod
    def get_max_depth(self):
        """
            Get the maximum depth of the hierarchy.
        """
        pass

    @abc.abstractmethod
    def get_root(self):
        """
            Return the GoalHierarchyNode representing the root of the hierarchy.
        """
        pass


class SimpleHierarchyNode:
    """
        A hierarchy node for the naive implementation.
    """
    class DummyHierarchyInfo:
        def __init__(self, unique_label):
            self.unique_label = unique_label

        def get_unique_label(self):
            return self.unique_label

        def is_goal(self):
            return False

        def is_valid(self):
            return False

    def __init__(self, config, hierarchy_info):
        self.config = config
        self.hierarchy_info = hierarchy_info
        self.children = []

    def get_children(self):
        return self.children

    def get_active_children(self):
        return self.children

    def get_num_children(self):
        return len(self.children)

    def get_max_num_children(self):
        return 1  # self.hierarchy_info.getPossibleNumChildren()

    def get_num_leaves_in_branch(self):
        return 0

    def get_max_num_leaves_in_branch(self):
        return 1  # self.hierarchy_info.getPossibleNumLeaves()

    def get_unique_label(self):
        return self.hierarchy_info.get_unique_label()

    def get_T(self):
        if self.hierarchy_info.is_goal() and self.hierarchy_info.is_valid():
            return 1.5
        if self.hierarchy_info.is_valid():
            return 1.0
        return 0.0

    def is_goal(self):
        return self.hierarchy_info.is_goal()

    def get_active_configuration(self):
        return self.config

    def add_child(self, child):
        self.children.append(child)


class NaiveGoalSampler:
    """
        The naive goal sampler, which always goes all the way down in the hierarchy.
    """

    def __init__(self, goal_region, num_iterations=40, debug_drawer=None):
        self.goal_region = goal_region
        self.depth_limit = goal_region.get_max_depth()
        self.goal_region.set_max_iter(num_iterations)
        self._debug_drawer = debug_drawer
        self.clear()

    def clear(self):
        self.cache = []
        self._root_node = SimpleHierarchyNode(None, self.goal_region.get_root())
        self._label_node_map = {}
        self._label_node_map['root'] = self._root_node
        if self._debug_drawer is not None:
            self._debug_drawer.clear()

    def get_num_goal_nodes_sampled(self):
        return len(self._label_node_map)

    def _compute_ancestor_labels(self, unique_label):
        label_as_list = ast.literal_eval(unique_label)
        depth = len(label_as_list) / 3
        n_depth = depth - 1
        ancestor_labels = []
        while n_depth >= 1:
            ancestor_label = []
            for f in range(3):
                ancestor_label.extend(label_as_list[f * depth:f * depth + n_depth])
            ancestor_labels.append(str(ancestor_label))
            label_as_list = ancestor_label
            depth -= 1
            n_depth = depth - 1
        ancestor_labels.reverse()
        return ancestor_labels

    def _add_new_sample(self, sample):
        unique_label = sample.hierarchy_info.get_unique_label()
        ancestor_labels = self._compute_ancestor_labels(unique_label)
        parent = self._root_node
        for ancestorLabel in ancestor_labels:
            if ancestorLabel in self._label_node_map:
                parent = self._label_node_map[ancestorLabel]
            else:
                ancestor_node = SimpleHierarchyNode(config=None,
                                                    hierarchy_info=SimpleHierarchyNode.DummyHierarchyInfo(
                                                        ancestorLabel))
                parent.add_child(ancestor_node)
                self._label_node_map[ancestorLabel] = ancestor_node
                parent = ancestor_node
        if unique_label in self._label_node_map:
            return
        new_node = SimpleHierarchyNode(config=sample.get_configuration(), hierarchy_info=sample.hierarchy_info)
        self._label_node_map[unique_label] = new_node
        parent.add_child(new_node)

    def sample(self, b_dummy=False):
        rospy.logdebug('[NaiveGoalSampler::sample] Sampling a goal in the naive way')
        my_sample = self.goal_region.sample(self.depth_limit)
        self._add_new_sample(my_sample)
        if self._debug_drawer is not None:
            self._debug_drawer.draw_hierarchy(self._root_node)
        if not my_sample.is_valid() or not my_sample.is_goal():
            rospy.logdebug('[NaiveGoalSampler::sample] Failed. Did not get a valid goal!')
            return SampleData(None)
        else:
            my_sample.cacheId = len(self.cache)
            self.cache.append(my_sample)

        rospy.logdebug('[NaiveGoalSampler::sample] Success. Found a valid goal!')
        return my_sample.to_sample_data()

    # def get_quality(self, sample_data):
    #     idx = sample_data.get_id()
    #     return self.cache[idx].hierarchy_info.get_quality()

    def is_goal(self, sample):
        sampled_before = 0 < sample.get_id() < len(self.cache)
        if sampled_before:
            return self.goal_region.is_goal(self.cache[sample.get_id()])
        return False


class FreeSpaceModel(object):
    """
        This class builds a model of the collision-free space in form of samples stored in trees.
    """

    def __init__(self, c_space_sampler):
        self._trees = []
        self._c_space_sampler = c_space_sampler

    def add_tree(self, tree):
        self._trees.append(tree)

    def remove_tree(self, tree_id):
        new_tree_list = []
        for tree in self._trees:
            if tree.get_id() != tree_id:
                new_tree_list.append(tree)
        self._trees = new_tree_list

    def get_nearest_configuration(self, config):
        (dist, nearest_config) = (float('inf'), None)
        for tree in self._trees:
            tree_node = tree.nearest_neighbor(SampleData(config))
            tmp_config = tree_node.get_sample_data().get_configuration()
            tmp_dist = self._c_space_sampler.distance(config, tmp_config)
            if tmp_dist < dist:
                dist = tmp_dist
                nearest_config = tmp_config
        return dist, nearest_config


class ExtendedFreeSpaceModel(FreeSpaceModel):
    """
        This class extends the FreeSpaceModel to also store individual samples, in two forms:
        - approximate
        - temporary
        An approximate sample is a configuration that is individually stored in a nearest neighbor data structure
        and represents a configuration of an approximate goal.
        A temporary sample is a configuration that is individually stored in a list and is used for what exactly?

    """

    def __init__(self, c_space_sampler):
        super(ExtendedFreeSpaceModel, self).__init__(c_space_sampler)
        self._scaling_factors = c_space_sampler.get_scaling_factors()
        prop = index.Property()
        prop.dimension = c_space_sampler.get_space_dimension()
        self._approximate_index = index.Index(properties=prop)
        self._approximate_configs = []
        self._temporal_mini_cache = []

    def get_nearest_configuration(self, config):
        (tree_dist, nearest_tree_config) = super(ExtendedFreeSpaceModel, self).get_nearest_configuration(config)
        (temp_dist, temp_config) = self.get_closest_temporary(config)
        if temp_dist < tree_dist:
            tree_dist = temp_dist
            nearest_tree_config = temp_config

        if len(self._approximate_configs) > 0:
            point_list = self._make_coordinates(config)
            nns = list(self._approximate_index.nearest(point_list))
            nn_id = nns[0]
            nearest_approximate = self._approximate_configs[nn_id]
            assert nearest_approximate is not None
            dist = self._c_space_sampler.distance(nearest_approximate, config)
            if nearest_tree_config is not None:
                if tree_dist <= dist:
                    return tree_dist, nearest_tree_config
            return dist, nearest_approximate
        elif nearest_tree_config is not None:
            return tree_dist, nearest_tree_config
        else:
            return float('inf'), None

    def add_temporary(self, configs):
        self._temporal_mini_cache.extend(configs)

    def clear_temporary_cache(self):
        self._temporal_mini_cache = []

    def get_closest_temporary(self, config):
        min_dist, closest = float('inf'), None
        for tconfig in self._temporal_mini_cache:
            tdist = self._c_space_sampler.distance(config, tconfig)
            if tdist < min_dist:
                min_dist = tdist
                closest = tconfig
        return min_dist, closest

    def add_approximate(self, config):
        cid = len(self._approximate_configs)
        self._approximate_configs.append(config)
        point_list = self._make_coordinates(config)
        self._approximate_index.insert(cid, point_list)

    def draw_random_approximate(self):
        idx = len(self._approximate_configs) - 1
        if idx == -1:
            return None
        config = self._approximate_configs.pop()
        assert config is not None
        self._approximate_index.delete(idx, self._make_coordinates(config))
        return config

    def _make_coordinates(self, config):
        point_list = list(config)
        point_list = map(lambda x, y: math.sqrt(x) * y, self._scaling_factors, point_list)
        point_list += point_list
        return point_list


class NodeRating(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._parameters = {}
        self._connected_space = None
        self._non_connected_space = None

    @abc.abstractmethod
    def max_rating(self):
        pass

    @abc.abstractmethod
    def update_ratings(self, node):
        """
            Update cached ratings of the branch rooted at node.
        """
        pass

    @abc.abstractmethod
    def get_root_rating(self):
        pass

    @abc.abstractmethod
    def t(self, node):
        pass

    def set_parameters(self, **kwargs):
        for (key, value) in kwargs.iteritems():
            if key in self._parameters.iteritems():
                self._parameters[key] = value

    def set_connected_space(self, connected_space):
        self._connected_space = connected_space

    def set_non_connected_space(self, non_connected_space):
        self._non_connected_space = non_connected_space

    def T(self, node):
        temps_children = 0.0
        t_node = self.t(node)
        avg_child_temp = t_node
        if len(node.get_active_children()) > 0:
            for child in node.get_active_children():
                temps_children += self.T(child)
            avg_child_temp = temps_children / float(len(node.get_active_children()))
        node.set_T((t_node + avg_child_temp) / 2.0)
        NodeRating.T_c(node)
        NodeRating.T_p(node)
        return node.get_T()

    @staticmethod
    def T_c(node):
        mod_branch_coverage = node.get_num_leaves_in_branch() / (node.get_max_num_leaves_in_branch() + 1)
        T_c = node.get_T() * (1.0 - mod_branch_coverage)
        node.set_T_c(T_c)
        return T_c

    @staticmethod
    def T_p(node):
        T_p = node.get_T() * (1.0 - node.get_coverage())
        node.set_T_p(T_p)
        return T_p


class ProximityRating(NodeRating):
    def __init__(self, cspace_diameter, **kwargs):
        """
            ---------
            Arguments
            ---------
            connected_space, FreeSpaceModel - free configuration space that is connected to the forward search tree
            non_connected_space, ExtendedFreeSpaceModel - free configuration space that is not yet connected to the forward search tree
            cspace_diameter, float - Diameter of the bounding box spanned by the configuration space limits.
            kwargs, parameters
        """
        super(ProximityRating, self).__init__()
        self._min_connection_chance = self._distance_kernel(cspace_diameter)
        self._min_free_space_chance = self._distance_kernel(cspace_diameter)
        self._parameters = {
            'connected_weight': 10.0,
            'free_space_weight': 5.0,
        }
        self.set_parameters(**kwargs)

    def max_rating(self):
        return self._parameters['connected_weight'] + self._parameters['free_space_weight']

    def update_ratings(self, node):
        """
            Update cached ratings of the branch rooted at node.
        """
        rospy.logdebug('[ProximityRating::update_ratings] Updating ratings')
        self.T(node)

    def get_root_rating(self):
        return self._parameters['free_space_weight']

    def t(self, node):
        if node.is_root():
            node.set_t(self._parameters['free_space_weight'])
            return self._parameters['free_space_weight']
        if not node.has_configuration() and node.is_extendible():
            parent = node.get_parent()
            assert parent is not None
            tN = parent.get_coverage() * parent.get_t()
            node.set_t(tN)
            return tN
        elif not node.has_configuration() and node.is_leaf():
            # TODO: we should actually set this to 0.0 and prune covered useless branches
            minimal_temp = self._min_connection_chance + self._min_free_space_chance
            node.set_t(minimal_temp)
            return minimal_temp
        max_temp = 0.0
        config_id = 0
        for config in node.get_configurations():
            connected_temp = self._parameters['connected_weight'] * self._compute_connection_chance(config)
            free_space_temp = self._parameters['free_space_weight'] * self._compute_free_space_chance(config)
            temp = connected_temp + free_space_temp
            if max_temp < temp:
                node.set_active_configuration(config_id)
                max_temp = temp
            config_id += 1
        node.set_t(max_temp)
        # if not ((node.is_valid() and max_temp >= self._free_space_weight) or not node.is_valid()):
        #     print "WTF Assertion fail here"
        #     import IPython
        #     IPython.embed()
        # TODO this assertion failed once. This indicates a serious bug, but did not manage to reproduce it!
        assert not node.is_valid() or max_temp >= self._parameters['free_space_weight']
        return node.get_t()

    def _compute_connection_chance(self, config):
        (dist, nearest_config) = self._connected_space.get_nearest_configuration(config)
        if nearest_config is None:
            return self._min_connection_chance
        return self._distance_kernel(dist)

    def _compute_free_space_chance(self, config):
        (dist, nearest_config) = self._non_connected_space.get_nearest_configuration(config)
        if nearest_config is None:
            return self._min_free_space_chance
        return self._distance_kernel(dist)

    def _distance_kernel(self, dist):
        return math.exp(-dist)


class SDFIntersectionRating(NodeRating):

    def __init__(self, scene_sdf, robot, link_names, b_include_attached=False,
                 cell_size=0.01):
        """
            Create a new SDFIntersectionRating.
            ---------
            Arguments
            ---------
            scene_sdf, SceneSDF - signed distance field used for intersection computation
            robot, OpenRAVE robot - the robot to plan for (its active manipulator and active DOFs are used)
            link_name, list of strings - names of the robot arm links for which to compute intersection
            b_include_attached, bool - if True, also computes intersection with attached bodies # TODO not supported yet
        """
        super(SDFIntersectionRating, self).__init__()
        self._scene_sdf = scene_sdf
        self._robot_occupancy_tree = RobotOccupancyOctree(cell_size, robot, link_names)
        self._robot = robot
        self._min_connection_chance = 1e-8  # TODO figure sth out here
        self._min_collision_free_chance = 1e-8  # TODO
        self._parameters = {
            'connected_weight': 5.0,
            'collision_weight': 2.0,
            'collision_cost_scale': 1.0,
        }

    def max_rating(self):
        return self._parameters['connected_weight'] + self._parameters['collision_weight']

    def update_ratings(self, node):
        self.T(node)

    def get_root_rating(self):
        return self._parameters['collision_weight']

    def t(self, node):
        if node.is_root():
            node.set_t(self.get_root_rating())
            return self.get_root_rating()
        if not node.has_configuration() and node.is_extendible():
            parent = node.get_parent()
            assert parent is not None
            tN = parent.get_coverage() * parent.get_t()
            node.set_t(tN)
            return tN
        elif not node.has_configuration() and node.is_leaf():
            # TODO figure some reasonable values out here
            minimal_val = self._min_collision_free_chance + self._min_connection_chance
            node.set_t(minimal_val)
            return minimal_val
        max_rating = 0.0
        config_id = 0
        for config in node.get_configurations():
            connected_temp = self._parameters['connected_weight'] * self._compute_connection_chance(config)
            free_space_temp = self._parameters['collision_weight'] * self._compute_collision_value(config)
            temp = connected_temp + free_space_temp
            if max_rating < temp:
                node.set_active_configuration(config_id)
                max_rating = temp
            config_id += 1
        node.set_t(max_rating)
        return node.get_t()

    def _compute_connection_chance(self, config):
        (dist, nearest_config) = self._connected_space.get_nearest_configuration(config)
        if nearest_config is None:
            return self._min_connection_chance
        return self._distance_kernel(dist)

    def _compute_collision_value(self, config):
        # TODO figure out which value to use
        _, _, dc, _ = self._robot_occupancy_tree.compute_intersection(self._robot.GetTransform(),
                                                                      config, self._scene_sdf)
        return self._distance_kernel(numpy.abs(dc), self._parameters["collision_cost_scale"])

    def _distance_kernel(self, dist, w=1.0):
        return math.exp(-w * dist)


class FreeSpaceProximityHierarchyNode(object):
    """
        This class represents a node in the hierarchy built by the FreeSpaceProximitySampler
    """

    def __init__(self, goal_node, config=None, initial_temp=0.0, active_children_capacity=20):
        self._goal_nodes = []
        self._goal_nodes.append(goal_node)
        self._active_goal_node_idx = 0
        self._children = []
        self._child_labels = []
        self._active_children = []
        self._inactive_children = []
        self._t = initial_temp
        self._T = 0.0
        self._T_c = 0.0
        self._T_p = 0.0
        self._num_leaves_in_branch = 1 if goal_node.is_leaf() else 0
        self._active_children_capacity = active_children_capacity
        self._configs = []
        self._configs.append(None)
        self._configs_registered = []
        self._configs_registered.append(True)
        self._parent = None
        # INVARIANT: _configs[0] is always None
        #            _goal_nodes[0] is hierarchy node that has all information
        #            _configs_registered[i] is False iff _configs[i] is valid and new
        #            _goal_nodes[i] and _configs[i] belong together for i > 0
        if config is not None:
            self._goal_nodes.append(goal_node)
            self._configs.append(config)
            self._active_goal_node_idx = 1
            self._configs_registered.append(not goal_node.is_valid())

    def get_T(self):
        return self._T

    def get_t(self):
        return self._t

    def get_T_c(self):
        return self._T_c

    def get_T_p(self):
        return self._T_p

    def set_T(self, value):
        self._T = value
        assert self._T > 0.0

    def set_t(self, value):
        self._t = value
        assert self._t > 0.0 or self.is_leaf() and not self.has_configuration()

    def set_T_p(self, value):
        self._T_p = value

    def set_T_c(self, value):
        self._T_c = value
        assert self._T_c > 0.0

    def update_active_children(self, up_temperature_fn):
        # For completeness, reactivate a random inactive child:
        if len(self._inactive_children) > 0:
            reactivated_child = self._inactive_children.pop()
            up_temperature_fn(reactivated_child)
            self._active_children.append(reactivated_child)

        while len(self._active_children) > self._active_children_capacity:
            p = random.random()
            children_T_cs = numpy.array([1.0 / child.get_T_c() for child in self._active_children])
            prefix_sum = numpy.cumsum(children_T_cs)
            i = numpy.argmax(prefix_sum >= p * children_T_cs[-1])
            # sum_temp = reduce(lambda s, child: s + 1.0 / child.get_T_c(), self._active_children, 0.0)
            # assert sum_temp > 0.0
            # acc = 0.0
            # i = 0
            # while acc < p:
            #     acc += 1.0 / self._active_children[i].get_T_c() * 1.0 / sum_temp
            #     i += 1
            deleted_child = self._active_children[i]
            self._active_children.remove(deleted_child)
            self._inactive_children.append(deleted_child)
            rospy.logdebug('[FreeSpaceProximityHierarchyNode::updateActiveChildren] Removing child with ' +
                           'temperature ' + str(deleted_child.get_T()) + '. It had index ' + str(i))
        assert len(self._children) == len(self._inactive_children) + len(self._active_children)

    def add_child(self, child):
        self._children.append(child)
        self._active_children.append(child)
        self._child_labels.append(child.get_label())
        child._parent = self
        if child.is_leaf():
            self._num_leaves_in_branch += 1
            parent = self._parent
            while parent is not None:
                parent._num_leaves_in_branch += 1
                parent = parent._parent

    def get_num_leaves_in_branch(self):
        return self._num_leaves_in_branch

    def get_max_num_leaves_in_branch(self):
        return self._goal_nodes[0].get_num_possible_leaves()

    def get_max_num_children(self):
        return self._goal_nodes[0].get_num_possible_children()

    def get_coverage(self):
        if self.is_leaf():
            return 1.0
        return self.get_num_children() / float(self.get_max_num_children())

    def get_parent(self):
        return self._parent

    # def get_quality(self):
    #     return self._goal_nodes[0].get_quality()

    def has_children(self):
        return self.get_num_children() > 0

    def get_num_children(self):
        return len(self._children)

    def get_children(self):
        return self._children

    def get_label(self):
        return self._goal_nodes[0].get_label()

    def get_sampled_child_labels(self):
        """
            Returns the labels of the all sampled children of this node.
        """
        return self._child_labels

    def get_active_children(self):
        return self._active_children

    def get_hashable_label(self):
        return self._goal_nodes[0].get_hashable_label()

    def get_configurations(self):
        """ Returns all configurations stored for this hierarchy node."""
        return self._configs[1:]

    def get_valid_configurations(self):
        """ Returns only valid configurations """
        valid_configs = []
        for idx in range(1, len(self._goal_nodes)):
            if self._goal_nodes[idx].is_valid():
                valid_configs.append(self._configs[idx])
        return valid_configs

    def get_depth(self):
        return self._goal_nodes[0].get_depth()

    def is_root(self):
        return self._goal_nodes[0].get_depth() == 0

    def set_active_configuration(self, idx):
        assert idx in range(len(self._goal_nodes) - 1)
        self._active_goal_node_idx = idx + 1

    def get_active_configuration(self):
        if self._active_goal_node_idx in range(len(self._configs)):
            return self._configs[self._active_goal_node_idx]
        else:
            return None

    def has_configuration(self):
        return len(self._configs) > 1

    def get_new_valid_configs(self):
        unregistered_goals = []
        unregistered_approx = []
        for i in range(1, len(self._configs)):
            if not self._configs_registered[i]:
                if self._goal_nodes[i].is_goal():
                    unregistered_goals.append((self._configs[i], self._goal_nodes[i].get_additional_data()))
                else:
                    unregistered_approx.append(self._configs[i])
                self._configs_registered[i] = True
        return unregistered_goals, unregistered_approx

    def is_goal(self):
        return self._goal_nodes[self._active_goal_node_idx].is_goal() and self.is_valid()

    def is_valid(self):
        # b_is_valid = self._goal_nodes[self._active_goal_node_idx].is_valid()
        # TODO had to remove the following assertion. If the grasp optimization is non deterministic
        # TODO it can happen that a result is once invalid and once valid. However, the label_cache
        # TODO should prevent this from happening
        # if not b_is_valid:
        # assert not reduce(lambda x, y: x or y, [x.is_valid() for x in self._goal_nodes], False)
        return self._goal_nodes[self._active_goal_node_idx].is_valid()

    def is_extendible(self):
        return self._goal_nodes[0].is_extendible()

    def is_leaf(self):
        return not self.is_extendible()

    def is_all_covered(self):
        if self.is_leaf():
            return True
        return self.get_num_children() == self.get_max_num_children()

    def get_goal_sampler_hierarchy_node(self):
        return self._goal_nodes[self._active_goal_node_idx]

    def to_sample_data(self, id_num=-1):
        return SampleData(self._configs[self._active_goal_node_idx],
                          data=self._goal_nodes[self._active_goal_node_idx].get_additional_data(),
                          id_num=id_num)

    def add_goal_sample(self, sample):
        sample_config = sample.get_configuration()
        if sample_config is None:
            return
        for config in self._configs[1:]:
            b_config_known = numpy.linalg.norm(config - sample_config) < NUMERICAL_EPSILON
            if b_config_known:
                return
        self._configs.append(sample.get_configuration())
        self._goal_nodes.append(sample)
        self._configs_registered.append(not sample.is_valid())


class LazyHierarchySampler(object):
    """
        A goal hierarchy sampler that utilizes a rating function to guide
        goal sampling from a hierarchy.
    """

    def __init__(self, goal_sampler, rating_function, k=4, num_iterations=10,
                 min_num_iterations=8, b_return_approximates=True, debug_drawer=None):
        self._goal_hierarchy = goal_sampler
        self._k = k
        # TODO decide how to set iterations properly
        self._num_iterations = max(1, goal_sampler.get_max_depth()) * [num_iterations]
        self._min_num_iterations = min_num_iterations
        # TODO set rating function externally
        self._rating_function = rating_function
        self._connected_space = None
        self._non_connected_space = None
        self._debug_drawer = debug_drawer
        self._label_cache = {}
        self._goal_labels = []
        self._root_node = FreeSpaceProximityHierarchyNode(goal_node=self._goal_hierarchy.get_root(),
                                                          initial_temp=self._rating_function.get_root_rating())
        self._b_return_approximates = b_return_approximates

    def clear(self):
        rospy.logdebug('[FreeSpaceProximitySampler::clear] Clearing caches etc')
        self._connected_space = None
        self._non_connected_space = None
        self._label_cache = {}
        self._goal_labels = []
        self._root_node = FreeSpaceProximityHierarchyNode(goal_node=self._goal_hierarchy.get_root(),
                                                          initial_temp=self._rating_function.get_root_rating())
        self._num_iterations = self._goal_hierarchy.get_max_depth() * [self._num_iterations[0]]
        if self._debug_drawer is not None:
            self._debug_drawer.clear()

    def get_num_goal_nodes_sampled(self):
        return len(self._label_cache)

    # def get_quality(self, sample_data):
    #     idx = sample_data.get_id()
    #     node = self._label_cache[self._goal_labels[idx]]
    #     return node.get_quality()

    def set_connected_space(self, connected_space):
        self._connected_space = connected_space
        self._rating_function.set_connected_space(connected_space)

    def set_non_connected_space(self, non_connected_space):
        self._non_connected_space = non_connected_space
        self._rating_function.set_non_connected_space(non_connected_space)

    def set_parameters(self, min_iterations=None, max_iterations=None,
                       use_approximates=None, k=None):
        if min_iterations is not None:
            self._min_num_iterations = min_iterations
        if max_iterations is not None:
            max_iterations = max(self._min_num_iterations, max_iterations)
            self._num_iterations = max(1, self._goal_hierarchy.get_max_depth()) * [max_iterations]
        if use_approximates is not None:
            self._b_return_approximates = use_approximates
        if k is not None:
            self._k = k

    def _get_hierarchy_node(self, goal_sample):
        label = goal_sample.get_hashable_label()
        b_new = False
        hierarchy_node = None
        if label in self._label_cache:
            hierarchy_node = self._label_cache[label]
            hierarchy_node.add_goal_sample(goal_sample)
            rospy.logwarn('[FreeSpaceProximitySampler::_getHierarchyNode] Sampled a cached node!')
        else:
            hierarchy_node = FreeSpaceProximityHierarchyNode(goal_node=goal_sample,
                                                             config=goal_sample.get_configuration())
            self._label_cache[label] = hierarchy_node
            b_new = True
        return hierarchy_node, b_new

    # TODO delete this function?
    # def _filter_redundant_children(self, children):
    #     labeled_children = []
    #     filtered_children = []
    #     for child in children:
    #         labeled_children.append((child.get_hashable_label(), child))
    #     labeled_children.sort(key=lambda x: x[0])
    #     prev_label = ''
    #     for labeledChild in labeled_children:
    #         if labeledChild[0] == prev_label:
    #             continue
    #         filtered_children.append(labeledChild[1])
    #         prev_label = labeledChild[0]
    #     return filtered_children

    def _pick_random_node(self, p, nodes):
        modified_temps = [self._rating_function.T_c(x) for x in nodes]
        acc_temp = sum(modified_temps)
        assert acc_temp > 0.0
        i = 0
        acc = 0.0
        while p > acc:
            acc += modified_temps[i] / acc_temp
            i += 1

        idx = max(i - 1, 0)
        other_nodes = nodes[:idx]
        if idx + 1 < len(nodes):
            other_nodes.extend(nodes[idx + 1:])
        return nodes[idx], other_nodes

    def _update_approximate(self, children):
        for child in children:
            goal_configs, approx_configs = child.get_new_valid_configs()
            assert len(goal_configs) == 0
            for config in approx_configs:
                self._non_connected_space.add_approximate(config)

    def _pick_random_approximate(self):
        random_approximate = self._non_connected_space.draw_random_approximate()
        rospy.logdebug('[FreeSpaceProximitySampler::_pickRandomApproximate] ' + str(random_approximate))
        return random_approximate

    def _add_temporary(self, children):
        for node in children:
            if node.is_valid():
                self._non_connected_space.add_temporary(node.get_valid_configurations())

    def _clear_temporary(self):
        self._non_connected_space.clear_temporary_cache()

    def _pick_random_child(self, node):
        if not node.has_children():
            return None
        node.update_active_children(self._rating_function.update_ratings)
        p = random.random()
        child, _ = self._pick_random_node(p, node.get_active_children())
        return child

    def _should_descend(self, parent, child):
        if child is None:
            return False
        if not child.is_extendible():
            return False
        if parent.is_all_covered():
            return True
        p = random.random()
        tP = self._rating_function.T_p(parent)
        sum_temp = tP + self._rating_function.T_c(child)
        if p <= tP / sum_temp:
            return False
        return True

    def _sample_child(self, parent_node):
        goal_node = parent_node.get_goal_sampler_hierarchy_node()
        depth = parent_node.get_depth()
        num_iterations = int(self._min_num_iterations +
                             parent_node.get_T() / (self._rating_function.max_rating()) *
                             (self._num_iterations[depth] - self._min_num_iterations))
        # num_iterations = max(self._minNumIterations, int(num_iterations))
        assert num_iterations >= self._min_num_iterations
        assert num_iterations <= self._num_iterations[depth]
        self._goal_hierarchy.set_max_iter(num_iterations)
        do_post_opt = depth == self._goal_hierarchy.get_max_depth() - 1
        child_labels = parent_node.get_sampled_child_labels()
        new_goal_node = self._goal_hierarchy.sample_warm_start(hierarchy_node=goal_node, depth_limit=1,
                                                               label_cache=child_labels,
                                                               post_opt=do_post_opt)
        if new_goal_node.is_goal() and new_goal_node.is_valid():
            rospy.logdebug('[FreeSpaceProximitySampler::_sampleChild] We sampled a valid goal here!!!')
        elif new_goal_node.is_valid():
            rospy.logdebug('[FreeSpaceProximitySampler::_sampleChild] Valid sample here!')
        (hierarchy_node, b_new) = self._get_hierarchy_node(new_goal_node)
        if b_new:
            parent_node.add_child(hierarchy_node)
        else:
            assert hierarchy_node.get_hashable_label() == new_goal_node.get_hashable_label()
        return hierarchy_node

    def is_goal(self, sample):
        if sample.get_id() >= 0:
            return True
        return False

    def sample(self):
        current_node = self._root_node
        rospy.logdebug('[FreeSpaceProximitySampler::sample] Starting to sample a new goal candidate' +
                       ' - the lazy way')
        num_samplings = self._k
        b_temperatures_invalid = True
        while num_samplings > 0:
            if self._debug_drawer is not None:
                self._debug_drawer.draw_hierarchy(self._root_node)
            rospy.logdebug('[FreeSpaceProximitySampler::sample] Picking random cached child')
            if b_temperatures_invalid:
                self._rating_function.update_ratings(current_node)
            child = self._pick_random_child(current_node)
            if self._should_descend(current_node, child):
                current_node = child
                b_temperatures_invalid = False
            elif current_node.is_all_covered():
                # There are no children left to sample, sample the null space of the child instead
                # TODO: It depends on our IK solver on what we have to do here. If the IK solver is complete,
                # we do not resample non-goal nodes. If it is not complete, we would need to give them
                # another chance here. Hence, we would also need to set the temperatures of such nodes
                # to sth non-zero
                if child.is_goal():
                    rospy.logwarn('[FreeSpaceProximitySampler::sample] Pretending to sample null space')
                    # TODO actually sample null space here and return new configuration or approx
                    b_temperatures_invalid = True
                num_samplings -= 1
            else:
                new_child = self._sample_child(current_node)
                b_temperatures_invalid = True
                num_samplings -= 1
                goal_configs, approx_configs = new_child.get_new_valid_configs()
                assert len(goal_configs) + len(approx_configs) <= 1
                # if new_child.is_valid() and new_child.is_goal():
                if len(goal_configs) > 0:
                    self._goal_labels.append(new_child.get_hashable_label())
                    return SampleData(config=goal_configs[0][0], data=goal_configs[0][1],
                                      id_num=len(self._goal_labels) - 1)
                # elif new_child.is_valid():
                elif len(approx_configs) > 0:
                    self._non_connected_space.add_approximate(approx_configs[0])
                    # self._computeTemperatures(current_node)

        if self._debug_drawer is not None:
            self._debug_drawer.draw_hierarchy(self._root_node)
        rospy.logdebug('[FreeSpaceProximitySampler::sample] The search led to a dead end. Maybe there is '
                       + 'sth in our approximate cache!')
        if self._b_return_approximates:
            return SampleData(self._pick_random_approximate())
        return SampleData(None)

    def debug_draw(self):
        # nodesToUpdate = []
        # nodesToUpdate.extend(self._rootNode.getChildren())
        # while len(nodesToUpdate) > 0:
        # nextNode = nodesToUpdate.pop()
        # nodesToUpdate.extend(nextNode.getChildren())
        if self._debug_drawer is not None:
            self._rating_function.update_ratings(self._root_node)
            self._debug_drawer.draw_hierarchy(self._root_node)
