#!/usr/bin/env python
import logging
import itertools
import hfts_grasp_planner.placement.optimization as optimization
import hfts_grasp_planner.external.transformations as transformations
import hfts_grasp_planner.placement.so3hierarchy as so3hierarchy
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf
import numpy as np
import math


class PlacementObjectiveFn(object):
    """
        Implements an objective function for object placement.
        #TODO: should this class be in a different module?
    """
    def __init__(self, env, scene_sdf, vol_approx=0.005):
        """
            Creates a new PlacementObjectiveFn
            @param env - openrave environment
            @param scene_sdf - SceneSDF of the environment
        """
        self._env = env
        self._obj_name = None
        self._model_name = None
        self._kinbody = None
        self._scene_sdf = scene_sdf
        self._kinbody_octree = None
        self._vol_approx = vol_approx

    def set_target_object(self, obj_name, model_name=None):
        """
            Sets the target object to be obj_name.
            Model name denotes the class of object. Assumed to be the same as obj_name if None
        """
        self._obj_name = obj_name
        self._model_name = model_name if model_name is not None else obj_name
        self._kinbody = self._env.GetKinBody(self._obj_name)
        # TODO we could/should disable the object within the scene_sdf if it is in it
        if not self._kinbody:
            raise ValueError("Could not set target object " + obj_name + " because it does not exist")
        self._kinbody_octree = kinbody_sdf.OccupancyOctree(self._vol_approx, self._kinbody)

    def get_target_object(self):
        """
            Returns the currently set target object name.
            May be None, if none set
        """
        return self._obj_name

    def evaluate(self, node):
        """
            Evalutes the objective function for the given node (must be SE3HierarchyNode)
        """
        assert(self._kinbody_octree)
        assert(self._scene_sdf)
        # set the transform to the pose to evaluate
        representative = node.get_representative_value()
        self._kinbody.SetTransform(representative)
        # compute the collision cost for this pose
        _, _, col_val, _ = self._kinbody_octree.compute_intersection(self._scene_sdf)
        # preference_val = np.dot(representative[:3, 1], np.array([0, 0, 1]))
        preference_val = 0.0
        #TODO return proper objective
        return col_val + preference_val

    @staticmethod
    def is_better(val_1, val_2):
        """
            Returns whether val_1 is better than val_2
        """
        return val_1 > val_2


class SE3Hierarchy(object):
    """
        A grid hierarchy defined on SE(3)
    """
    class SE3HierarchyNode(object):
        """
            A representative for a node in the hierarchy
        """
        def __init__(self, cartesian_box, so3_key, depth, hierarchy, global_id=None):
            """
                Creates a new SE3HierarchyNode.
                None of the parameters may be None
                @param bounding_box - cartesian bounding box (min_point, max_point)
                @param so3_key - key of SO(3) hierarchy node
                @param depth - depth of this node
                @param hierarchy - hierarchy this node belongs to
                @param global_id - global id of this node within the hierarchy, None if root
            """
            self._global_id = global_id
            if self._global_id is None:
                self._global_id = ((), (), (), (), ()) 
            self._relative_id = SE3Hierarchy.extract_relative_id(self._global_id)
            self._cartesian_box = cartesian_box
            self._so3_key = so3_key
            self._depth = depth
            self._hierarchy = hierarchy
            self._so3_hierarchy = hierarchy._so3_hierarchy
            self._cartesian_range = cartesian_box[1] - cartesian_box[0]
            self._child_dimensions = self._cartesian_range / self._hierarchy._cart_branching
            self._child_cache = {}

        def get_random_node(self):
            """
                Return a random child node, as required by optimizers defined in
                the optimization module. This function simply calls get_random_child().
            """
            return self.get_random_child()

        def get_random_child(self):
            """
                Returns a randomly selected child node of this node.
                This selection respects the node blacklist of the hierarchy.
                Returns None, if this node is at the bottom of the hierarchy.
            """
            if self._depth == self._hierarchy._max_depth:
                return None
            bfs = self._so3_hierarchy.get_branching_factors(self._depth)
            random_child = np.array([np.random.randint(self._hierarchy._cart_branching),
                                     np.random.randint(self._hierarchy._cart_branching),
                                     np.random.randint(self._hierarchy._cart_branching),
                                     np.random.randint(bfs[0]),
                                     np.random.randint(bfs[1])], np.int)
            # TODO respect blacklist
            return self.get_child_node(random_child)

        def get_random_neighbor(self, node):
            """
                Returns a randomly selected neighbor of the given child node.
                This selection respects the node blacklist of the hierarchy and will extend the neighborhood
                if all direct neighbors are black listed.
                @param node has to be a child of this node.
            """
            random_dir = np.array([np.random.randint(-1, 2),
                                   np.random.randint(-1, 2),
                                   np.random.randint(-1, 2)])
            child_id = node.get_id(relative_to_parent=True)
            max_ids = [self._hierarchy._cart_branching - 1, self._hierarchy._cart_branching - 1, self._hierarchy._cart_branching - 1]
            # TODO respect blacklist
            neighbor_id = np.zeros(5, np.int)
            neighbor_id[:3] = np.clip(child_id[:3] + random_dir, 0, max_ids)
            so3_key = self._so3_hierarchy.get_random_neighbor(node.get_so3_key())
            neighbor_id[3] = so3_key[0][-1]
            neighbor_id[4] = so3_key[1][-1]
            return self.get_child_node(neighbor_id)

        def get_child_node(self, child_id):
            """
                Returns a node representing the child with the specified local id.
                @param child_id - numpy array of type int and length 5 (expected to be in range)
            """
            child_id_key = tuple(child_id)
            if child_id_key in self._child_cache:  # check whether we already have this child
                return self._child_cache[child_id_key]
            # else construct the child
            # construct range for cartesian part
            offset = child_id[:3] * self._child_dimensions
            min_point = np.zeros(3)
            min_point = self._cartesian_box[0] + offset
            max_point = min_point + self._child_dimensions
            global_child_id = SE3Hierarchy.construct_id(self._global_id, child_id_key)
            child_node = SE3Hierarchy.SE3HierarchyNode((min_point, max_point), 
                                                        (global_child_id[3], global_child_id[4]),
                                                        self._depth + 1, self._hierarchy,
                                                        global_id=global_child_id)
            self._child_cache[tuple(child_id_key)] = child_node
            return child_node

        def get_children(self):
            """
                Return a generator that allows to iterate over all children.
            """
            bfs = self._so3_hierarchy.get_branching_factors(self._depth)
            local_keys = itertools.product(range(self._hierarchy._cart_branching),
                                           range(self._hierarchy._cart_branching),
                                           range(self._hierarchy._cart_branching),
                                           range(bfs[0]), range(bfs[1]))
            for lkey in local_keys:
                child = self.get_child_node(lkey)
                yield child

        def get_representative_value(self, rtype=0):
            """
                Returns a point in SE(3) that represents this cell, i.e. the center of this cell
                @param rtype - Type to represent point (0 = 4x4 matrix)
            """
            position = self._cartesian_box[0] + self._cartesian_range / 2.0
            quaternion = self._so3_hierarchy.get_quaternion(self._so3_key)
            if rtype == 0:  # return a matrix
                matrix = transformations.quaternion_matrix(quaternion)
                matrix[:3, 3] = position
                return matrix
            raise RuntimeError("Return types different from matrix are not implemented yet!")

        def get_id(self, relative_to_parent=False):
            """
                Returns the id of this node. By default global id. Local id if relative_to_parent is true
                You should not modify the returned id!
            """
            if relative_to_parent:
                return SE3Hierarchy.extract_relative_id(self._global_id)
            return self._global_id
        
        def get_so3_key(self):
            return self._so3_key

    """
        Implements a hierarchical grid on SE3.
    """
    def __init__(self, bounding_box, cart_branching, depth):
        """
            Creates a new hierarchical grid on SE(3), i.e. R^3 x SO(3)
            @param bounding_box - (min_point, max_point), where min_point and max_point are numpy arrays of length 3
                            describing the edges of the bounding box this grid should cover
            @param cart_branching - branching factor (number of children) for cartesian coordinates, i.e. x, y, z
            @param depth - maximal depth of the hierarchy (can be an integer in {1, 2, 3, 4})
        """
        self._cart_branching = cart_branching
        self._so3_hierarchy = so3hierarchy.SO3Hierarchy()
        self._max_depth = max(0, min(depth, self._so3_hierarchy.max_depth()))
        self._root = self.SE3HierarchyNode(cartesian_box=bounding_box,
                                           so3_key=self._so3_hierarchy.get_root_key(),
                                           depth=0,
                                           hierarchy=self)
        self._blacklist = None  # TODO need some data structure to blacklist nodes

    @staticmethod
    def extract_relative_id(global_id):
        """
            Extracts the local id from the given global id. 
            Note that a local is a tuple (a,b,c,d,e,f) where all elements are integers.
            This is different from a global id!
        """
        depth = len(global_id[0])
        if depth == 0:
            return ()  # local root id is empty
        return tuple((global_id[i][depth - 1] for i in xrange(5)))

    @staticmethod
    def construct_id(parent_id, child_id):
        """
            Constructs the global id for the given child.
            Note that a global id is of type tuple(tuple(int, ...), ..., tuple(int, ...)),
            a local id on the other hand of type tuple(int, ..., int)
            @param parent_id - global id of the parent
            @param child_id - local id of the child
        """
        if parent_id is None:
            return tuple(((x,) for x in child_id))  # global id is of format ((a), (b), (c), (d), (e))
        return tuple((parent_id[i] + (child_id[i],) for i in xrange(5)))  # append elements of child id to individual elements of global id

    def get_root(self):
        return self._root
    
    def get_depth(self):
        return self._max_depth


class PlacementGoalPlanner:
    """This class allows to search for object placements in combination with
        the FreeSpaceProximitySampler.
    """
    def __init__(self, object_io_interface,
                 env, scene_sdf, visualize=False):
        """
            Creates a PlacementGoalPlanner
            @param object_io_interface IOObject Object that handles IO requests
            @param env OpenRAVE environment
            @param scene_sdf SceneSDF of the OpenRAVE environment
            @param visualize If true, the internal OpenRAVE environment is set to be visualized
        """
        self._hierarchy = None
        self._object_io_interface = object_io_interface
        self._env = env
        self._objective_function = PlacementObjectiveFn(env, scene_sdf)
        self._optimizer = optimization.StochasticOptimizer(self._objective_function)
        # self._optimizer = optimization.StochasticGradientDescent(self._objective_function)
        self._placement_volume = None
        self._env = env
        self._parameters = {'cart_branching': 3, 'max_depth': 4, 'num_iterations': 100}  # TODO update
        self._initialized = False

    def set_placement_volume(self, workspace_volume):
        """
            Sets the workspace volume in world frame in which the planner shall search for a placement pose.
            @param workspace_volume - (min_point, max_point), where both are np.arrays of length 3
        """
        self._placement_volume = workspace_volume
        self._initialized = False

    def sample(self, depth_limit, post_opt=True):
        """ Samples a placement configuration from the root level. """
        #TODO
        if not self._initialized:
            self._initialize()
        current_node = self.get_root()
        num_iterations = self._parameters['num_iterations']
        best_val = None
        best_node = None
        for depth in xrange(min(depth_limit, self.get_max_depth())):
            best_val, best_node = self._optimizer.run(current_node, num_iterations)
            current_node = best_node
        return best_node, best_val

    def sample_warm_start(self, hierarchy_node, depth_limit, label_cache=None, post_opt=False):
        """ Samples a placement configuration from the given node on. """
        #TODO
        pass

    def is_goal(self, sampling_result):
        """ Returns whether the given node is a goal or not. """
        # TODO
        pass

    def load_hand(self, hand_path, hand_cache_file, hand_config_file, hand_ball_file):
        """ Does nothing. """
        pass

    def set_object(self, obj_id, model_id=None):
        """ Set the object.
            @param obj_id String identifying the object.
            @param model_id (optional) Name of the model data. If None, it is assumed to be identical to obj_id
        """
        self._objective_function.set_target_object(obj_id, model_id)
        self._initialized = False

    def set_max_iter(self, iterations):
        self._parameters['num_iterations']= iterations

    def get_max_depth(self):
        return self._hierarchy.get_depth()

    def get_root(self):
        if not self._initialized:
            self._initialize()
        return self._hierarchy.get_root()

    def set_parameters(self, **kwargs):
        self._initialized = False
        for (key, value) in kwargs.iteritems():
            self._parameters[key] = value

    def _initialize(self):
        if self._placement_volume is None:
            raise ValueError("Could not intialize as there is no placement volume available")
        if self._objective_function.get_target_object() is None:
            raise ValueError("Could not intialize as there is no placement target object available")
        self._hierarchy = SE3Hierarchy(self._placement_volume,
                                       self._parameters['cart_branching'],  # TODO it makes more sense to provide a resolution instead
                                       self._parameters['max_depth'])
        self._initialized = True
