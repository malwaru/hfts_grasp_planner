import numpy as np
from abc import ABCMeta, abstractmethod
"""
    This module describes the goal sampler interface for placement planning.
"""


class PlacementGoalSampler(object):
    __metaclass__ = ABCMeta

    class PlacementGoal(object):
        """
            The numpy arrays in this object are all read only! If you need to modify an array, copy it first.
        """
        __metaclass__ = ABCMeta

        def __init__(self, manip, arm_config, obj_tf, key, objective_value, grasp_tf, grasp_config, data=None, grasp_id = 0):
            """
                Create a new PlacementGoal.
                ---------
                Arguments
                ---------
                manip - OpenRAVE manipulator this goal is for
                arm_config, numpy array (n,) - arm configuration
                obj_tf, numpy array (4, 4) - pose of the object
                key, int - key information that can be used by the placement goal sampler to identify this goal
                objective_value, float - objective value of this solution
                grasp_tf, np.array (4, 4) - eef frame in object frame
                grasp_config, np.array (q,) - q = manip.GetGripperDOF(), hand configuration for grasp
                data, object - optional additional data
            """
            self.manip = manip
            self.arm_config = arm_config
            self.obj_tf = obj_tf
            self.key = key
            self.objective_value = objective_value
            self.grasp_tf = grasp_tf
            self.grasp_config = grasp_config
            self.data = data
            self.sample_num = 0
            self.grasp_id = grasp_id # index in dmg grasp_order list
        
        def get_arm_config(self):
            return self.arm_config
        
        def copy(self):
            """
                Construct a copy of this goal. All elements that are unique to this goal, e.g. arm configuration,
                are deep-copied.
            """
            new_goal = PlacementGoalSampler.PlacementGoal(
                self.manip, np.array(self.arm_config), np.array(
                    self.obj_tf), self.key, self.objective_value, np.array(self.grasp_tf),
                np.array(self.grasp_config), self.data)  # TODO deep copy data
            return new_goal

    @abstractmethod
    def sample(self, num_solutions, max_attempts=1000, b_improve_objective=True):
        """
            Sample new solutions.
            ---------
            Arguments
            ---------
            num_solutions, int - number of new solutions to sample
            max_attempts, int - maximal number of attempts (iterations or sth similar)
            b_improve_objective, bool - if True, requires samples to achieve better objective than
                all reached goals so far
            -------
            Returns
            -------
            a dictionary that maps manipulator name to a list of PlacementGoals
        """
        pass

    @abstractmethod
    def set_reached_goals(self, goals):
        """
            Inform the placement goal sampler that the given goals have been reached.
            ---------
            Arguments
            ---------
            goals, list of PlacementGoals
        """
        pass

    @abstractmethod
    def set_reached_configurations(self, manip, configs):
        """
            Inform the placement goal sampler about reached arm configurations for the given manipulator.
            ---------
            Arguments
            ---------
            # TODO nearest neighbor data structure or just numpy array?
        """
        pass

    @abstractmethod
    def improve_path_goal(self, traj, goal):
        """
            Attempt to extend the given path locally to a new goal that achieves a better objective.
            In case the goal can not be further improved locally, traj and goal is returned.
            ---------
            Arguments
            ---------
            traj, OpenRAVE trajectory - arm trajectory leading to goal
            goal, PlacementGoal - the goal traj leads to
            -------
            Returns
            -------
            traj - extended by a new path segment to a new goal
            new_goal, PlacementGoal - the new goal that achieves a better objective than goal or goal
                if improving objective failed
        """
        pass


class PlacementHierarchy(object):
    # TODO define an interface that allows exchanging afr hierarchy easily?
    @abstractmethod
    def get_child_key_gen(self, key):
        """
            Return a key-generator for the children of the given key.
            ---------
            Arguments
            ---------
            key, tuple - see the implementation's documentation for key description
            -------
            Returns
            -------
            generator of tuple of int - the generator produces children of the node with key key
                If there are no children, None is returned
        """
        pass

    @abstractmethod
    def get_random_child_key(self, key):
        """
            Return a random key of a child of the given key.
            ---------
            Arguments
            ---------
            key, tuple - see the implementation's documentation for key description
            -------
            Returns
            -------
            child key, tuple - a randomly generated key that describes a child of the input key.
                If there is no child defined, None is returned
        """
        pass

    @abstractmethod
    def get_minimum_depth_for_construction(self):
        """
            Return the minimal depth, i.e. length of a key, for which it is possible
            to construct a solution.
        """
        pass

    @abstractmethod
    def is_leaf(self, key):
        """
            Return whether the given key corresponds to a leaf node.
        """
        pass

    @abstractmethod
    def get_num_children(self, key):
        """
            Return the total number of possible children for the given key.
        """
        pass

    @abstractmethod
    def get_path(self, key_a, key_b):
        """
            Return a list of keys from key_a to key_b.
            If key_b is a descendant of key_a, this function returns a list
            [key_1, ..., key_b], where key_i is the parent of key_(i + 1).
            If key_b is not a descendant of key_a, None is returned.
            ---------
            Arguments
            ---------
            key_a, tuple - see the implentation's documentation for key description
            key_b, tuple - see the implentation's documentation for key description
            -------
            Returns
            -------
            path, list of tuple - each element a key, None if key_b is not within a branch rooted at key_a.
        """
        pass


class PlacementGoalConstructor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def construct_solution(self, key, b_optimize_constraints=False):
        """
            Construct a new PlacementGoal from a hierarchy key.
            ---------
            Arguments
            ---------
            key, object - a key object that identifies a node in a PlacementHierarchy
            boptimize_constraints, bool - if True, the solution constructor may put additional computational
                effort into computing a valid solution, e.g. some optimization of a constraint relaxation
            -------
            Returns
            -------
            PlacementGoal sol, a placement solution for the given key
        """
        pass

    @abstractmethod
    def can_construct_solution(self, key):
        """
            Return whether it is possible to construct a solution from the given (partially defined) key.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_num_construction_calls(self, b_reset=True):
        """
            Return statistics on how many times construct_solution has been called.
            ---------
            Arguments
            ---------
            b_reset, bool - if True, reset counter
            -------
            Returns
            -------
            num_calls, int - number of times construct_solution has been called
        """
        pass

    @abstractmethod
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
        pass


class PlacementValidator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_minimal_objective(self, val):
        """
            Sets the minimal objective that a placement needs to achieve in order to be considered valid.
            ---------
            Arguments
            ---------
            val, float - minimal objective value
        """
        pass

    @abstractmethod
    def is_valid(self, solution, b_improve_objective):
        """
            Return whether the given PlacementGoal is valid.
            ---------
            Arguments
            ---------
            solution, PlacementGoal - solution to evaluate
            b_improve_objective, bool - If True, the solution has to be better than the current minimal objective.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_constraint_weights(self):
        """
            Return the weight factors for all constraint relaxations.
            -------
            Returns
            -------
            weights, np.array of length (n,) - weights of constraint relaxations. weights[-1] is guaranteed to be
                the objective_constraint weight
        """
        pass

    @abstractmethod
    def get_num_validity_calls(self, b_reset=True):
        """
            Return statistics on how many times is_valid has been called.
            ---------
            Arguments
            ---------
            b_reset, bool - if True, reset counter
            -------
            Returns
            -------
            num_calls, int - number of times the validity check has been performed
        """
        pass

    @abstractmethod
    def get_num_relaxation_calls(self, b_reset=True):
        """
            Return statistics on how many times get_constraint_relaxation has been called.
            ---------
            Arguments
            ---------
            b_reset, bool - if True, reset counter
            -------
            Returns
            -------
            num_calls, int - number of times the validity check has been performed
        """
        pass


class PlacementObjective(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, solution):
        """
            Return the objective value for the given solution.
            ---------
            Arguments
            ---------
            solution, PlacementGoal - solution to evaluate
                solution.obj_tf must not be None
            --------
            Returns
            --------
            val, float - objective value (the larger the better)
        """
        pass

    @abstractmethod
    def get_num_evaluate_calls(self, b_reset=True):
        """
            Return statistics on how many times evaluate has been called.
            ---------
            Arguments
            ---------
            b_reset, bool - if True, reset counter
            -------
            Returns
            -------
            num_calls, int - number of times the evaluate function has been called
        """
        pass
