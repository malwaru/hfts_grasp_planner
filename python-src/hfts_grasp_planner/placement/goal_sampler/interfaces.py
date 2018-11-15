from abc import ABCMeta, abstractmethod
"""
    This module describes the goal sampler interface for placement planning.
"""


class PlacementGoalSampler:
    __metaclass__ = ABCMeta

    class PlacementGoal:
        __metaclass__ = ABCMeta

        def __init__(self, manip, arm_config, obj_tf, key, objective_value, data=None):
            """
                Create a new PlacementGoal.
                ---------
                Arguments
                ---------
                manip - OpenRAVE manipulator this goal is for
                arm_config, numpy array (n,) - arm configuration
                obj_tf, numpy array (4, 4) - pose of the object
                key, object - key information that can be used by the placement goal sampler to identify this goal
                objective_value - 
                data, object - optional additional data
            """
            self.manip = manip
            self.arm_config = arm_config
            self.obj_tf = obj_tf
            self.key = key
            self.data = data

    @abstractmethod
    def sample(self, num_solutions):
        """
            Sample new solutions.
            ---------
            Arguments
            ---------
            num_solutions, int - number of new solutions to sample
            -------
            Returns
            -------
            a list of PlacementGoals
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


class PlacementHierarchy:
    # TODO define an interface that allows exchanging arpo hierarchy easily?
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


class PlacementSolutionConstructor:
    __metaclass__ = ABCMeta

    @abstractmethod
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
        pass


class PlacementValidator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def is_valid(self, solution):
        """
            Return whether the given PlacementSolution is valid.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
        """
        pass

    @abstractmethod
    def get_constraint_relaxation(self, solution):
        """
            Return a relaxation value between [0, 1] that is 0
            if the solution is invalid and goes towards 1 the closer the solution is to
            something valid.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
            -------
            Returns
            -------
            val, float - relaxation value in [0, 1]
        """
        pass


class PlacementObjective:
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, solution):
        """
            Return the objective value for the given solution.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - solution to evaluate
            --------
            Returns
            --------
            val, float - objective value (the smaller the better)
        """
        pass
