import heapq
import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
"""
    This module contains the definition of a naive placement goal sampler - purely random sampler.
    The random sampler samples an arpo_hierarchy uniformly at random.
"""


class RandomPlacementSampler(plcmnt_interfaces.PlacementGoalSampler):
    def __init__(self, hierarchy, solution_constructor, validator, objective):
        self._hierarchy = hierarchy
        self._solution_constructor = solution_constructor
        self._validator = validator
        self._objective = objective

    def sample(self, num_solutions, max_attempts):
        """
            Sample new solutions.
            ---------
            Arguments
            ---------
            num_solutions, int - number of new solutions to sample
            max_attempts, int - maximal number of attempts (iterations or sth similar)
            -------
            Returns
            -------
            a list of PlacementGoals
        """
        solutions = []
        for _ in xrange(max_attempts):
            # stop if we have sufficient solutions
            if len(solutions) == num_solutions:
                break
            # sample a random key
            key = ()
            child_key = self._hierarchy.get_random_child_key(key)
            while child_key is not None:
                key = child_key
                child_key = self._hierarchy.get_random_child_key(key)
            assert(key is not None)
            solution = self._solution_constructor.construct_solution(key, True, True)
            if self._validator.is_valid(solution):
                value = self._objective.evaluate(solution)
                solutions.append((value, solution))
        solutions.sort(key=lambda x: x[0])
        if len(solutions) > 0:
            return [sol for (_, sol) in solutions]
        return []

    def set_reached_goals(self, goals):
        """
            Inform the placement goal sampler that the given goals have been reached.
            ---------
            Arguments
            ---------
            goals, list of PlacementGoals
        """
        # Nothing to do here
        pass

    def set_reached_configurations(self, manip, configs):
        """
            Inform the placement goal sampler about reached arm configurations for the given manipulator.
            ---------
            Arguments
            ---------
            # TODO nearest neighbor data structure or just numpy array?
        """
        # Nothing to do here
        pass
