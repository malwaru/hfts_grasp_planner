import rospy
import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
"""
    This module contains the definition of a naive placement goal sampler - purely random sampler.
    The random sampler samples an arpo_hierarchy uniformly at random.
"""


class RandomPlacementSampler(plcmnt_interfaces.PlacementGoalSampler):
    def __init__(self, hierarchy, solution_constructor, validator, objective, manip_names):
        self._hierarchy = hierarchy
        self._solution_constructor = solution_constructor
        self._validator = validator
        self._objective = objective
        self._manip_names = manip_names

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
            a dict of PlacementGoals
            num_found_sol, int - The number of found solutions.
        """
        num_found_solutions = 0
        # store solutions for each manipulator separately
        solutions = {manip_name: [] for manip_name in self._manip_names}
        for num_attempts in xrange(max_attempts):
            # stop if we have sufficient solutions
            if num_found_solutions == num_solutions:
                break
            # sample a random key
            key = ()
            child_key = self._hierarchy.get_random_child_key(key)
            while child_key is not None:
                key = child_key
                child_key = self._hierarchy.get_random_child_key(key)
            assert(key is not None)
            solution = self._solution_constructor.construct_solution(key, True, False)
            if self._validator.is_valid(solution):
                solution.objective_value = self._objective.evaluate(solution)
                solutions[solution.manip.GetName()].append(solution)
                num_found_solutions += 1
        rospy.logdebug("Random sampler made %i attempts to sample %i solutions" % (num_attempts, num_found_solutions))
        # TODO should we sort solutions here?
        return solutions, num_found_solutions

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
