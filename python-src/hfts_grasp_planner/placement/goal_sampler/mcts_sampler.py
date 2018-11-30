import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
"""
    This module contains the definition of an Monte-Carlo-tree-search-based placement
    goal sampler. The MCTS sampler operates on an arpo_hierarchy and attempts to smartly
    balance between sampling in parts of the hierarchy that are feasible and those that 
    are unexplored yet. The sampler can be applied to incrementally optimize an objective
    function by informing it about reached solutions. Subsequent sampling calls will only return
    solutions that achieve better objective.
"""


class MCTSPlacementSampler(plcmnt_interfaces.PlacementGoalSampler):
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
        for _ in xrange(max_attempts):
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
            solution = self._solution_constructor.construct_solution(key, False, False)
            bvalid = self._validator.is_valid(solution)
            # in any case evaluate the relaxation
            relaxation = self._validator.get_constraint_relaxation(solution)
            print "Solution is valid: %i, relaxation value: %f" % (bvalid, relaxation)
            if bvalid:
                solution.objective_value = self._objective.evaluate(solution)
                solutions[solution.manip.GetName()].append(solution)
                num_found_solutions += 1
        if num_found_solutions > 0:
            # TODO should we sort solutions here?
            return solutions, num_found_solutions
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
