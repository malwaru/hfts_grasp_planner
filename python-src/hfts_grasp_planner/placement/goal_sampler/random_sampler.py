import rospy
import numpy as np
import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
import hfts_grasp_planner.utils as utils
"""
    This module contains the definition of a naive placement goal sampler - purely random sampler.
    The random sampler samples an arpo_hierarchy uniformly at random.
"""


class RandomPlacementSampler(plcmnt_interfaces.PlacementGoalSampler):
    def __init__(self, hierarchy, solution_constructor, validator, objective, manip_names, b_go_to_leaf,
                 b_optimize_constraints, p_descend=0.5, stats_recorder=None):
        """
            Create a new RandomPlacementSampler.
            ---------
            Arguments
            ---------
            hierarchy, interfaces.PlacementHierarchy - hierarchy to sample placements from
            solution_constructor, interfaces.PlacementSolutionConstructor
            validator, interfaces.PlacementValidator
            objective, interfaces.PlacementObjective
            manip_names, list of string - manipulator names
            b_go_to_leaf, bool - If True, the random sampler will only sample leaves of the hierarchy, else
                it descends the hierarchy at a random location and at each depth (that is sampleable) tosses a coin
                to decide whether to descend further, or construct a solution for that depth.
            b_optimize_constraints, bool - If True, tell solution constructor to use projection function to increase the
                chance to hit a valid solution
            p_descend, float - probability to descend if b_go_to_leaf is False
            stats_recorder(optional), statsrecording.GoalSamplingStatsRecorder - recorder for stats
        """
        self._hierarchy = hierarchy
        self._solution_constructor = solution_constructor
        self._validator = validator
        self._objective = objective
        self._manip_names = manip_names
        self._best_reached_goal = None
        self._b_go_to_leaf = b_go_to_leaf
        self._boptimize_constraints = b_optimize_constraints
        self._p_descend = p_descend
        self._stats_recorder = stats_recorder

    def sample(self, num_solutions, max_attempts, b_improve_objective=True):
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
            a dict of PlacementGoals
            num_found_sol, int - The number of found solutions.
        """
        if b_improve_objective and self._best_reached_goal is not None:
            self._validator.set_minimal_objective(self._best_reached_goal.objective_value)
        num_found_solutions = 0
        # store solutions for each manipulator separately
        solutions = {manip_name: [] for manip_name in self._manip_names}
        for num_attempts in xrange(max_attempts):
            # stop if we have sufficient solutions
            if num_found_solutions == num_solutions:
                break
            # sample a random base key, i.e. of a node that we can sample
            key = ()
            while not self._solution_constructor.can_construct_solution(key):
                key = self._hierarchy.get_random_child_key(key)
            # from here on descend depending on what was selected during construction
            child_key = key
            while child_key is not None and (self._b_go_to_leaf or bool(np.random.binomial(1, self._p_descend))):
                key = child_key
                child_key = self._hierarchy.get_random_child_key(key)
            assert(key is not None)
            solution = self._solution_constructor.construct_solution(key, self._boptimize_constraints)
            if self._validator.is_valid(solution, b_improve_objective):
                solution.objective_value = self._objective.evaluate(solution)
                solutions[solution.manip.GetName()].append(solution)
                num_found_solutions += 1
                if self._stats_recorder is not None:
                    self._stats_recorder.register_new_goal(solution)
        rospy.logdebug("Random sampler made %i attempts to sample %i solutions" %
                       (num_attempts + 1, num_found_solutions))
        return solutions, num_found_solutions

    def set_reached_goals(self, goals):
        """
            Inform the placement goal sampler that the given goals have been reached.
            ---------
            Arguments
            ---------
            goals, list of PlacementGoals
        """
        if len(goals) > 0:
            if self._best_reached_goal is not None:
                ovals = np.empty(len(goals) + 1)
                ovals[0] = self._best_reached_goal.objective_value
                ovals[1:] = [g.objective_value for g in goals]
                best_idx = np.argmax(ovals)
                if best_idx != 0:
                    self._best_reached_goal = goals[best_idx - 1]
            else:
                ovals = np.array([g.objective_value for g in goals])
                best_idx = np.argmax(ovals)
                self._best_reached_goal = goals[best_idx]

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
        new_goal, path = self._solution_constructor.locally_improve(goal)
        if len(path) > 0:
            traj = utils.extend_or_traj(traj, path)
            return traj, new_goal
        return traj, goal
