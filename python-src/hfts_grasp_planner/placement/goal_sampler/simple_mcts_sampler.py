import rospy
import bisect
import numpy as np
import hfts_grasp_planner.utils as utils
import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
"""
    This module contains the definition of a Monte-Carlo-tree-search-based placement
    goal sampler. The MCTS sampler operates on an afr_hierarchy and attempts to smartly
    balance between sampling in parts of the hierarchy that are feasible and those that
    are unexplored yet. The sampler can be applied to incrementally optimize an objective
    function by informing it about reached solutions. Subsequent sampling calls will only return
    solutions that achieve better objective.
"""


class SimpleMCTSPlacementSampler(plcmnt_interfaces.PlacementGoalSampler):
    class MCTSNode(object):
        """
            Stores node-cache information for Monte Carlo Tree search.
        """

        def __init__(self, key, parent):
            """
                Create a new MCTSNode for the hierarchy node with the given key.
            """
            self.num_constructions = 0
            self.num_new_valid_constr = 0
            self.key = key  # hierarchy key
            self.sampleable = False
            self.solutions = []  # list of all solutions that were created for this particular node
            self.branch_solutions = []  # list of all solutions found in this branch
            self.child_gen = None
            self.children = {}
            self.parent = parent
            # members for uct scores
            self.acc_rewards = 0.0  # stores sum of all rewards (excluding objective value based reward)
            self.num_visits = 0  # number of times visited (number of times a reward was observed in this branch)
            # other members
            self.last_uct_value = 0.0  # for debug purposes
            self.last_fup_value = 0.0  # for debug purposes

        def update_rec(self, new_solution, base_reward):
            """
                Back-propagate the new rewards.
                ----------
                Parameters
                ----------
                new_solution, PlacementSolution - new placement solution
                base_reward, float - the constraint relaxation value of a new solution
            """
            self.num_visits += 1
            self.acc_rewards += base_reward
            self.branch_solutions.append(new_solution)
            if self.parent is not None:
                self.parent.update_rec(new_solution, base_reward)

    def __init__(self, hierarchy, solution_constructor, validator, objective, manip_names, parameters=None,
                 debug_visualizer=None, stats_recorder=None):
        """
            Create new MCTS Sampler.
            ---------
            Arguments
            ---------
            hierarchy, interfaces.PlacementHierarchy - hierarchy to sample placements from
            solution_constructor, interfaces.PlacementSolutionConstructor
            validator, interfaces.PlacementValidator
            objective, interfaces.PlacementObjective
            manip_names, list of string - manipulator names
            parameters(optional), dict - dictionary of parameters, see class description
            debug_visualizer(optional), mcts_visualization.MCTSVisualizer - visualizer for MCTS hierarchy
            stats_recorder(optional), statsrecording.GoalSamplingStatsRecorder - stats recorder
        """
        self._hierarchy = hierarchy
        self._solution_constructor = solution_constructor
        self._validator = validator
        self._objective = objective
        self._manip_names = manip_names
        if parameters is None:
            parameters = {'c': 0.5, 'use_proj': True}
        self._parameters = parameters
        self._c = self._parameters["c"]
        self._b_use_projection = self._parameters["proj_type"] != "None"
        self._debug_visualizer = debug_visualizer
        self._root_node = self._create_new_node(None, ())
        self._best_reached_goal = None
        if self._debug_visualizer:
            self._debug_visualizer.add_node(self._root_node)
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
        for i in xrange(max_attempts):
            # stop if we have sufficient solutions
            if num_found_solutions == num_solutions:
                break
            num_found_solutions += self._run_mcts(solutions, b_improve_objective)
        rospy.logdebug("Goal sampling finished. Found %i/%i solutions within %i attempts" %
                       (num_found_solutions, num_solutions, i + 1))
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
        # TODO
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
            # extend path
            traj = utils.extend_or_traj(traj, path)
            # add new_goal to mcts hierarchy
            leaf_key = self._solution_constructor.get_leaf_key(new_goal)
            self._insert_solution(self._root_node, leaf_key, new_goal, 1.0)
            return traj, new_goal
        return traj, goal

    def _insert_solution(self, start_node, key, solution, reward):
        """
            Add the given solution to the node with given key. If the node is not in the hierarchy yet,
            it is created.
            -------
            Returns
            -------
            b_new_sol, bool - True if the solution falls into a new branch of the hierarchy, or
                belongs to a leaf that has been reached before
        """
        b_new_sol = False
        key_path = self._hierarchy.get_path(start_node.key, key)
        if key_path is None:
            raise ValueError("Could not insert node with key %s as descendant of node with key %s" %
                             (str(key), str(start_node.key)))
        node = start_node
        for ckey in key_path:
            if ckey in node.children:
                node = node.children[ckey]
            else:
                b_new_sol = True
                node = self._create_new_node(node, ckey)
        # add solution and propagate reward up
        node.update_rec(solution, reward)
        return b_new_sol

    def _compute_uct_fup(self, node, bcompute_fup):
        """
            Compute uct, and fup scores for the given node.
            The uct scores are UCB1 scores for the children of the given node.
            THe fup score denotes whether a child of the given node should be selected to forcibly be selected.
            ---------
            Arguments
            ---------
            node, MCTSNode - node to compute scores for
            bcompute_fup, bool - if True compute fup score, else return -np.inf for both
            -------
            Returns
            -------
            scores, np.array of float - UCB scores (might be empty) of children
            fup, float - fup score of the node
        """
        fup_score = -np.inf
        uct_scores = np.array([])
        log_visits = 0 if node.num_visits == 0 else np.log(node.num_visits)
        assert(not node.sampleable or node.num_visits > 0)
        # compute uct scores, if there are children
        if len(node.children):
            children = node.children.values()
            acc_rewards = np.array([child.acc_rewards for child in children])
            visits = np.array([child.num_visits for child in node.children.values()])
            avg_rewards = acc_rewards / visits
            uct_scores = avg_rewards + self._c * np.sqrt(2.0 * log_visits / visits)

        # compute fup score if requested
        if bcompute_fup:
            if len(node.children):
                fup_score = np.mean(avg_rewards) + self._c * np.sqrt(2.0 * log_visits / len(node.children))
            else:
                fup_score = np.inf
        return uct_scores, fup_score

    def _pull_fup_arm(self, node):
        """
            Add a new child of node to the MCTS tree.
            ---------
            Arguments
            ---------
            node, MCTSNode - node to add a child from
            -------
            Returns
            -------
            new_node, MCTSNode - new child node
        """
        assert(node.child_gen is not None)
        # try:
        new_child_key = node.child_gen.next()
        while new_child_key in node.children:
            new_child_key = node.child_gen.next()
        return self._create_new_node(node, new_child_key)

    def _create_new_node(self, parent, child_key):
        new_node = SimpleMCTSPlacementSampler.MCTSNode(child_key, parent)
        # new_node.child_gen = self._hierarchy.get_child_key_gen(new_node.key)
        new_node.child_gen = self._hierarchy.get_random_child_key_gen(new_node.key)
        new_node.sampleable = self._solution_constructor.can_construct_solution(new_node.key)
        if parent:
            parent.children[child_key] = new_node
        if self._debug_visualizer:
            self._debug_visualizer.add_node(new_node)
            self._debug_visualizer.render(bupdate_data=False)
        return new_node

    def _monte_carlo_rollout(self, node, b_impr_obj):
        child_key = node.key
        # descend in the hierarchy to a leaf
        while child_key is not None:
            key = child_key
            child_key = self._hierarchy.get_random_child_key(key)
        solution = self._solution_constructor.construct_solution(key, False)
        bvalid = self._validator.is_valid(solution, b_impr_obj)
        if bvalid:
            reward = 1.0
        elif self._hierarchy.is_leaf(node.key):
            reward = 0.0
        else:
            reward = self._solution_constructor.get_constraint_relaxation(solution, True)
        node.update_rec(solution, reward)
        return solution if bvalid else None

    def _run_mcts(self, solutions, b_impr_obj):
        """
            # TODO
        """
        current_node = self._root_node
        # descend in hierarchy until we reach the first unsampled sampleable node
        while not current_node.sampleable or (current_node.num_visits > 0 and not self._hierarchy.is_leaf(current_node.key)):
            # get arms
            b_compute_fup = self._hierarchy.get_num_children(current_node.key) > len(current_node.children)
            uct_scores, fup_score = self._compute_uct_fup(current_node, b_compute_fup)
            # get best performing arm
            best_choice = np.argmax(uct_scores) if len(uct_scores) > 0 else None
            # decide whether to move down in mct or add a new child to mct
            if best_choice is None or fup_score >= uct_scores[best_choice]:
                # if we decide to add a new child, choose this child as new current node
                assert(self._hierarchy.get_num_children(current_node.key) > len(current_node.children))
                current_node = self._pull_fup_arm(current_node)
            else:
                # choose child with best uct score
                current_node = current_node.children.values()[best_choice]
        assert(current_node.sampleable)
        # sample it: compute a new solution, propagate result up
        # this is equal to a Monte Carlo rollout + backpropagation
        new_solution = self._monte_carlo_rollout(current_node, b_impr_obj)
        if self._debug_visualizer:
            self._debug_visualizer.render(bupdate_data=True)
        if new_solution is not None:
            solutions[new_solution.manip.GetName()].append(new_solution)
            if self._stats_recorder is not None:
                self._stats_recorder.register_new_goal(new_solution)
            return 1
        return 0
