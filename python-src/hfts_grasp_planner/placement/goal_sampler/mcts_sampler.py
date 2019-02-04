import rospy
import bisect
import numpy as np
import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
"""
    This module contains the definition of a Monte-Carlo-tree-search-based placement
    goal sampler. The MCTS sampler operates on an arpo_hierarchy and attempts to smartly
    balance between sampling in parts of the hierarchy that are feasible and those that
    are unexplored yet. The sampler can be applied to incrementally optimize an objective
    function by informing it about reached solutions. Subsequent sampling calls will only return
    solutions that achieve better objective.
"""


class MCTSPlacementSampler(plcmnt_interfaces.PlacementGoalSampler):
    class MCTSNode(object):
        """
            Stores node-cache information for Monte Carlo Tree search.
        """

        def __init__(self, key, parent):
            """
                Create a new MCTSNode for the hierarchy node with the given key.
            """
            self.key = key
            self.solutions = []  # list of all solutions that were created for this particular node
            self.branch_objectives = []  # stores all objective values found in this branch
            self.acc_rewards = 0.0  # stores sum of all rewards (excluding objective value based reward)
            self.num_visits = 0
            self.child_gen = None
            self.children = {}
            self.parent = parent
            self.last_uct_value = 0.0  # for debug purposes
            self.last_fup_value = 0.0  # for debug purposes
            # for internal use to remember whether we need to update self.branch_objectives
            self._last_min_obj = -float("inf")

        def update_rec(self, obj_value, base_reward):
            """
                Back-propagate the new rewards.
                ----------
                Parameters
                ----------
                obj_value, float - objective value of a new solution
                base_reward, float - the constraint relaxation value of a new solution
            """
            self.num_visits += 1
            self.acc_rewards += base_reward
            self.branch_objectives.append(obj_value)
            if self.parent is not None:
                self.parent.update_rec(obj_value, base_reward)

        def update_objectives(self, min_obj):
            """
                Update the branch objectives list, given that the minimal objective is now min_obj.
                As a result the list self.branch_objectives only contains values > min_obj and it is sorted.
                If min_obj is less than or equal to a value this function has called before with, this is a no-op.
                ---------
                Arguments
                ---------
                min_obj, float - minimal objective
            """
            if min_obj <= self._last_min_obj:
                return
            self.branch_objectives.sort()
            i = bisect.bisect(self.branch_objectives, min_obj)
            self.branch_objectives = self.branch_objectives[i:]
            self._last_min_obj = min_obj

    def __init__(self, hierarchy, solution_constructor, validator, objective, manip_names, parameters=None,
                 debug_visualizer=None):
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
        """
        self._hierarchy = hierarchy
        self._solution_constructor = solution_constructor
        self._validator = validator
        self._objective = objective
        self._manip_names = manip_names
        if parameters is None:
            parameters = {'c': 1.0, 'objective_epsilon': 0.1}
        self._parameters = parameters
        self._c = self._parameters["c"]
        weights = self._objective.get_constraint_weights()
        self._objective_weight = weights[-1]
        self._reward_normalizer = np.sum(weights[:-1])
        self._debug_visualizer = debug_visualizer
        self._root_node = MCTSPlacementSampler.MCTSNode((), None)
        self._root_node.child_gen = self._hierarchy.get_child_key_gen(self._root_node.key)
        self._best_reached_goal = None
        if self._debug_visualizer:
            self._debug_visualizer.add_node(self._root_node)

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

    def _is_sampleable(self, node):
        """
            Return whether (bool) the given node can be queried for a solution.
            A node is sampleable, if it is possible to construct a solution from it and
            it is either a leaf, or has not been visited before.
        """
        return self._solution_constructor.can_construct_solution(node.key) and \
            (self._hierarchy.is_leaf(node.key) or node.num_visits == 0)

    # def _compute_uct(self, node):
    #     """
    #         Compute UCT score for the given node.
    #         ---------
    #         Arguments
    #         ---------
    #         node, MCTSNode - node to compute score for
    #         parent, MCTSNode - parent node
    #         -------
    #         Returns
    #         -------
    #         score, float - UCB score
    #     """
    #     # TODO incorporate objective values
    #     # assert(node.parent is not None)
    #     # assert(node.num_visits > 0)
    #     node.last_uct_value = node.acc_rewards / node.num_visits + \
    #         self._c * np.sqrt(2.0 * np.log(node.parent.num_visits) / node.num_visits)
    #     return node.last_uct_value

    # def _compute_uct_batch(self, node):
    #     """
    #         Compute uct scores for the children of the given node.
    #         ---------
    #         Arguments
    #         ---------
    #         node, MCTSNode - node to compute score for
    #         -------
    #         Returns
    #         -------
    #         scores, np.array of float - UCB scores (might be empty)
    #     """
    #     if node.num_visits:
    #         acc_rewards = np.array([child.acc_rewards for child in node.children])
    #         visits = np.array([child.num_visits for child in node.children])
    #         uct_scores = acc_rewards / visits + self._c * np.sqrt(2.0 * np.log(node.num_visits) / visits)
    #         return uct_scores
    #     return np.array([])

    def _compute_fup_and_uct(self, node, bcompute_fup):
        """
            Compute uct scores for the children of the given node and the node's fup score.
            ---------
            Arguments
            ---------
            node, MCTSNode - node to compute scores for
            bcompute_fup, bool - if True compute fup score, else return 0.0 for it
            -------
            Returns
            -------
            scores, np.array of float - UCB scores (might be empty) of children
            fup, float - fup score of the node
        """
        if len(node.children):
            acc_rewards = np.array([child.acc_rewards for child in node.children.values()])
            if self._best_reached_goal is not None:
                node.update_objectives(self._best_reached_goal.objective_value - self._parameters["objective_epsilon"])
                obj_rewards = len(node.branch_objectives) * self._objective_weight / self._reward_normalizer
                acc_rewards += obj_rewards
            visits = np.array([child.num_visits for child in node.children.values()])
            avg_rewards = acc_rewards / visits
            uct_scores = avg_rewards + self._c * np.sqrt(2.0 * np.log(node.num_visits) / visits)
            if bcompute_fup:
                fup_score = np.mean(avg_rewards) + self._c * np.sqrt(2.0 * np.log(node.num_visits) / len(node.children))
            else:
                fup_score = 0.0
            return uct_scores, fup_score
        return np.array([]), float("inf")
    # def _compute_uct_batch(self, node):
    #     """
    #         Compute uct scores for the children of the given node.
    #     """
    #     values = np.array([(child.acc_rewards, child.num_visits) for child in node.children])
    #     if len(values):
    #         uct_scores = values[:, 0] / values[:, 1] + self._c * np.sqrt(2.0 * np.log(node.num_visits) / values[:, 1])
    #     else:
    #         uct_scores = []
    #     return uct_scores

    # def _compute_uct_batch(self, node):
    #     """
    #         Compute uct scores for the children of the given node.
    #     """
    #     values = np.array([[child.acc_rewards, child.num_visits] for child in node.children]).transpose()
    #     if len(values):
    #         return values[0, :] / values[1, :] + self._c * np.sqrt(2.0 * np.log(node.num_visits) / values[1, :])
    #     return np.array([])

    # def _compute_fup_uct(self, node):
    #     """
    #         Compute UCT score for first play urgency move from the given node.
    #         ---------
    #         Arguments
    #         ---------
    #         node, MCTSNode - node to compute score for
    #         -------
    #         Returns
    #         -------
    #         score, float - UCB score (may be infinite)
    #     """
    #     # TODO incoporate objective values
    #     avg_rewards = np.array([child.acc_rewards / child.num_visits for child in node.children])
    #     num_visits = len(node.children)
    #     if num_visits == 0:
    #         node.last_fup_value = float("inf")
    #     else:
    #         node.last_fup_value = np.mean(avg_rewards) + self._c * np.sqrt(2.0 * np.log(node.num_visits) / num_visits)
    #     return node.last_fup_value

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
        new_node = MCTSPlacementSampler.MCTSNode(child_key, parent)
        new_node.child_gen = self._hierarchy.get_child_key_gen(new_node.key)
        parent.children[child_key] = new_node
        if self._debug_visualizer:
            self._debug_visualizer.add_node(new_node)
            self._debug_visualizer.render(bupdate_data=False)
        return new_node

    def _sample(self, node, b_impr_obj):
        """
            Sample a new solution from the given node.
            ---------
            Arguments
            ---------
            node, MCTSNode - node to compute new solution on
            b_impr_obj, bool - whether objective needs to be improved
            -------
            Returns
            -------
            new_solution, PlacementSolution - newly sampled placement solution
        """
        assert(self._solution_constructor.can_construct_solution(node.key))
        new_solution = self._solution_constructor.construct_solution(node.key, True)
        # first, check whether all other non-objective constraints are fullfilled
        b_is_valid = self._validator.is_valid(new_solution, False)
        if not b_is_valid and not self._hierarchy.is_leaf(node.key):
            base_reward = self._validator.get_constraint_relaxation(new_solution)
        else:
            # TODO decide whether we wanna keep this
            base_reward = 1.0 if b_is_valid else 0.0  # leaves do not get a relaxation reward
            if node.num_visits > 0:
                base_reward /= node.num_visits
        obj_value = self._objective.evaluate(new_solution)
        if b_is_valid:  # if the solution is valid, we credit this to the full subbranch that this solution falls into
            # add the path to the resulting solution
            leaf_key = self._solution_constructor.get_leaf_key(new_solution)
            key_path = self._hierarchy.get_path(node.key, leaf_key)
            for key in key_path:
                if key in node.children:
                    node = node.children[key]
                else:
                    node = self._create_new_node(node, key)
        node.update_rec(obj_value, base_reward)
        node.solutions.append(new_solution)
        if b_is_valid:
            return new_solution
        return None

    def _run_mcts(self, solutions, b_impr_obj):
        """
            # TODO
        """
        current_node = self._root_node
        # descend in hierarchy until we reach the first unsampled sampleable node
        while current_node.children or not self._is_sampleable(current_node):
            # get arms
            # ucts_scores = self._compute_uct_batch(current_node)
            b_compute_fup = self._hierarchy.get_num_children(current_node.key) > len(current_node.children)
            ucts_scores, fup_score = self._compute_fup_and_uct(current_node, b_compute_fup)
            # get best performing arm
            best_choice = np.argmax(ucts_scores) if len(ucts_scores) > 0 else None
            # add an additional arm if there are children that haven't been added to mct yet (first play urgency)
            # if self._hierarchy.get_num_children(current_node.key) > len(current_node.children):
            #     fup_score = self._compute_fup_uct(current_node)
            # else:
            #     fup_score = 0.0
            # decide whether to move down in mct or add a new child to mct
            if best_choice is None or fup_score > ucts_scores[best_choice]:
                # if we decide to add a new child, choose this child as new current node
                assert(self._hierarchy.get_num_children(current_node.key) > len(current_node.children))
                current_node = self._pull_fup_arm(current_node)
            else:
                # choose child with best uct score
                current_node = current_node.children.values()[best_choice]
        assert(self._is_sampleable(current_node))
        # sample it: compute a new solution, propagate result up
        # this is equal to a Monte Carlo rollout + backpropagation
        new_solution = self._sample(current_node, b_impr_obj)
        if self._debug_visualizer:
            self._debug_visualizer.render(bupdate_data=True)
        if new_solution is not None:
            solutions[new_solution.manip.GetName()].append(new_solution)
            return 1
        return 0
