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
            # self.solution = None
            self.branch_objectives = []  # stores all objective values found in this branch
            self.num_visits = 0
            self.acc_rewards = 0.0  # stores sum of all rewards (excluding objective value based reward)
            self.obj_reward = 0.0  # stores the reward computed from objective values
            self.child_gen = None
            self.children = []
            self.parent = parent

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

    def __init__(self, hierarchy, solution_constructor, validator, objective, manip_names, c=1.0, objective_weight=1.0/3.0):
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
            c, float - exploration parameter
            objective_weight, float - parameter to weight objective constraint relaxation w.r.t to other constraints
        """
        self._hierarchy = hierarchy
        self._solution_constructor = solution_constructor
        self._validator = validator
        self._objective = objective
        self._manip_names = manip_names
        self._root_node = MCTSPlacementSampler.MCTSNode((), None)
        self._root_node.child_gen = self._hierarchy.get_child_key_gen(self._root_node.key)
        self._c = c
        self._obj_constr_weight = objective_weight

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
            print _
            num_found_solutions += self._run_mcts(solutions)
        return solutions, num_found_solutions

    def set_reached_goals(self, goals):
        """
            Inform the placement goal sampler that the given goals have been reached.
            ---------
            Arguments
            ---------
            goals, list of PlacementGoals
        """
        # TODO
        pass

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

    def _compute_uct(self, node):
        """
            Compute UCT score for the given node.
            ---------
            Arguments
            ---------
            node, MCTSNode - node to compute score for
            parent, MCTSNode - parent node
            -------
            Returns
            -------
            score, float - UCB score
        """
        # TODO incorporate objective values
        assert(node.parent is not None)
        assert(node.num_visits > 0)
        return node.acc_rewards / node.num_visits + \
            self._c * np.sqrt(2.0 * np.log(node.parent.num_visits) / node.num_visits)

    def _compute_fup_uct(self, node):
        """
            Compute UCT score for first play urgency move from the given node.
            ---------
            Arguments
            ---------
            node, MCTSNode - node to compute score for
            -------
            Returns
            -------
            score, float - UCB score (may be infinite)
        """
        # TODO incoporate objective values
        avg_rewards = np.array([child.acc_rewards / child.num_visits for child in node.children])
        num_visits = len(node.children)
        if num_visits == 0:
            return float("inf")
        return np.mean(avg_rewards) + self._c * np.sqrt(2.0 * np.log(node.num_visits) / num_visits)

    def _add_new_child(self, node):
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
        new_node = MCTSPlacementSampler.MCTSNode(new_child_key, node)
        new_node.child_gen = self._hierarchy.get_child_key_gen(node.key)
        node.children.append(new_node)
        # except:
        # node.child_gen = None
        return new_node

    def _sample(self, node):
        """
            Sample a new solution from the given node.
            ---------
            Arguments
            ---------
            node, MCTSNode - node to compute new solution on
            -------
            Returns
            -------
            new_solution, PlacementSolution - newly sampled placement solution
        """
        assert(self._solution_constructor.can_construct_solution(node.key))
        # TODO what about optimization flags?
        new_solution = self._solution_constructor.construct_solution(node.key, True, True)
        b_is_valid = self._validator.is_valid(new_solution)
        if not b_is_valid and not self._hierarchy.is_leaf(node.key):
            base_reward = self._validator.get_constraint_relaxation(new_solution)
        else:
            base_reward = 1.0 if b_is_valid else 0.0  # leaves do not get a relaxation reward
        obj_value = self._objective.evaluate(new_solution)
        # TODO evaluate whether objective constraint is fulfilled
        node.update_rec(obj_value, base_reward)
        print "Sampled solution from node ", node.key
        if b_is_valid:
            return new_solution
        return None

    def _run_mcts(self, solutions):
        """
            # TODO
        """
        current_node = self._root_node
        # descend in hierarchy until we reach the first unsampled sampleable node
        while current_node.children or not self._is_sampleable(current_node):
            # get arms
            ucts_scores = np.array([self._compute_uct(child) for child in current_node.children])
            # get best performing arm
            best_choice = np.argmax(ucts_scores) if len(ucts_scores) > 0 else None
            # add an additional arm if there are children that haven't been added to mct yet (first play urgency)
            if self._hierarchy.get_num_children(current_node.key) > len(current_node.children):
                fup_score = self._compute_fup_uct(current_node)
            else:
                fup_score = 0.0
            # decide whether to move down in mct or add a new child to mct
            if best_choice is None or fup_score > ucts_scores[best_choice]:
                # if we decide to add a new child, choose this child as new current node
                assert(self._hierarchy.get_num_children(current_node.key) > len(current_node.children))
                current_node = self._add_new_child(current_node)
            else:
                # choose child with best uct score
                current_node = current_node.children[best_choice]
        assert(self._is_sampleable(current_node))
        # sample it: compute a new solution, propagate result up
        new_solution = self._sample(current_node)  # this is equal to a Monte Carlo rollout + backpropagation
        if new_solution is not None:
            solutions[new_solution.manip.GetName()].append(new_solution)
            return 1
        return 0
