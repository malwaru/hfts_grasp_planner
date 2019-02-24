import rospy
import bisect
import numpy as np
import hfts_grasp_planner.utils as utils
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
            self.key = key  # hierarchy key
            self.sampleable = False
            self.solutions = []  # list of all solutions that were created for this particular node
            self.branch_solutions = []  # list of all solutions found in this branch
            self.child_gen = None
            self.children = {}
            self.parent = parent
            # members for uct scores
            self.branch_objectives = []  # stores all objective values found in this branch
            self.acc_rewards = 0.0  # stores sum of all rewards (excluding objective value based reward)
            self.normalized_obj_reward = 0.0  # reward stemming from objectives in this branch
            self.num_visits = 0  # number of times visited (number of times a reward was observed in this branch)
            self.num_explore_visits = 0  # modified visit count for exploration score
            # the number of times a valid solution was constructed in this branch (updated when min objective changes)
            self.num_valid_branch_constr = 0
            # values for resample arm
            self.num_constructions = 0  # number of times we constructed a solution from THIS node (excluding children)
            # number of times a valid solution was constructed from this node (updated when min objective changes)
            self.num_valid_constr = 0
            self.num_new_valid_constr = 0  # number of times constructing a solution from THIS node led to a new valid leaf
            self.num_self_explore_visits = 0  # number of times this node can be resampled for free after objective changed
            # other members
            self.last_uct_value = 0.0  # for debug purposes
            self.last_fup_value = 0.0  # for debug purposes
            # for internal use to remember whether we need to update self.branch_objectives and others
            self._last_min_obj = -float("inf")

        def update_rec(self, new_solution, base_reward, bvalid):
            """
                Back-propagate the new rewards.
                ----------
                Parameters
                ----------
                new_solution, PlacementSolution - new placement solution
                base_reward, float - the constraint relaxation value of a new solution
                bvalid, bool - if True, signals that the solution is feasible and improves on the objective
            """
            self.num_visits += 1
            self.num_explore_visits += 1
            self.acc_rewards += base_reward
            self.branch_solutions.append(new_solution)
            self.num_valid_branch_constr += bvalid
            if new_solution.objective_value is not None:
                self.branch_objectives.append(new_solution.objective_value)
            if self.parent is not None:
                self.parent.update_rec(new_solution, base_reward, bvalid)

        def update_objectives(self, min_obj, objective_weight, reward_normalizer):
            """
                Update the branch objectives list, given that the minimal objective is now min_obj.
                As a result the list self.branch_objectives only contains values > min_obj and it is sorted.
                If min_obj is less than or equal to a value this function has called before with, this is a no-op.
                Additionally, the number of visits is reset to max(1, self.num_visits - self.num_valid_branch_constr)
                and self.num_valid_branch_constr is reset to 0 afterwards.
                ---------
                Arguments
                ---------
                min_obj, float - minimal objective
                objective_weight, float - weight of the objective reward
                reward_normalizer, float - normalizer for objective reward
            """
            if min_obj <= self._last_min_obj:
                return
            # recompute branch ratings based on the objectives that we have -> this can only decrease the exploitation part of uct
            self.branch_objectives.sort()
            i = bisect.bisect(self.branch_objectives, min_obj)
            self.branch_objectives = self.branch_objectives[i:]
            self._last_min_obj = min_obj
            self.normalized_obj_reward = len(self.branch_objectives) * objective_weight / reward_normalizer
            # we want to encourage re-exploration of branches that contain valid solutions as we might be able to
            # improve on the objective from those. For this, we increase the exploration term if there were valid samples in this branch
            # we do this by reducing the visit count used in the exploration term in ucb1
            self.num_explore_visits = max(self.num_explore_visits - self.num_valid_branch_constr, 1)
            self.num_valid_branch_constr = 0
            self.num_self_explore_visits = max(self.num_self_explore_visits - self.num_valid_constr, 1)
            self.num_valid_constr = 0

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
            parameters = {'c': 0.5, 'objective_impr_f': 0.1}
        self._parameters = parameters
        self._c = self._parameters["c"]
        weights = self._objective.get_constraint_weights()
        self._objective_weight = weights[-1]
        self._reward_normalizer = np.sum(weights)
        self._base_reward = np.sum(weights[:-1])
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
            self._insert_solution(self._root_node, leaf_key, new_goal, True, 1.0)
            return traj, new_goal
        return traj, goal

    def _insert_solution(self, start_node, key, solution, bvalid, reward):
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
        node.update_rec(solution, reward, True)
        return b_new_sol

    def _compute_uct_resample_fup(self, node, bcompute_fup):
        """
            Compute uct, resample and fup scores for the given node.
            The uct scores are UCB1 scores for the children of the given node.
            The resample score is a score denoting whether the current node should be resampled.
            THe fup score denotes whether a child of the given node should be selected to forcibly be selected.
            ---------
            Arguments
            ---------
            node, MCTSNode - node to compute scores for
            bcompute_fup, bool - if True compute resample and fup score, else return -np.inf for both
            -------
            Returns
            -------
            scores, np.array of float - UCB scores (might be empty) of children
            resample_score, float - resample score for sampling this node for the nth time
            fup, float - fup score of the node
        """
        resample_score = -np.inf
        fup_score = -np.inf
        uct_scores = np.array([])
        log_visits = 0 if node.num_visits == 0 else np.log(node.num_visits)
        assert(not node.sampleable or node.num_constructions > 0)
        # compute uct scores, if there are children
        if len(node.children):
            children = node.children.values()
            acc_rewards = np.array([child.acc_rewards for child in children])
            if self._best_reached_goal is not None:
                bobjv = self._best_reached_goal.objective_value
                for child in children:
                    child.update_objectives(bobjv - self._parameters["objective_impr_f"] * abs(bobjv),
                                            self._objective_weight, self._reward_normalizer)
            obj_rewards = np.array([child.normalized_obj_reward for child in children])
            acc_rewards += obj_rewards
            visits = np.array([child.num_visits for child in node.children.values()])
            avg_rewards = acc_rewards / visits
            expl_visits = np.array([child.num_explore_visits for child in node.children.values()])
            uct_scores = avg_rewards + self._c * np.sqrt(2.0 * log_visits / expl_visits)

        # compute fup and resample_score if requested
        if bcompute_fup:
            if node.sampleable:  # can we resample?
                assert(node.num_visits > 0)
                # if node.num_visits == 0:  # if have never visited this node, definitely sample
                # return uct_scores, np.inf, fup_score
                fup_score = self._c * np.sqrt(2.0 * log_visits / (1 + len(node.children)))
                # compute resample score
                resample_score = node.num_new_valid_constr / node.num_constructions + \
                    self._c * np.sqrt(2.0 * log_visits / node.num_self_explore_visits)
            else:  # fup is the only option to add new children
                if len(node.children):
                    fup_score = np.mean(avg_rewards) + self._c * np.sqrt(2.0 * log_visits / len(node.children))
                else:
                    fup_score = np.inf
        return uct_scores, resample_score, fup_score

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
        # new_node.child_gen = self._hierarchy.get_child_key_gen(new_node.key)
        new_node.child_gen = self._hierarchy.get_random_child_key_gen(new_node.key)
        new_node.sampleable = self._solution_constructor.can_construct_solution(new_node.key)
        if parent:
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
        assert(node.sampleable)
        bnode_is_leaf = self._hierarchy.is_leaf(node.key)
        new_solution = self._solution_constructor.construct_solution(node.key, True)
        node.num_constructions += 1
        node.num_self_explore_visits += 1
        node.solutions.append(new_solution)
        # first, check whether all other non-objective constraints are fullfilled
        b_is_valid = self._validator.is_valid(new_solution, False)
        # next compute whether the objective improvement constraint is fullfilled
        if b_is_valid:
            obj_value = self._objective.evaluate(new_solution)
            if b_impr_obj and self._best_reached_goal is not None:
                b_improves_obj = obj_value > self._best_reached_goal.objective_value
            else:
                b_improves_obj = True
        else:
            b_improves_obj = False

        if b_is_valid and b_improves_obj:
            reward = self._base_reward
            # if the solution is all valid, we credit this to the full subbranch that this solution falls into
            leaf_key = self._solution_constructor.get_leaf_key(new_solution)
            b_new_solution = self._insert_solution(node, leaf_key, new_solution, True, reward)
            node.num_new_valid_constr += b_new_solution
            node.num_valid_constr += 1
            return new_solution
        elif node.num_constructions == 1 and not bnode_is_leaf:  # constructing from this non-leaf for the first time?
            reward = self._validator.get_constraint_relaxation(new_solution, True)
        else:
            reward = 0.0
        # if we got here, the solution is not valid, and we only report the reward
        node.update_rec(new_solution, reward, False)
        return None

    def _run_mcts(self, solutions, b_impr_obj):
        """
            # TODO
        """
        current_node = self._root_node
        # descend in hierarchy until we reach the first unsampled sampleable node
        while not current_node.sampleable or current_node.num_constructions > 0:
            # get arms
            b_compute_fup = self._hierarchy.get_num_children(current_node.key) > len(current_node.children)
            # uct_scores = scores of children, resample_score = self sample score, fup_score = select child node score
            uct_scores, resample_score, fup_score = self._compute_uct_resample_fup(current_node, b_compute_fup)
            # get best performing arm
            best_choice = np.argmax(uct_scores) if len(uct_scores) > 0 else None
            # decide whether to move down in mct or add a new child to mct
            if best_choice is None or max(resample_score, fup_score) >= uct_scores[best_choice]:
                if resample_score < fup_score:
                    # if we decide to add a new child, choose this child as new current node
                    assert(self._hierarchy.get_num_children(current_node.key) > len(current_node.children))
                    current_node = self._pull_fup_arm(current_node)
                else:
                    # we decided to sample current node itself again, break so that we sample current_node
                    break
            else:
                # choose child with best uct score
                current_node = current_node.children.values()[best_choice]
        assert(current_node.sampleable)
        # sample it: compute a new solution, propagate result up
        # this is equal to a Monte Carlo rollout + backpropagation
        new_solution = self._sample(current_node, b_impr_obj)
        if self._debug_visualizer:
            self._debug_visualizer.render(bupdate_data=True)
        if new_solution is not None:
            solutions[new_solution.manip.GetName()].append(new_solution)
            if self._stats_recorder is not None:
                self._stats_recorder.register_new_goal(new_solution)
            return 1
        return 0
