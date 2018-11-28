import rospy
import numpy as np
import openravepy as orpy
import hfts_grasp_planner.utils as hfts_utils


class RedirectableOMPLPlanner:
    """
        Convenience wrapper around the RedirectableOMPLPlanner from or_ompl.
    """

    def __init__(self, planner_name, manip):
        """
            Create a new instance of a RedirectableOMPLPlanner using the given algorithm for
            the given manipulator.
            ---------
            Arguments
            ---------
            planner_name, string - name of the algorithm
            manip, OpenRAVE Manipulator - manipulator to use
        """
        env = manip.GetRobot().GetEnv()
        self._params = orpy.Planner.PlannerParameters()
        self._traj = orpy.RaveCreateTrajectory(env, '')
        self._robot = manip.GetRobot()
        with self._robot:
            self._robot.SetActiveManipulator(manip.GetName())
            self._robot.SetActiveDOFs(manip.GetArmIndices())
            self._params.SetRobotActiveJoints(self._robot)
        self._ompl_planner = orpy.RaveCreatePlanner(env, planner_name)
        self._simplifier = orpy.RaveCreatePlanner(env, "OMPL_Simplifier")
        self._manip = manip
        self._bplanner_initialized = False  # whether the underlying motion planner has been intialized
        self._self_initialized = False  # whether this object has been initialized
        self._grasped_obj = None

    def setup(self, grasped_obj, start_config=None, time_limit=1.0):
        """
            Reset motion planner to plan from the current or given start configuration.
            ---------
            Arguments
            ---------
            grasped_obj, OpenRAVE Kinbody - grasped kinbody
            start_config(optional), numpy array of shape (n,) with n = #dofs of manipulator
                If provided, planner plans starting from this configuration, else from the current
                configuration of the manipulator.
            time_limit, float - max time limit to plan for
        """
        if start_config is not None:
            self._params.SetInitialConfig(start_config)
        else:
            with self._robot:
                self._robot.SetActiveDOFs(self._manip.GetArmIndices())
                self._params.SetInitialConfig(self._robot.GetActiveDOFValues())
        self._params.SetExtraParameters("<time_limit>%f</time_limit>" % time_limit)
        self._bplanner_initialized = False
        self._self_initialized = True
        self._grasped_obj = grasped_obj

    def plan(self, plcmnt_goals):
        """
            Plan towards the given goals without resetting the internal motion planner.
            In other words: you can call this function multiple times with different goals and
            the planner will reuse as much knowledge as possible.
            ---------
            Arguments
            ---------
            plcmnt_goals, a list of PlacementGoals to plan to
            NOTE: it is assumed that the grasp for all placement goals is the same (TODO)
            time_limit, float - planning time limit in seconds
            -------
            Returns
            -------
            traj, OpenRAVE trajectory - if success, a trajectory, else None
            goal_id, int - index of the goal that was reached. If no solution found, -1
        """
        if not self._self_initialized:
            raise RuntimeError("Can not plan path before setup has been called!")
        goals = np.array([goal.arm_config for goal in plcmnt_goals])
        # TODO for now we assume that for particular manipulator there is always a single grasp
        inv_grasp_tf = hfts_utils.inverse_transform(plcmnt_goals[0].grasp_tf)
        grasp_config = plcmnt_goals[0].grasp_config
        # TODO set timeout
        self._traj = orpy.RaveCreateTrajectory(self._robot.GetEnv(), '')
        if (type(goals) != np.ndarray or len(goals.shape) != 2 or goals.shape[1] != self._manip.GetArmDOF()):
            raise ValueError("Invalid goals input: " + str(goals))
        with self._robot:
            with self._grasped_obj:
                try:
                    # grasp object first
                    hfts_utils.set_grasp(self._manip, self._grasped_obj, inv_grasp_tf, grasp_config)
                    self._robot.SetActiveDOFs(self._manip.GetArmIndices())
                    if not self._bplanner_initialized or not self._supports_goal_reset():
                        self._params.SetGoalConfig(goals.flat)
                        self._ompl_planner.InitPlan(self._robot, self._params)
                        self._bplanner_initialized = True
                    else:
                        goals_string = np.array2string(goals, separator=',').replace('\n', '').replace(' ', '')
                        self._ompl_planner.SendCommand("ResetGoals " + goals_string)
                    # do the actual planning!
                    result = self._ompl_planner.PlanPath(self._traj)
                    if result == orpy.PlannerStatus.HasSolution:
                        # check what goal we planned to
                        return_string = self._ompl_planner.SendCommand("GetReachedGoals")
                        try:
                            reached_goal = int(return_string)
                        except ValueError:
                            raise RuntimeError(
                                "OMPLPlanner function GetReachedGoals returned invalid string: " + return_string)
                        assert(reached_goal >= 0 and reached_goal < goals.shape[0])
                        return self._traj, reached_goal
                    return None, -1
                finally:
                    # always release the object again so others can use it
                    self._robot.Release(self._grasped_obj)

    def simplify(self, traj, goal, time_limit=1.0):
        """
            Simplifies the given path.
            ---------
            Arguments
            ---------
            traj, OpenRAVE trajectory
            goal, PlacementGoal the path leads to (required to extract grasp)
            time_limit, float - time_limit
            -------
            Returns
            -------
            traj, OpenRAVE trajectory, the same object as traj
        """
        if not self._self_initialized:
            raise RuntimeError("Can not simplify path before setup has been called!")
        inv_grasp_tf = hfts_utils.inverse_transform(goal.grasp_tf)
        with self._robot:
            with self._grasped_obj:
                try:
                    hfts_utils.set_grasp(self._manip, self._grasped_obj, inv_grasp_tf, goal.grasp_config)
                    self._robot.SetActiveDOFs(self._manip.GetArmIndices())
                    params = orpy.Planner.PlannerParameters()
                    params.SetExtraParameters("<time_limit>%f</time_limit>" % time_limit)
                    self._simplifier.InitPlan(self._robot, params)
                    self._simplifier.PlanPath(traj)
                finally:
                    self._robot.Release(self._grasped_obj)
        return traj

    def _supports_goal_reset(self):
        return_string = self._ompl_planner.SendCommand("IsSupportingGoalReset")
        try:
            bool_val = bool(int(return_string))
            return bool_val
        except ValueError:
            raise RuntimeError("OMPLPlanner function IsSupportingGoalReset returned invalid string: " + return_string)


class AnyTimePlacementPlanner:
    """
        This class implements an integrated anytime placement planner.
        This algorithm combines a PlacementGoalSampler as defined in goal_sampler.interfaces
        with an OMPL motion planner.
    """

    def __init__(self, goal_sampler, manips, mplanner=None, **kwargs):
        """
            Create a new AnyTimePlacementPlanner.
            ---------
            Arguments
            ---------
            goal_sampler, PlacementGoalSampler - goal sampler to use
            manips, list of OpenRAVE manipulators - manipulators to plan for (it is assumed that they all belong
                to the same robot)
            mplanner, string(optional) - name of the motion planner. See or_ompl description for available choices.
                Default is OMPL_RRTConnect
            kwargs: Any parameter defined in class documentation.

        """
        if mplanner is None:
            # mplanner = "OMPL_LazyPRM"
            # mplanner = "OMPL_LazyPRMstar"
            mplanner = "OMPL_RRTConnect"
            # mplanner = "OMPL_SPARStwo"
        self.goal_sampler = goal_sampler
        self._motion_planners = {}  # store separate motion planner for each manipulator
        self._params = {"num_goal_samples": 10, "num_goal_iterations": 1000, "vel_scale": 0.1}
        for manip in manips:
            self._motion_planners[manip.GetName()] = RedirectableOMPLPlanner(mplanner, manip)
        self._robot = manips[0].GetRobot()
        self.set_parameters(**kwargs)

    def plan(self, max_iter, target_object):
        """
            Plan a new solution to a placement. The algorithm plans from the current robot configuration.
            The target volume etc. need to be specified on the goal sampler directly.
            ---------
            Arguments
            ---------
            max_iter, int - maximum number of iterations to run
            target_object, OpenRAVE Kinbody - the target object to plan the placement for
            # TODO pass terminal condition to let the outside decide when to terminate, i.e. real anytime
            -------
            Returns
            -------
            traj, OpenRAVE trajectory - arm trajectory to a placement goal. None in case of failure.
            goal, PlacementGoal - the placement goal traj leads to. None in case of failure.
        """
        num_goal_samples = self._params["num_goal_samples"]
        num_goal_iter = self._params["num_goal_iterations"]
        best_solution = None  # store tuple (Trajectory, PlacementGoal)
        # initialize motion planners
        for _, planner in self._motion_planners.iteritems():
            planner.setup(grasped_obj=target_object)
        # repeatedly query new goals, and plan motions
        for iter_idx in xrange(max_iter):
            rospy.logdebug("Running iteration %i" % iter_idx)
            connected_goals = []  # store goals that we manage to connect to in this iteration
            # TODO we may have some goals left from a previous iteration, what about those?
            rospy.logdebug("Sampling %i new goals" % num_goal_samples)
            new_goals, num_new_goals = self.goal_sampler.sample(num_goal_samples, num_goal_iter)
            rospy.logdebug("Got %i valid new goals" % num_new_goals)
            # TODO we could/should plan motions for each manipulator in parallel. For now, instead, plan
            # TODO for one at a time
            if num_new_goals > 0:
                # compute in which order to plan (i.e. which manipulator first)
                manip_goal_pairs = self._compute_planning_order(new_goals)
                for manip_name, _, manip_goals in manip_goal_pairs:
                    # get motion planner for this manipulator
                    motion_planner = self._motion_planners[manip_name]
                    # filter goals out that are worse than our current best solution
                    # TODO should we always pass all goals?
                    remaining_goals = self._filter_goals(manip_goals, best_solution)
                    if len(remaining_goals) > 0:
                        traj, goal_id = motion_planner.plan(remaining_goals)
                        if traj is not None:
                            reached_goal = remaining_goals[goal_id]
                            # by invariant a newly reached goal should always have better objective
                            assert(best_solution is None or
                                   best_solution[1].objective_value > reached_goal.objective_value)
                            best_solution = (traj, reached_goal)
                            rospy.logdebug("Found new solution - it has objective value %f" %
                                           best_solution[1].objective_value)
                            connected_goals.append(reached_goal)
            # lastly, inform goal sampler about the goals we reached this round
            self.goal_sampler.set_reached_goals(connected_goals)
        if best_solution is not None:
            planner = self._motion_planners[best_solution[1].manip.GetName()]
            planner.simplify(best_solution[0], best_solution[1])
            self.time_traj(best_solution[0])
            return best_solution
        return None, None

    def time_traj(self, traj):
        """
            Retime the given trajectory.
        """
        with self._robot:
            vel_limits = self._robot.GetDOFVelocityLimits()
            self._robot.SetDOFVelocityLimits(self._params['vel_scale'] * vel_limits)
            orpy.planningutils.RetimeTrajectory(traj, hastimestamps=False)
            self._robot.SetDOFVelocityLimits(vel_limits)
        return traj

    def _compute_planning_order(self, new_goals):
        """
            Compute an order in which to plan motions, given the newly produced goals.
            ---------
            Arguments
            ---------
            new_goals, dict as returned by PlacementGoalSampler.sample
            -------
            Returns
            -------
            list of tuples (manip, value, goals), where manip is OpenRAVE manipulator,
                value is the average objective value of the goals for manip and
                goals is a list of PlacementGoals. The returned list is sorted
                w.r.t to value
        """
        # compute average score per manipulator
        goal_candidates = [(manip_name, sum([goal.objective_value for goal in goals]) / len(goals), goals)
                           for manip_name, goals in new_goals.iteritems() if len(goals) > 0]
        # sort based on average score
        goal_candidates.sort(key=lambda x: x[1])
        return goal_candidates

    def _filter_goals(self, goals, best_solution):
        """
            Filter the given list of goals based on objective value.
            ---------
            Arguments
            ---------
            goals, list of PlacementGoals
            best_solution, tuple (traj, PlacementGoal), where PlacementGoal is the best reached so far. The tuple may be None
            -------
            Returns
            -------
            remaining_goals, list of PlacementGoals (might be empty)
        """
        if best_solution is None:
            return goals
        return filter(lambda x: x.objective_value < best_solution[1].objective_value, goals)

    def set_parameters(self, **kwargs):
        """
            Set parameters
        """
        for key, value in kwargs.iteritems():
            if key in self.params:
                self.params[key] = value
            else:
                rospy.logwarn("Unknown parameter: %s" % key)
