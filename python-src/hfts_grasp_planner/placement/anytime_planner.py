import rospy
import numpy as np
import openravepy as orpy


class AnyTimePlacementPlanner:
    """
        This class implements an integrated anytime placement planner.
        This algorithm combines a PlacementGoalSampler as defined in goal_sampler.interfaces
        with an OMPL motion planner.
    """
    class MotionPlanningInformation:
        """
            Struct that stores motion planner, parameters, trajectory for a manipulator.
            Parameters:
                num_goal_samples, int - number of goal samples to query for in every iteration
                num_goal_iterations, int - number of sample iterations to run goal sampler in every iteration
                vel_scale, float - scaling factor for maximum velocity
        """

        def __init__(self, planner_name, manip):
            """
                Construct new MotionPlanningInformation struct.
                ---------
                Arguments
                ---------
                planner_name, string - name of the motion planner
                manip, OpenRAVE manipulator - manipulator to create planner etc for
            """
            env = manip.GetRobot().GetEnv()
            self.params = orpy.Planner.PlannerParameters()
            self.traj = orpy.RaveCreateTrajectory(env, '')
            robot = manip.GetRobot()
            with robot:
                robot.SetActiveManipulator(manip.GetName())
                robot.SetActiveDOFs(manip.GetArmIndices())
                self.params.SetRobotActiveJoints(robot)
            self.planner = orpy.RaveCreatePlanner(env, planner_name)
            self.manip = manip

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
            mplanner = "OMPL_LazyPRM"
            # mplanner = "OMPL_RRTConnect"
        self.goal_sampler = goal_sampler
        self._mplanning_infos = {}  # store separate motion planner for each manipulator
        self._params = {"num_goal_samples": 10, "num_goal_iterations": 1000, "vel_scale": 0.1}
        for manip in manips:
            self._mplanning_infos[manip.GetName()] = AnyTimePlacementPlanner.MotionPlanningInformation(mplanner, manip)
        self._robot = manips[0].GetRobot()
        self.set_parameters(**kwargs)
        self._best_solution = None

    def plan(self, max_iter):
        """
            Plan a new solution to a placement. The algorithm plans from the current robot configuration.
            The target volume etc. need to be specified on the goal sampler directly.
            ---------
            Arguments
            ---------
            max_iter, int - maximum number of iterations to run
            # TODO pass terminal condition to let the outside decide when to terminate, i.e. real anytime
        """
        self._best_solution = None
        # initialize motion planners
        with self._robot:
            for _, mpinfo in self._mplanning_infos.iteritems():
                self._robot.SetActiveDOFs(mpinfo.manip.GetArmIndices())
                mpinfo.params.SetInitialConfig(self._robot.GetActiveDOFValues())
                mpinfo.params.SetGoalConfig(self._robot.GetActiveDOFValues())  # dummy goal
                # mpinfo.params.SetExtraParameters('<reset>True</reset>')
                mpinfo.planner.InitPlan(self._robot, mpinfo.params)
        num_goal_samples = self._params["num_goal_samples"]
        num_goal_iter = self._params["num_goal_iterations"]
        for _ in xrange(max_iter):
            new_goals, num_new_goals = self.goal_sampler.sample(num_goal_samples, num_goal_iter)
            # TODO we could/should plan motions for each manipulator in parallel. For now, instead, plan
            # TODO for one at a time
            if num_new_goals > 0:
                # compute in which order to plan (i.e. which manipulator first)
                manip_goal_pairs = self._compute_planning_order(new_goals)
                for manip_name, _, manip_goals in manip_goal_pairs:
                    # get motion planning info for this manipulator
                    mpinfo = self._mplanning_infos[manip_name]
                    # TODO filter goals that are worse than what we have reached so far (only if we do not plan parallel)
                    # goal_configs = np.array([new_goal.arm_config for new_goal in manip_goals])
                    mpinfo.params.SetGoalConfig(np.array(manip_goals).flat)  # TODO should we always pass all goals?
                    # mpinfo.params.SetExtraParameters('<reset>False</reset>')  # tell motion planner not to reset
                    # TODO set timeout
                    with self._robot:
                        self._robot.SetActiveDOFs(mpinfo.manip.GetArmIndices())
                        mpinfo.planner.InitPlan(self._robot, mpinfo.params)
                        result = mpinfo.planner.PlanPath(mpinfo.traj)
                        print "Try the planner!"
                        import IPython
                        IPython.embed()
                    if result == orpy.PlannerStatus.HasSolution:
                        # save best solution, reuse traj object
                        self._best_solution, mpinfo.traj = mpinfo.traj, orpy.RaveCreateTrajectory(
                            self._robot.GetEnv(), '')
                        # TODO inform goal sampler that some solutions have been reached
                        # TODO update best value
        if self._best_solution is not None:
            self.shortcut_path(self._best_solution)
            self.time_traj(self._best_solution)
            return self._best_solution
        return None

    def shortcut_path(self, traj):
        """
            Shortcut the given path.
        """
        # TODO
        pass

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
                goals is a list of goal configurations. The returned list is sorted
                w.r.t to value
        """
        # compute average score per manipulator
        goal_candidates = [(manip_name, sum([goal.objective_value for goal in goals]) / len(goals),
                            [goal.arm_config for goal in goals]) for manip_name, goals in new_goals.iteritems()]
        # sort based on average score
        goal_candidates.sort(key=lambda x: x[1])
        return goal_candidates

    def set_parameters(self, **kwargs):
        """
            Set parameters
        """
        for key, value in kwargs.iteritems():
            if key in self.params:
                self.params[key] = value
            else:
                rospy.logwarn("Unknown parameter: %s" % key)
