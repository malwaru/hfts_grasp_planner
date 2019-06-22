import rospy
import numpy as np
import openravepy as orpy
import hfts_grasp_planner.utils as hfts_utils
import time
import random


class MGMotionPlanner(object):
    """
        Wrapper around OpenRAVE plugin for multi-grasp motion planning.
    """

    def __init__(self, algorithm_name, manip, vel_factor=0.4):
        """
            Create a new instance of a multi-grasp motion planner using the given
            algorithm for the given manipulator.
            ---------
            Arguments
            ---------
            planner_name, string - name of the algorithm.
                Valid choices: SequentialMGBiRRT, ParallelMGBiRRT
            manip, OpenRAVE Manipulator - manipulator to use
            vel_factor, float - percentage of maximum velocity for trajectories
        """
        self._manip = manip
        self._robot = manip.GetRobot()
        self._env = self._robot.GetEnv()
        self._planner_interface = orpy.RaveCreateModule(self._env, algorithm_name)
        if self._planner_interface == None:
            raise ValueError("Could not create planner with name %s" % algorithm_name)
        self._simplifier = orpy.RaveCreatePlanner(self._env, "OMPL_Simplifier")
        self._initialized = False  # whether this object has been initialized
        self._grasped_obj = None
        self._known_grasps = None
        self._dof = 0
        self._vel_factor = vel_factor
        self._goals = {}

    def setup(self, grasped_obj, start_config=None):
        """
            Reset motion planner to plan from the current or given start configuration.
            ---------
            Arguments
            ---------
            grasped_obj, OpenRAVE Kinbody - grasped kinbody
            start_config(optional), numpy array of shape (n,) with n = #dofs of manipulator
                If provided, planner plans starting from this configuration, else from the current
                configuration of the manipulator.
        """
        with self._robot:
            self._robot.SetActiveManipulator(self._manip.GetName())
            self._robot.SetActiveDOFs(self._manip.GetArmIndices())
            if start_config is not None:
                self._robot.SetActiveDOFValues(start_config)
            self._initialized = True
            self._grasped_obj = grasped_obj
            self._planner_interface.SendCommand("initPlan %i %i" %
                                                (self._robot.GetEnvironmentId(), self._grasped_obj.GetEnvironmentId()))
            self._known_grasps = set()
            self._goals = {}
            self._dof = self._robot.GetActiveDOF()

    def plan(self, time_limit=1.0):
        """
            Plan towards the current goals without resetting the internal motion planner.
            This function returns either until the time limit is reached or new paths have been
            found. In case the underlying algorithm is parallized, multiple new paths may be
            returned.
            ---------
            Arguments
            ---------
            time_limit, float - planning time limit in seconds
            -------
            Returns
            -------
            trajs, list of OpenRAVE trajectory - list of newly computed trajectories 
                (storing only paths, no velocities) 
            goals, list of PlacementGoals - list of goals that the trajectories lead to
        """
        command_str = "plan %f" % time_limit
        result_str = self._planner_interface.SendCommand(command_str)
        if len(result_str) > 0:
            ids = map(int, result_str.split(' '))
            trajs = []
            reached_goals = []
            for idx in ids:
                path_str = self._planner_interface.SendCommand("getPath " + str(idx))
                reached_goals.append(self._goals[idx])
                self._goals.pop(idx)
                # need to parse path now
                path_lines = path_str.splitlines()
                path = []
                for l in path_lines:
                    path.append(np.array(map(float, l.split(" "))))
                with self._robot:
                    self._robot.SetActiveDOFs(self._manip.GetArmIndices())
                    traj = hfts_utils.path_to_trajectory(self._robot, path, bvelocities=False)
                    trajs.append(traj)
            return trajs, reached_goals
        return [], []

    def addGoals(self, goals):
        """
            Add the given list of new goals.
            ---------
            Arguments
            ---------
            goals, a list of PlacementGoals to plan to. These can be for different grasps.
        """
        # first check for new grasps
        for g in goals:
            if g.grasp_id not in self._known_grasps:
                command_str = "addGrasp %i" % g.grasp_id
                command_str += " %f %f %f" % (g.grasp_tf[0, 3], g.grasp_tf[1, 3], g.grasp_tf[2, 3])
                gquat = orpy.quatFromRotationMatrix(g.grasp_tf)
                command_str += " %f %f %f %f" % (gquat[0], gquat[1], gquat[2], gquat[3])
                for v in g.grasp_config:
                    command_str += " %f" % v
                self._planner_interface.SendCommand(command_str)
                self._known_grasps.add(g.grasp_id)
            # next add goal
            command_str = "addGoal %i %i" % (g.key, g.grasp_id)
            for v in g.arm_config:
                command_str += " %f" % v
            self._planner_interface.SendCommand(command_str)
            # finally add it to the set of known goals
            self._goals[g.key] = g

    def removeGoals(self, goals):
        """
            Remove the given list of goals.
            ---------
            Arguments
            ---------
            goals, a list of PlacementGoals to remove from the plan, i.e. to stop planning to.
        """
        goal_ids = [g.key for g in goals if g.key in self._goals]
        if len(goal_ids) > 0:
            command_str = "removeGoals " + str(goal_ids).replace('[', '').replace(']', '').replace(',', '')
            self._planner_interface.SendCommand(command_str)
            for g in goal_ids:
                self._goals.pop(g)


class PathSimplifier(object):
    """
        Wrapper around OR_OMPL_Simplifier.
    """

    def __init__(self, manip):
        """
            Create a new instance of a path simplifier for the given manipulator.
            ---------
            Arguments
            ---------
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
        self._simplifier = orpy.RaveCreatePlanner(env, "OMPL_Simplifier")
        self._manip = manip

    def simplify(self, traj, goal, grasped_obj, time_limit=1.0):
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
        inv_grasp_tf = hfts_utils.inverse_transform(goal.grasp_tf)
        with self._robot:
            with grasped_obj:
                try:
                    hfts_utils.set_grasp(self._manip, grasped_obj, inv_grasp_tf, goal.grasp_config)
                    self._robot.SetActiveDOFs(self._manip.GetArmIndices())
                    params = orpy.Planner.PlannerParameters()
                    params.SetExtraParameters("<time_limit>%f</time_limit>" % time_limit)
                    self._simplifier.InitPlan(self._robot, params)
                    self._simplifier.PlanPath(traj)
                finally:
                    self._robot.Release(grasped_obj)
        return traj


class MGAnytimePlacementPlanner(object):
    """
        This class implements an integrated anytime placement planner.
        This algorithm combines a PlacementGoalSampler as defined in goal_sampler.interfaces
        with a multi-grasp motion planner, i.e. a motion planner that can plan motions under different grasps.
        Parameters:
            num_goal_samples, int  - number of goal samples to query for in every iteration
            num_goal_iterations, int  - max number of iterations the goal sampler can run in each iteration
            vel_scale, float - percentage of max velocity
            mp_timeout, float - computation time the motion planner for each manipulator has in each iteration
    """

    def __init__(self, goal_sampler, manips, mplanner=None, stats_recorder=None, **kwargs):
        """
            Create a new MGAnytimePlacementPlanner.
            ---------
            Arguments
            ---------
            goal_sampler, PlacementGoalSampler - goal sampler to use
            manips, list of OpenRAVE manipulators - manipulators to plan for (it is assumed that they all belong
                to the same robot)
            mplanner, string(optional) - name of the motion planner. Valid choices: SequentialMGBiRRT, ParallelMGBiRRT
            kwargs: Any parameter defined in class documentation.
        """
        if mplanner is None:
            mplanner = "ParallelMGBiRRT"
        self.goal_sampler = goal_sampler
        self._motion_planners = {}  # store separate motion planner for each manipulator
        self._simplifiers = {}  # same for path simplifiers
        self._params = {"num_goal_samples": 2, "num_goal_iterations": 50, "vel_scale": 0.1,
                        "mp_timeout": 1.0}
        if mplanner == "ParallelMGBiRRT":
            self._params["mp_timeout"] = 0.0
        for manip in manips:
            self._motion_planners[manip.GetName()] = MGMotionPlanner(mplanner, manip)
            self._simplifiers[manip.GetName()] = PathSimplifier(manip)
        self._robot = manips[0].GetRobot()
        self.set_parameters(**kwargs)
        self.solutions = []
        self._stats_recorder = stats_recorder

    def plan(self, timeout, target_object):
        """
            Plan a new solution to a placement. The algorithm plans from the current robot configuration.
            The target volume etc. need to be specified on the goal sampler directly.
            ---------
            Arguments
            ---------
            timeout, float - maximum time
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
        self.solutions = []
        best_solution = None  # store tuple (Trajectory, PlacementGoal)
        goal_set = {}  # stores for each manipuluator the current set of goals
        # initialize motion planners
        for _, planner in self._motion_planners.iteritems():
            planner.setup(grasped_obj=target_object)
        # repeatedly query new goals, and plan motions
        start_time = time.time()
        iter_idx = 0
        while time.time() - start_time < timeout:
            rospy.logdebug("Running iteration %i" % iter_idx)
            connected_goals = []  # store goals that we manage to connect to in this iteration
            # sample new goals
            rospy.logdebug("Sampling %i new goals" % num_goal_samples)
            new_goals, num_new_goals = self.goal_sampler.sample(num_goal_samples, num_goal_iter, True)
            rospy.logdebug("Got %i valid new goals" % num_new_goals)
            self._merge_goal_sets(goal_set, new_goals)
            # add new goals
            for manip_name, planner in self._motion_planners.iteritems():
                planner.addGoals(new_goals[manip_name])
            # plan
            for manip_name, planner in self._motion_planners.iteritems():
                # filter goals out that are worse than our current best solution
                current_goals = goal_set[manip_name]
                current_goals, goals_to_remove = self._filter_goals(current_goals, best_solution)
                goal_set[manip_name] = current_goals
                planner.removeGoals(goals_to_remove)
                if len(current_goals) > 0:
                    trajs, reached_goals = planner.plan(self._params["mp_timeout"])
                    if len(trajs) > 0:
                        # select the reached goal with maximal objective value
                        idx = np.argmax([g.objective_value for g in reached_goals])
                        reached_goal = reached_goals[idx]
                        traj = trajs[idx]
                        # locally improve this goal
                        traj, improved_goal = self.goal_sampler.improve_path_goal(traj, reached_goal)
                        self.solutions.append((traj, improved_goal))
                        best_solution = (traj, improved_goal)
                        rospy.logdebug("Found new solution - it has objective value %f" %
                                       best_solution[1].objective_value)
                        connected_goals.extend(reached_goals)
                        connected_goals.append(improved_goal)
                        if self._stats_recorder:
                            # record both reached and improved goal (before and after local optimization)
                            self._stats_recorder.register_new_solution(reached_goal)
                            if reached_goal != improved_goal:
                                self._stats_recorder.register_new_solution(improved_goal)
            # lastly, inform goal sampler about the goals we reached this round
            self.goal_sampler.set_reached_goals(connected_goals)
            iter_idx += 1
        if best_solution is not None:
            simplifier = self._simplifiers[best_solution[1].manip.GetName()]
            simplifier.simplify(best_solution[0], best_solution[1], target_object)
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

    def get_solution(self, i, bsimplify=True):
        """
            Return the ith trajectory found. Optionally, simplify the solution.
        """
        if i >= len(self.solutions):
            return None, None
        if bsimplify:
            simplifier = self._simplifiers[self.solutions[i][1].manip.GetName()]
            simplifier.simplify(self.solutions[i][0], self.solutions[i][1])
        self.time_traj(self.solutions[i][0])
        return self.solutions[i]

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
            goals_to_keep, list of PlacementGoals (might be empty)
            goals_to_remove, list of PlacementGoals (might be emtpy)
        """
        if best_solution is None:
            return goals, []
        goals_to_keep = [x for x in goals if x.objective_value > best_solution[1].objective_value]
        goals_to_remove = [x for x in goals if x.objective_value <= best_solution[1].objective_value]
        return goals_to_keep, goals_to_remove

    def _merge_goal_sets(self, old_goals, new_goals):
        """
            Merge new_goals into old_goals.
            ---------
            Arguments
            ---------
            old_goals, dict - manip_name -> list of goals
            new_goals, dict - manip_name -> list of goals
            -------
            Returns
            -------
            old_goals
        """
        for (key, goals) in new_goals.iteritems():
            if key in old_goals:
                old_goals[key].extend(goals)
            else:
                old_goals[key] = goals
        return old_goals

    def set_parameters(self, **kwargs):
        """
            Set parameters
        """
        for key, value in kwargs.iteritems():
            if key in self._params:
                self._params[key] = value
            else:
                rospy.logwarn("Unknown parameter: %s" % key)


class DummyPlanner(object):
    """
        Dummy placement planner that pretends to plan motions.
        It samples several goals, and then randomly decides that one of them was reached by a motion planner.
        Subsequently, the goal sampler is informed about this and queried again.
    """

    def __init__(self, goal_sampler, num_goal_samples=10, num_goal_iterations=10, stats_recorder=None):
        """
            num_goal_samples, int - number of goal samples the goal sampler should acquire in each iteration
            num_goal_trials, int - total number of trials the goal sampler has to do so in each iteration
        """
        self._goal_sampler = goal_sampler
        self._stats_recorder = stats_recorder
        self.num_goal_samples = num_goal_samples
        self.num_goal_iterations = num_goal_iterations

    def plan(self, timeout, target_object):
        """
            Run pseudo planning.
            ---------
            Arguments
            ---------
            timeout, float - maximum time
            target_object, Kinbody - body to place
        """
        objectives = []
        solutions = []
        goal_set = {}
        best_solution = None
        start_time = time.time()
        while time.time() - start_time < timeout:
            new_goals, _ = self._goal_sampler.sample(self.num_goal_samples, self.num_goal_iterations, True)
            self._check_objective_invariant(new_goals, best_solution)
            goal_set = self._merge_goal_sets(goal_set, new_goals)
            if len(goal_set) > 0:
                all_sols = []
                for (_, sols) in goal_set.iteritems():
                    all_sols.extend(sols)
                all_sols = self._filter_goals(all_sols, best_solution)
                if len(all_sols) > 0:
                    selected_goal = random.choice(all_sols)
                    self._goal_sampler.set_reached_goals([selected_goal])
                    best_solution = selected_goal
                    objectives.append(best_solution.objective_value)
                    solutions.append(best_solution)
                    if self._stats_recorder:
                        self._stats_recorder.register_new_solution(selected_goal)

        return objectives, solutions

    def _check_objective_invariant(self, goals, best_sol):
        if best_sol is None:
            return
        for (key, goal_set) in goals.iteritems():
            for goal in goal_set:
                assert(goal.objective_value >= best_sol.objective_value)

    def _merge_goal_sets(self, old_goals, new_goals):
        """
            Merge new_goals into old_goals.
            ---------
            Arguments
            ---------
            old_goals, dict - manip_name -> list of goals
            new_goals, dict - manip_name -> list of goals
            -------
            Returns
            -------
            old_goals
        """
        for (key, goals) in new_goals.iteritems():
            if key in old_goals:
                old_goals[key].extend(goals)
            else:
                old_goals[key] = goals
        return old_goals

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
        return [x for x in goals if x.objective_value > best_solution.objective_value]


class RedirectableOMPLPlanner(object):
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
        if self._ompl_planner is None:
            raise ValueError("Could not create OMPL planner with name %s" % planner_name)
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
                    rospy.logdebug("Querying motion planner with %i goals" % goals.shape[0])
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


class AnyTimePlacementPlanner(object):
    """
        This class implements an integrated anytime placement planner.
        This algorithm combines a PlacementGoalSampler as defined in goal_sampler.interfaces
        with an OMPL motion planner.
        Parameters:
            num_goal_samples, int  - number of goal samples to query for in every iteration
            num_goal_iterations, int  - max number of iterations the goal sampler can run in each iteration
            vel_scale, float - percentage of max velocity
            mp_timeout, float - computation time each motion planner has in each iteration
    """

    def __init__(self, goal_sampler, manips, mplanner=None, stats_recorder=None, **kwargs):
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
            # mplanner = "OMPL_RRTConnect"
            mplanner = "OMPL_RedirectableRRTConnect"
            # mplanner = "OMPL_SPARStwo"
        self.goal_sampler = goal_sampler
        self._motion_planners = {}  # store separate motion planner for each manipulator
        self._params = {"num_goal_samples": 2, "num_goal_iterations": 50, "vel_scale": 0.1,
                        "mp_timeout": 5.0}
        for manip in manips:
            self._motion_planners[manip.GetName()] = RedirectableOMPLPlanner(mplanner, manip)
        self._robot = manips[0].GetRobot()
        self.set_parameters(**kwargs)
        self.solutions = []
        self._stats_recorder = stats_recorder

    def plan(self, timeout, target_object):
        """
            Plan a new solution to a placement. The algorithm plans from the current robot configuration.
            The target volume etc. need to be specified on the goal sampler directly.
            ---------
            Arguments
            ---------
            timeout, float - maximum time
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
        self.solutions = []
        best_solution = None  # store tuple (Trajectory, PlacementGoal)
        goal_set = {}
        # initialize motion planners
        for _, planner in self._motion_planners.iteritems():
            planner.setup(grasped_obj=target_object, time_limit=self._params["mp_timeout"])
        # repeatedly query new goals, and plan motions
        start_time = time.time()
        iter_idx = 0
        while time.time() - start_time < timeout:
            rospy.logdebug("Running iteration %i" % iter_idx)
            connected_goals = []  # store goals that we manage to connect to in this iteration
            rospy.logdebug("Sampling %i new goals" % num_goal_samples)
            new_goals, num_new_goals = self.goal_sampler.sample(num_goal_samples, num_goal_iter, True)
            rospy.logdebug("Got %i valid new goals" % num_new_goals)
            goal_set = self._merge_goal_sets(goal_set, new_goals)
            # TODO we could/should plan motions for each manipulator in parallel. For now, instead, plan
            # TODO for one at a time
            if len(goal_set) > 0:
                # compute in which order to plan (i.e. which manipulator first)
                manip_goal_pairs = self._compute_planning_order(goal_set)
                for manip_name, _, manip_goals in manip_goal_pairs:
                    # get motion planner for this manipulator
                    motion_planner = self._motion_planners[manip_name]
                    # filter goals out that are worse than our current best solution
                    remaining_goals = self._filter_goals(manip_goals, best_solution)
                    if len(remaining_goals) > 0:
                        traj, goal_id = motion_planner.plan(remaining_goals)
                        if traj is not None:
                            reached_goal = remaining_goals[goal_id]
                            # by invariant a newly reached goal should always have better objective
                            assert(best_solution is None or
                                   best_solution[1].objective_value < reached_goal.objective_value)
                            traj, improved_goal = self.goal_sampler.improve_path_goal(traj, reached_goal)
                            self.solutions.append((traj, improved_goal))
                            best_solution = (traj, improved_goal)
                            rospy.logdebug("Found new solution - it has objective value %f" %
                                           best_solution[1].objective_value)
                            connected_goals.append(improved_goal)
                            if self._stats_recorder:
                                # record both reached and improved goal (before and after local optimization)
                                self._stats_recorder.register_new_solution(reached_goal)
                                if reached_goal != improved_goal:
                                    self._stats_recorder.register_new_solution(improved_goal)
            # lastly, inform goal sampler about the goals we reached this round
            self.goal_sampler.set_reached_goals(connected_goals)
            iter_idx += 1
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

    def get_solution(self, i, bsimplify=True):
        """
            Return the ith trajectory found. Optionally, simplify the solution.
        """
        if i >= len(self.solutions):
            return None, None
        if bsimplify:
            planner = self._motion_planners[self.solutions[i][1].manip.GetName()]
            planner.simplify(self.solutions[i][0], self.solutions[i][1])
        self.time_traj(self.solutions[i][0])
        return self.solutions[i]

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
        goal_candidates.reverse()
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
        return [x for x in goals if x.objective_value > best_solution[1].objective_value]

    def _merge_goal_sets(self, old_goals, new_goals):
        """
            Merge new_goals into old_goals.
            ---------
            Arguments
            ---------
            old_goals, dict - manip_name -> list of goals
            new_goals, dict - manip_name -> list of goals
            -------
            Returns
            -------
            old_goals
        """
        for (key, goals) in new_goals.iteritems():
            if key in old_goals:
                old_goals[key].extend(goals)
            else:
                old_goals[key] = goals
        return old_goals

    def set_parameters(self, **kwargs):
        """
            Set parameters
        """
        for key, value in kwargs.iteritems():
            if key in self._params:
                self._params[key] = value
            else:
                rospy.logwarn("Unknown parameter: %s" % key)


class DummyPlanner(object):
    """
        Dummy placement planner that pretends to plan motions.
        It samples several goals, and then randomly decides that one of them was reached by a motion planner.
        Subsequently, the goal sampler is informed about this and queried again.
    """

    def __init__(self, goal_sampler, num_goal_samples=10, num_goal_iterations=10, stats_recorder=None):
        """
            num_goal_samples, int - number of goal samples the goal sampler should acquire in each iteration
            num_goal_trials, int - total number of trials the goal sampler has to do so in each iteration
        """
        self._goal_sampler = goal_sampler
        self._stats_recorder = stats_recorder
        self.num_goal_samples = num_goal_samples
        self.num_goal_iterations = num_goal_iterations

    def plan(self, timeout, target_object):
        """
            Run pseudo planning.
            ---------
            Arguments
            ---------
            timeout, float - maximum time
            target_object, Kinbody - body to place
        """
        objectives = []
        solutions = []
        goal_set = {}
        best_solution = None
        start_time = time.time()
        while time.time() - start_time < timeout:
            new_goals, _ = self._goal_sampler.sample(self.num_goal_samples, self.num_goal_iterations, True)
            self._check_objective_invariant(new_goals, best_solution)
            goal_set = self._merge_goal_sets(goal_set, new_goals)
            if len(goal_set) > 0:
                all_sols = []
                for (_, sols) in goal_set.iteritems():
                    all_sols.extend(sols)
                all_sols = self._filter_goals(all_sols, best_solution)
                if len(all_sols) > 0:
                    selected_goal = random.choice(all_sols)
                    self._goal_sampler.set_reached_goals([selected_goal])
                    best_solution = selected_goal
                    objectives.append(best_solution.objective_value)
                    solutions.append(best_solution)
                    if self._stats_recorder:
                        self._stats_recorder.register_new_solution(selected_goal)

        return objectives, solutions

    def _check_objective_invariant(self, goals, best_sol):
        if best_sol is None:
            return
        for (key, goal_set) in goals.iteritems():
            for goal in goal_set:
                assert(goal.objective_value >= best_sol.objective_value)

    def _merge_goal_sets(self, old_goals, new_goals):
        """
            Merge new_goals into old_goals.
            ---------
            Arguments
            ---------
            old_goals, dict - manip_name -> list of goals
            new_goals, dict - manip_name -> list of goals
            -------
            Returns
            -------
            old_goals
        """
        for (key, goals) in new_goals.iteritems():
            if key in old_goals:
                old_goals[key].extend(goals)
            else:
                old_goals[key] = goals
        return old_goals

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
        return [x for x in goals if x.objective_value > best_solution.objective_value]
