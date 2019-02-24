import time
import hfts_grasp_planner.external.transformations as tf_mod


class GoalSamplingStatsRecorder(object):
    """
        GoalSamplingStatsRecorder can be used to record how placement goal sampler is performing.
        Whenever the goal sampler finds a new placement goal it should call this object's
        register_new_goal(..) function, which stores the following information:
            - wall time when the solution was found (relative to start time)
            - number of goal sample attempts
            - total number of valid goal samples sampled before
            - objective value
            - global pose (x, y, z, ex, ey, ez)
    """

    def __init__(self, sol_validator, sol_constructor, objective):
        """
            Create a new recorder.
            ---------
            Arguments
            ---------
            sol_validator, PlacementValidator - validator that the placement planner uses.
            sol_constructor, PlacementConstructor - constructor that the placement planner uses.
            objective, PlacementObjective - objective fn that the placement planner uses
        """
        self._validator = sol_validator
        self._constructor = sol_constructor
        self._objective_fn = objective
        self._stats = []
        self._start_time = time.clock()
        self.reset()

    def reset(self):
        """
            Reset statistics and time. Call this just before you start planning.
        """
        self._stats = []
        self._constructor.get_num_construction_calls(True)
        self._validator.get_num_validity_calls(True)
        self._validator.get_num_relaxation_calls(True)
        self._objective_fn.get_num_evaluate_calls(True)
        self._start_time = time.clock()

    def register_new_goal(self, solution):
        """
            Register stats for the new solution.
            ---------
            Arguments
            ---------
            solution, PlacementSolution - a new solution found by a goal sampler
        """
        x, y, z = solution.obj_tf[:3, 3]
        ex, ey, ez = tf_mod.euler_from_matrix(solution.obj_tf)
        self._stats.append((time.clock() - self._start_time,  # wall clock
                            self._constructor.get_num_construction_calls(False),  # num goal samples
                            len(self._stats),  # num valid goals so far
                            solution.objective_value,
                            x, y, z, ex, ey, ez,  # pose
                            self._validator.get_num_validity_calls(False),
                            self._validator.get_num_relaxation_calls(False),
                            self._objective_fn.get_num_evaluate_calls(False)))

    def save_stats(self, file_name):
        """
            Save stats to file. If the file already exists, it will be overwritten.
            The file will contain the stats recorded so far as csv.
            The first line constains headers, the remaining lines the stats recorded.
            ---------
            Arguments
            ---------
            file_name, string - where to store stats.
        """
        with open(file_name, 'w') as the_file:
            the_file.write(
                'wall_clock,#samples,#goals,objective,x,y,z,ex,ey,ez,#validity_checks,#relax_calls,#evaluate_calls\n')
            for stat in self._stats:
                the_file.write(str(stat).replace('(', '').replace(')', '') + '\n')


class PlacementMotionStatsRecorder(object):
    """
        For the overall algorithm, it records the following information whenever a new solution was found:
            - wall time when it was found
            - objective value
            - global pose (x, y, z, ex, ey, ez)
    """

    def __init__(self):
        self._stats = []
        self._start_time = time.clock()

    def reset(self):
        self._stats = []
        self._start_time = time.clock()

    def register_new_solution(self, sol):
        x, y, z = sol.obj_tf[:3, 3]
        ex, ey, ez = tf_mod.euler_from_matrix(sol.obj_tf)
        self._stats.append((time.clock() - self._start_time, sol.objective_value, x, y, z, ex, ey, ez))

    def save_stats(self, file_name):
        """
            Save stats to file. If the file already exists, it is overwritten.
            The file will contain the stats recorded so far as csv.
            The first line constains headers, the remaining lines the stats recorded.
            ---------
            Arguments
            ---------
            file_name, string - where to store stats.
        """
        with open(file_name, 'w') as the_file:
            the_file.write(
                'wall_clock,objective,x,y,z,ex,ey,ez\n')
            for stat in self._stats:
                the_file.write(str(stat).replace('(', '').replace(')', '') + '\n')
