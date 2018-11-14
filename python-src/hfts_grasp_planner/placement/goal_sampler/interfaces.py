from abc import ABCMeta
"""
    This module describes the goal sampler interface for placement planning.
"""


class PlacementGoalSampler:
    __metaclass__ = ABCMeta

    class PlacementGoals:
        __metaclass__ = ABCMeta

        def __init__(self, manip, arm_config, key, data=None):
            """
                Create a new PlacementGoal.
                ---------
                Arguments
                ---------
                manip - OpenRAVE manipulator this goal is for
                arm_config, numpy array (n,) - arm configuration
                key, object - key information that can be used by the placement goal sampler to identify this goal
                data, object - optional additional data
            """
            self.manip = manip
            self.arm_config = arm_config
            self.key = key
            self.data = data

    @ABCMeta.abstractmethod
    def sample(self, num_solutions):
        """
            Sample new solutions.
            ---------
            Arguments
            ---------
            num_solutions, int - number of new solutions to sample
            -------
            Returns
            -------
            a list of PlacementGoals
        """
        pass

    @ABCMeta.abstractmethod
    def set_reached_goals(self, goals):
        """
            Inform the placement goal sampler that the given goals have been reached.
            ---------
            Arguments
            ---------
            goals, list of PlacementGoals
        """
        pass

    @ABCMeta.abstractmethod
    def set_reached_configurations(self, manip, configs):
        """
            Inform the placement goal sampler about reached arm configurations for the given manipulator.
            ---------
            Arguments
            ---------
            # TODO nearest neighbor data structure or just numpy array?
        """
        pass


# class PlacementHierarchy:
    # TODO define an interface that allows exchanging arpo hierarchy easily?
