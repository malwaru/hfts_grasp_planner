import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
"""
    This module contains the definition of a naive placement goal sampler - purely random sampler.
    The random sampler samples an arpo_hierarchy uniformly at random.
"""


class RandomPlacementSampler(plcmnt_interfaces.PlacementGoalSampler):
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
        # TODO randomly sample everything!!!!

    def set_reached_goals(self, goals):
        """
            Inform the placement goal sampler that the given goals have been reached.
            ---------
            Arguments
            ---------
            goals, list of PlacementGoals
        """
        # Nothing to do here
        pass

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
