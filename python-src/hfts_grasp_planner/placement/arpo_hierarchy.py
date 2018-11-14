import numpy as np
import hfts_grasp_planner.placement.so2hierarchy as so2hierarchy
"""
    This module defines the Arm-Region-PlacementOrientation(arpo)-hierarchy.
    An arpo-hierarchy allows to search for an object placement with a dual arm robot.
    On the root level,the hierarchy has as many branches as the robot has arms, i.e.
    it represents the decision which arm to use to place the object.
    On the second level, the hierarchy has as many branches as there are placement regions
    in the target volume.
    On the third level, it has as many branches as there are placement orientations for the object.
    Subsequent hierarchy levels arise from a subdivison of the SE(2) state space of the object
    in a particular region for the chose placement orientation.
"""


class ARPOHierarchy(object):
    """
        Defines an ARPO hierarchy.
        A node in this hierarchy is identified by a key, which is a tuple of ints followed by two more tuples:
        (a, r, o, subregion_key, so2_key). The integers a, r, o define the arm, region and placement orientation.
        The elements subregion_key and so2_key are themselves tuple of ints representing 1. a subregion of the region r
        and 2. an interval of SO2. A key may be partially defined from left to right. Valid key formats are:
        (,) - root
        (a,) - chosen arm a, nothing else
        (a, r) - chosen arm a, region r, nothing else
        (a, r, o) - chosen arm a, region r, placement orientation
        (a, r, o, subregion_key, so2_key) - subregion_key and so2_key can also be partially defined in the same way.
    """

    def __init__(self, manipulators, regions, orientations, so2_branching):
        """
            Create a new ARPO hierarchy.
            ---------
            Arguments
            ---------
            manipulators, list of OpenRAVE manipulators - each manipulator represents one arm
            regions, list of PlacementRegions (see module placement_regions)
            orientations, list of PlacementOrientation (see module placement_orientations)
            so2_branching, int - branching factor for SO2 on hierarchy levels below level 3.
        """
        self._manips = manipulators
        self._regions = regions
        self._orientations = orientations
        self._so2_branching = so2_branching

    def get_child_key_gen(self, key):
        """
            Return a key-generator for the children of the given key.
            ---------
            Arguments
            ---------
            key, tuple of int - see class documentation for key description
            -------
            Returns
            -------
            generator of tuple of int - the generator produces children of the node with key key
        """
        if len(key) == 0:
            return ((i,) for i in xrange(len(self._manips)))
        elif len(key) == 1:
            return (key + (i,) for i in xrange(len(self._regions)))
        elif len(key) == 2:
            return (key + (i,) for i in xrange(len(self._orientations)))
        else:
            # extract sub region key
            subregion_key = key[3]
            subregion = self.get_placement_region(subregion_key)
            return (key + (r, o) for r in xrange(subregion.get_num_subregions())
                    for o in so2hierarchy.get_key_gen(key[4], self._so2_branching))

    def get_placement_region(self, region_key):
        """
            Return the placement region with the specified region key.
            The key may either indentify a whole region or a subregion.
            ---------
            Arguments
            ---------
            region_key, int of tuple(int, tuple(int, ...)) - 
                The region key can either be a single int or a tuple consisting of an
                int and another tuple of ints. In case it is a single int, it identifies 
                a placement region. In case it is a tuple of the form (int, tuple(int, ...)),
                region_key[0] identifies the placement region and region_key[1] identifies the 
                subregion of that region.
        """
        if type(region_key) is int:
            return self._regions[region_key]
        assert(type(region_key)) is tuple
        region = self._regions[region_key[0]]
        return region.get_subregion(region_key[1])

    def get_arpo_information(self, key):
        """
            Return the arm, region, placement orientation, subregion and so2 interval represented
            by the given key.
            ---------
            Arguments
            ---------
            key, tuple as described in class definition
        """
        if len(key) == 0:
            return None
        elif len(key) == 1:
            return (self._manips[key[0]],)
        elif len(key) == 2:
            return (self._manips[key[0]], self._regions[key[1]])
        elif len(key) == 3:
            return (self._manips[key[0]], self._regions[key[1]], self._regions[key[2]])

        subregion = self.get_placement_region(key[3])
        so2region = so2hierarchy.get_interval(key[4], self._so2_branching)
        return (self._manips[key[0]], self._regions[key[1]], self._regions[key[2]], subregion, so2region)
