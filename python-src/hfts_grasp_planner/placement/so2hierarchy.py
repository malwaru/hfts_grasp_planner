import numpy as np
"""
    This module contains functions that define a grid hierarchy on SO(2).
    All functions in this module that take a key argument assume this key
    to be tuples of integers. Each integer determines a cell on a layer
    of the hierarchy. For instance, the key (2, 3, 4) identifies the cell
    that is the 4th cell on layer 2 in the 3rd cell on layer 1 in the 2nd cell
    on layer 0. The total number of cells per layer is determined by the
    branching_factor. Layer 0 represents the full interval [0, 2pi].
"""


def get_angle(key, branching_factor):
    """
        Compute the center angle of the hierarchy cell with the given key.
        ---------
        Arguments
        ---------
        key, tuple of ints - key identifying which cell to get the center value for.
        branching_factor, int - number of cells per layer
        -------
        Returns
        -------
        angle, float
    """
    interval = get_interval(key, branching_factor)
    return (interval[0] + interval[1]) / 2.0


def get_interval(key, branching_factor):
    """
        Return the interval of [0, 2pi] that is covered by the cell with the given
        key for the given branching factor.
        ---------
        Arguments
        ---------
        key, tuple of ints - key identifying which cell to get the center value for.
        branching_factor, int - number of cells per layer
        -------
        Returns
        -------
        interval, numpy array of shape (2,) - min and max values of interval
    """
    interval = np.array([0, 2.0 * np.pi])
    for i in xrange(len(key)):
        cell_width = (interval[1] - interval[0]) / branching_factor
        interval[0] += key[i] * cell_width
        interval[1] = interval[0] + cell_width
    return interval


def get_leaf_key(value, branching_factor, depth):
    """
        Return the key of the leaf interval that the given value lies in.
        ---------
        Arguments
        ---------
        value, float - angle in range [0, 2pi]
        branching_factor, int - number of cells per layer
        depth, int - depth of the hierarchy
        -------
        Returns
        -------
        key, tuple - key where value lies in, or None if value is out of [0, 2pi]
    """
    key = []
    rvalue = value
    if value < 0.0 or value > 2.0 * np.pi:
        return None
    for i in xrange(depth):
        interval_size = 2.0 * np.pi / np.power(branching_factor, i + 1)
        child_id = np.floor(rvalue / interval_size)
        rvalue -= child_id * interval_size
        key.append(int(child_id))
    return tuple(key)


def get_key_gen(key, branching_factor):
    """
        Return a generator for child keys for the given key.
        ---------
        Arguments
        ---------
        key, tuple of ints - key identifying which cell to get child keys for
        branching_factor, int - number of cells per layer
        -------
        Returns
        -------
        key generator that produces the keys of all children of the given key
    """
    return (key + (i,) for i in xrange(branching_factor))


def is_leaf(key, depth):
    """
        Return whether the given key is a leaf.
    """
    return len(key) >= depth
