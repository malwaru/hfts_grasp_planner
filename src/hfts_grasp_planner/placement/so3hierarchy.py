import os
import healpy
import numpy as np
from itertools import izip, product

"""
    This module contains functions that define a grid hierarchy on SO(3) based on Yershova et al.'s work
    'Generating Uniform Inccremental Grids on SO(3) using the Hopf Fibration'.
    This grid has depth 4 (root level + 3 sublevels), and uses Hopf Fibration to represent
    SO(3). The hierarchy essentially consists of two sub-hierarchies, one for S^2 and one for S^1.
    As hierarchy on S^2 this hierarchy applies HEALPix; for S^1 it applies just a simple grid on the interval [0,2pi].
    At the root level, HEALPix has 12 patches, which for each subsequent level in the hierarchy are subdivided by 4.
    The grid on S^1 has 6 elements on the root level and each element is split by 2 for each subsequent level.
    Arguments in favor for this hierarchy:
        - minimizes dispersion and discrepancy on each level of the hierarchy, 
            i.e. we get a good cover of SO(3) on each level
        - children of hierarchy nodes are all nearest neighbors of their parents w.r.t. to
            metric p(x, y) = arccos(|x dot y|), where x and y are quaternions representing
            rotations
        - no preference given to what axis is more important. all rotations are treated equally
            (this makes it look unintuitive when plotting as frames)
"""


S2_ROOT_NEIGHBORS = [[3, 4, 5, 1], [0, 5, 6, 2], [1, 6, 7, 3], [2, 7, 4, 0],
                     [3, 11, 8, 0], [0, 8, 9, 1], [1, 9, 10, 2], [2, 10, 11, 3],
                     [4, 11, 5, 9], [8, 5, 6, 10], [9, 6, 7, 11], [10, 7, 4, 8]]


def hopf_to_quaternion(hopf):
    """
        Return a quaternion for the given point in Hopf coordinates.
        ---------
        Arguments
        ---------
        hopf, list or numpy array of length 3 - Hopf coordinates (theta, phi, psi)
        -------
        Returns
        -------
        quat, numpy array of shape (4,) - quaternion representing orientation of hopf.
    """
    quat = np.empty(4)
    # quat[0] = np.cos(hopf[0] / 2.0) * np.cos(hopf[2] / 2.0)
    # quat[1] = np.cos(hopf[0] / 2.0) * np.sin(hopf[2] / 2.0)
    # quat[2] = np.sin(hopf[0] / 2.0) * np.cos(hopf[1] + hopf[2] / 2.0)
    # quat[3] = np.sin(hopf[0] / 2.0) * np.sin(hopf[1] + hopf[2] / 2.0)
    quat[0] = np.cos(hopf[0] / 2.0) * np.cos(hopf[2] / 2.0)
    quat[1] = np.sin(hopf[0] / 2.0) * np.sin(hopf[1] - hopf[2] / 2.0)
    quat[2] = np.sin(hopf[0] / 2.0) * np.cos(hopf[1] - hopf[2] / 2.0)
    quat[3] = np.cos(hopf[0] / 2.0) * np.sin(hopf[2] / 2.0)
    return quat


def key_to_indices(key):
    """
        Translate a hierarchy key to healpix index and index within S1.
        ---------
        Arguments
        ---------
        key - hierarchy key, see module description.
        ---------
        Returns
        ---------
        healpix_idx, int - index of the healpix
        s1_idx, int - index of the inteval in S1
        level, int - level of the hierarchy (order for healpix)

        Note: If key is ((), ()) then 0, 0, -1 is returned
    """
    assert(len(key) == 2)
    s2_key, s1_key = key[0], key[1]  # extract keys for both hierarchies
    assert(len(s2_key) == len(s1_key))
    # compute heal_pix_id
    level = len(s2_key) - 1  # the level of the hierarchy we are on (0 for root)
    healpix_idx = 0  # index of this healpix on this level, a value within [0, 12 * 4^level)
    s1_idx = 0  # s1 index, an integer within [0, 6 * 2^level)
    for d in xrange(len(s2_key)):  # compute offsets
        healpix_idx += s2_key[d] * pow(4, level - d)
        s1_idx += s1_key[d] * pow(2, level - d)
    return healpix_idx, s1_idx, level


def get_root_key():
    """
        Return the key that represents the root (no node) of the hierarchy.
    """
    return ([], [])


def get_branching_factors(depth):
    """
        Return branching factors for the given depth.
        ---------
        Arguments
        ---------
        depth, int
        ---------
        Returns
        ---------
        bfs, tuple of int
    """
    if depth == 0:
        return (12, 6)
    return (4, 2)


def get_hopf_coordinates(key):
    """
        Return the Hopf coordinates of the element with the given key.
    """
    healpix_idx, s1_idx, level = key_to_indices(key)
    num_s1_values = 6 * pow(2, level)  # number of s1_values on this level for each healpix
    nside = healpy.order2nside(level)
    theta, phi = healpy.pix2ang(nside, healpix_idx, nest=True)
    psi = (s1_idx + 0.5) * 2.0 * np.pi / num_s1_values
    return np.array([theta, phi, psi])


def get_quaternion(key):
    """
        Return the quaternion representing the element of the hierarchy with the given key.
        @param key - The key is expected to be a tuple (S^2_key, S^1_key), where 
            S^2_key and S^1_key are expected to be lists of positive integers with maximum length 4.
            In both lists, each integer denotes the local id of a child, e.g. the key [3, 1] represents
            the first child of the third child on the root level. The integers must be between 0 and the maximum
            number of children on the respective level, i.e. all S^2_keys must be elementwise smaller than
            [12, 4, 4, 4] and all S^1 keys must be elementwise smaller than [6, 2, 2, 2].
        @return quaternion (x1, x2, x3, x4), where x1 is the real part, and x2, x3, x4 are associated with complex
            parts i,j,k respectively.
    """
    hopf_coordinates = get_hopf_coordinates(key)
    return hopf_to_quaternion(hopf_coordinates)


def get_random_neighbor(key):
    """
        Return a random neighbor of the element identified by key.
        ---------
        Arguments
        ---------
        key, tuple - A key identifying a node as described in get_quaternion(..).
        -------
        Returns
        -------
        neighbor_key, tuple - A key identifying a random neighbor
    """
    # first select a random S^2 neighbor
    s2_key = list(key[0])
    s1_key = list(key[1])
    if len(s2_key) == 0:  # just return in case we have an empty key
        return ()
    if len(s2_key) == 1:  # in case we are on the root level, we have specific choices
        s2_key[-1] = np.random.choice(S2_ROOT_NEIGHBORS[s2_key[-1]])
        rdir = np.random.randint(-1, 2)
        s1_key[-1] = np.clip(s1_key[-1] + rdir, 0, 5)
    else:
        s2_key[-1] = np.random.randint(4)
        s1_key[-1] = np.random.randint(2)
    return (s2_key, s1_key)


def get_hopf_coordinate_range(key):
    """
        Return the range for the Hopf coordinates that are represented
        by the element with the given key. TODO Currently it only returns
        an approximation of this range (a bounding box).
        ---------
        Arguments
        ---------
        key, tuple - A key identifying a node as described in get_quaternion(..).
        -------
        Returns
        -------
        range, numpy array of shape (3, 2) - Each row of range represents the min/max values
            for the respective dimension. The order is theta, phi, psi.
    """
    result = np.empty((3, 2))
    assert(len(key) == 2)
    s2_key, s1_key = key[0], key[1]  # extract keys for both hierarchies
    assert(len(s2_key) == len(s1_key))
    healpix_idx, s1_idx, level = key_to_indices(key)  # get indices
    num_s1_values = 6 * pow(2, level)  # number of s1_values on this level for each healpix
    nside = healpy.order2nside(level)  # compute nside for healpix
    # we approximate the range of theta and phi by computing the corners of the healpix
    thetas, phis = healpy.vec2ang(healpy.boundaries(nside, healpix_idx, nest=True).transpose())
    result[0, 0] = np.min(thetas)
    result[0, 1] = np.max(thetas)
    result[1, 0] = np.min(phis)
    result[1, 1] = np.max(phis)
    result[2, 0] = float(s1_idx) / num_s1_values * np.pi
    result[2, 1] = (s1_idx + 1.0) / num_s1_values * np.pi
    return result
