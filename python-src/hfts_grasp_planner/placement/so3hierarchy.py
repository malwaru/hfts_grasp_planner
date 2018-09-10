import os
import healpy
import numpy as np
# import functools
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
    quat[0] = np.cos(hopf[0] / 2.0) * np.cos(hopf[2] / 2.0)
    quat[1] = np.sin(hopf[0] / 2.0) * np.sin(hopf[1] + hopf[2] / 2.0)
    quat[2] = np.sin(hopf[0] / 2.0) * np.cos(hopf[1] + hopf[2] / 2.0)
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
            S^2_key and S^1_key are expected to be lists of positive integers with equal lengths
            In both lists, each integer denotes the local id of a child, e.g. the key [3, 1] represents
            the first child of the third child on the root level. The integers must be between 0 and the maximum
            number of children on the respective level, i.e. all S^2_keys must be elementwise smaller than
            [12, 4, 4, 4, ...] and all S^1 keys must be elementwise smaller than [6, 2, 2, 2, ...].
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
    if len(s2_key) == 0:  # root
        result[0, :] = [0.0, np.pi]
        result[1, :] = [0.0, 2.0 * np.pi]  # NOTE the 2pi should actually be exclusive
        result[2, :] = [0.0, 2.0 * np.pi]  # NOTE the 2pi should actually be exclusive
    else:
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

# def _compute_ks(s2_key, nside):
#     """
#         Compute the number of healpix splits that occur above for northern or
#         below for southern pole pixels. The function does not produce meaningful results for equatorial pixels.
#         This function is for internal use.
#         ---------
#         Arguments
#         ---------
#         s2_key - Key of the pixel in the hierarchy
#         nside, int - nside of healpix hierarchy
#         -------
#         Returns
#         -------
#         k_left, int - split lines on top (on bottom) on the left
#         k_right, int - split lines on top (on bottom) on the right
#     """
#     k_left, k_right = 0, 0
#     if s2_key[0] <= 3:  # North Pole
#         for d in xrange(1, len(s2_key)):
#             if s2_key[d] == 0:
#                 k_left += pow(2, len(s2_key) - 1 - d)
#                 k_right += pow(2, len(s2_key) - 1 - d)
#             elif s2_key[d] == 1:
#                 k_right += pow(2, len(s2_key) - 1 - d)
#             elif s2_key[d] == 2:
#                 k_left += pow(2, len(s2_key) - 1 - d)
#     elif s2_key[0] >= 8:  # South Pole
#         for d in xrange(1, len(s2_key)):
#             if s2_key[d] == 3:
#                 k_left += pow(2, len(s2_key) - 1 - d)
#                 k_right += pow(2, len(s2_key) - 1 - d)
#             elif s2_key[d] == 1:
#                 k_right += pow(2, len(s2_key) - 1 - d)
#             elif s2_key[d] == 2:
#                 k_left += pow(2, len(s2_key) - 1 - d)
#     else:
#         rospy.logwarn("Computing ks for equatorial pixels does not make sense. This indicates a logic bug!")
#     return k_left, k_right


# def _domain_constr(min_values, max_values, sub_constr, vals):
#     """
#         Internal use!
#         Contrain values to the proper domain and return violation values as well
#         as violation of projected sub_constraint.
#         ---------
#         Arguments
#         ---------
#         min_values, numpy array of shape (2,) - minimum values for theta and phi
#         max_values, numpy array of shape (2,) - maximum values for theta and phi
#         sub_const, another constraint that depends on theta and phi to be within the specified box
#         vals, numpy array of shape (2,) - [theta, phi]
#     """
#     assert(len(vals) == 2)
#     # first project point to box spanned by min_values, max_values
#     min_violation = vals - min_values  # negative values mean violation
#     max_violation = max_values - vals  # negative values mean violation
#     if min_violation[0] < 0.0 and min_violation[1] < 0.0:  # bottom left corner
#         box_violation = -np.linalg.norm(min_violation)
#         proj_vals = min_values
#     elif min_violation[0] < 0.0 and min_violation[1] >= 0.0 and max_violation[1] >= 0:  # left side
#         box_violation = min_violation[0]
#         proj_vals = np.array([min_values[0], vals[1]])
#     elif min_violation[0] < 0.0 and max_violation[1] < 0.0:  # top left corner
#         box_violation = -np.linalg.norm(np.array([min_violation[0], max_violation[1]]))
#         proj_vals = np.array([min_values[0], max_values[1]])
#     elif min_violation[0] >= 0.0 and max_violation[0] >= 0.0 and max_violation[1] < 0.0:  # top side
#         box_violation = max_violation[1]
#         proj_vals = np.array([vals[0], max_values[1]])
#     elif max_violation[0] < 0.0 and max_violation[1] < 0.0:  # top right corner
#         box_violation = -np.linalg.norm(max_violation)
#         proj_vals = max_values
#     elif max_violation[0] < 0.0 and max_violation[1] >= 0.0 and min_violation[1] >= 0.0:  # right side
#         box_violation = max_violation[0]
#         proj_vals = np.array([max_values[0], vals[1]])
#     elif max_violation[0] < 0.0 and min_violation[1] < 0.0:  # right bottom corner
#         box_violation = -np.linalg.norm(np.array([max_violation[0], min_violation[1]]))
#         proj_vals = np.array([max_values[0], min_values[1]])
#     elif min_violation[1] < 0.0 and min_violation[0] >= 0.0 and max_violation[0] >= 0.0:  # bottom side
#         box_violation = min_violation[1]
#         proj_vals = np.array([vals[0], min_values[1]])
#     else:  # values lie inside the box
#         box_violation = 0.0
#         proj_vals = vals
#     return box_violation + sub_constr(proj_vals)


# def _get_healpix_boundary_constraints(s2_key, s2_idx, nside):
#     """
#         Return the boundary constraints for the healpix identied by s2_key.
#         This function is used internally and you probably want to call get_hopf_coordinate_range(key) instead.
#         ---------
#         Arguments
#         ---------
#         s2_key - Key in the s2 hierarchy
#         s2_idx, int - healpix index in nested scheme
#         nside, int - healpix nside
#         -------
#         Returns
#         -------
#         boundary_constr, list of length 4 - each element of this list is a function representing
#             an edge of the healpix. The functions take theta and phi as arguments and return a value
#             r > 0, if (cos(theta), phi) are on the correct side of the edge, r = 0 if on the edge and r < 0
#             if on the wrong side.
#     """
#     boundary_constr = []

#     if nside == 1:
#         # TODO return root boundaries
#         pass
#     ring_idx = healpy.pix2ring(nside, np.array([s2_idx]), nest=True)
#     if ring_idx <= nside - 1:  # North pole
#         left_k, right_k = _compute_ks(s2_key, nside)

#         def in_bounding_box()

#         def left_upper_bound(k, nside, theta, phi):
#             if phi >= np.pi:
#                 return -
#             phi_t = phi % (np.pi / 2.0)
#             return 1.0 - k**2 / (3.0 * nside**2) * (np.pi / (2.0 * phi_t - np.pi))**2 - np.cos(theta)

#         # TODO
#     elif ring_idx <= 3 * nside:
#         # equatorial zone, we have four lines. Compute the line parameters from the corner positions
#         corner_thetas, corner_phis = healpy.vec2ang(healpy.boundaries(nside, s2_idx, nest=True))
#         cos_thetas = np.cos(corner_thetas)
#         b = np.pi * (cos_thetas[0] - cos_thetas[1]) / (corner_phis[0] - corner_phis[1])

#         def constraint_1(a, b, c, theta, phi):
#             return a + b * (phi - c) / (2.0 * np.pi) - np.cos(theta)

#         def constraint_2(a, b, c, theta, phi):
#             return a - b * (phi - c) / (2.0 * np.pi) - np.cos(theta)

#         def constraint_3(a, b, c, theta, phi):
#             return np.cos(theta) - a + b * (phi - c) / (2.0 * np.pi)

#         def constraint_4(a, b, c, theta, phi):
#             return np.cos(theta) - a - b * (phi - c) / (2.0 * np.pi)
#         boundary_constr = [
#             functools.partial(constraint_1, cos_thetas[1], b, corner_phis[1]),
#             functools.partial(constraint_2, cos_thetas[0], b, corner_phis[1]),
#             functools.partial(constraint_3, cos_thetas[1], b, corner_phis[1]),
#             functools.partial(constraint_4, cos_thetas[2], b, corner_phis[1]),
#         ]
#     else:
#         # South pole
#         left_k, right_k = _compute_ks(s2_key, nside)
#     return boundary_constr


# def get_hopf_coordinate_range(key):
#     """
#         Return the range for the Hopf coordinates that are represented
#         by the element with the given key. The range is implicitly described
#         through inequality constraints, represented by two functions described in
#         more detail below.
#         ---------
#         Arguments
#         ---------
#         key, tuple - A key identifying a node as described in get_quaternion(..).
#         -------
#         Returns
#         -------
#         theta_phi_constr, a function - call this function like this: r = theta_phi_constr(theta, phi),
#             where theta and phi are the first two Hopf coordinates. r is negative if the pair (theta, phi)
#             is out of bounds, 0 on the boundary and positive inside the range. The function is continuous.
#         psi_constr, a function - call this function like this: r = psi_constr(psi),
#             where psi is the third Hopf coordinate. r is negative if psi
#             is out of bounds, 0 if on the boundary and positive if inside the range.
#             The function is continuous.
#     """
#     assert(len(key) == 2)
#     s2_key, s1_key = key[0], key[1]  # extract keys for both hierarchies
#     assert(len(s2_key) == len(s1_key))
#     s2_idx, s1_idx, level = key_to_indices(key)  # get indices
#     num_s1_values = 6 * pow(2, level)  # number of s1_values on this level for each healpix
#     boundary_constr = get_healpix_boundary_constraints(s2_idx, s2_key, healpy.order2nside(level))

#     def interval_constr(lv, uv, x):
#         return np.min(x - lv, uv - x)

#     def pixel_constr(boundary_constr, theta, phi):
#         return np.min(np.min(boundary_constr[0](theta, phi), boundary_constr[1](theta, phi)),
#                       np.min(boundary_constr[2](theta, phi), boundary_constr[3](theta, phi)))

#     psi_constr = functools.partial(interval_constr, float(s1_idx) / num_s1_values *
#                                    np.pi, (s1_idx + 1.0) / num_s1_values * np.pi)
#     theta_phi_constr = functools.partial(pixel_constr, boundary_constr)
#     return theta_phi_constr, psi_constr
