import os
import numpy as np
from itertools import izip, product


class SO3Hierarchy(object):
    """
        A precomputed grid hierarchy on SO(3) based on Yershova et al.'s work 
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

    def __init__(self):
        self._representatives = np.load(os.path.join(os.path.dirname(__file__), 'so3hdata.npy'))

    def get_quaternion(self, key):
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
        idx = self._get_index(key)
        return self._representatives[idx]

    @staticmethod
    def max_depth():
        return 4

    @staticmethod
    def get_root_key():
        return ([], [])

    @staticmethod
    def get_branching_factors(depth):
        if depth == 0:
            return (12, 6)
        return (4, 2)

    @staticmethod
    def get_random_neighbor(key):
        # first select a random S^2 neighbor
        s2_key = list(key[0])
        s1_key = list(key[1])
        if len(s2_key) == 0: # just return in case we have an empty key
            return ()
        if len(s2_key) == 1: # in case we are on the root level, we have specific choices
            s2_key[-1] = np.random.choice(SO3Hierarchy.S2_ROOT_NEIGHBORS[s2_key[-1]])
            rdir = np.random.randint(-1, 2)
            s1_key[-1] = np.clip(s1_key[-1] + rdir, 0, 5)
        else:
            s2_key[-1] = np.random.randint(4)
            s1_key[-1] = np.random.randint(2)
        return (s2_key, s1_key)

    @staticmethod
    def _get_index(key):
        s2_key, s1_key = key[0], key[1]  # extract keys for both hierarchies
        level_offset = 0  # we first compute an index offset for the level
        level = len(s2_key) - 1  # the level of the hierarchy we are on (0 for root)
        level_offset = 72 * (pow(8, level) - 1) / 7  # this is the number of elements for the previous levels
        num_s1_values = 6 * pow(2, level)  # number of s1_values on this level for each healpix
        healpix_idx = 0  # index of this healpix on this level, a value within [0, 12 * 4^level)
        s1_offset = 0  # offset for s1 indices within a healpix, a value within [0, 6 * 2^level)
        for d in xrange(len(s2_key)):  # compute offsets
            healpix_idx += s2_key[d] * pow(4, level - d)  
            s1_offset += s1_key[d] * pow(2, level - d)
        within_level_offset = healpix_idx * num_s1_values 
        return level_offset + within_level_offset + s1_offset


if __name__ == '__main__':
    # Test indexing
    hierarchy = SO3Hierarchy()
    test_array = np.zeros((hierarchy._representatives.shape[0], 1))
    num_keys = 0
    print 'Test array shape is ', test_array.shape
    # root level
    s2_keys = product(range(12))
    s1_keys = product(range(6))
    all_keys = product(s2_keys, s1_keys)
    for key in all_keys:
        idx = hierarchy._get_index(key)
        test_array[idx] += 1
        num_keys += 1
    # level 1
    s2_keys = product(range(12), range(4))
    s1_keys = product(range(6), range(2))
    all_keys = product(s2_keys, s1_keys)
    for key in all_keys:
        idx = hierarchy._get_index(key)
        test_array[idx] += 1
        num_keys += 1
    # level 2
    s2_keys = product(range(12), range(4), range(4))
    s1_keys = product(range(6), range(2), range(2))
    all_keys = product(s2_keys, s1_keys)
    for key in all_keys:
        idx = hierarchy._get_index(key)
        test_array[idx] += 1
        num_keys += 1
    # level 3
    s2_keys = product(range(12), range(4), range(4), range(4))
    s1_keys = product(range(6), range(2), range(2), range(2))
    all_keys = product(s2_keys, s1_keys)
    for key in all_keys:
        idx = hierarchy._get_index(key)
        test_array[idx] += 1
        num_keys += 1
    print 'Test array sum is', sum(test_array)
    print 'Test array min is', min(test_array)
    print 'Test array max is', max(test_array)
    print 'Num keys tried is ', num_keys