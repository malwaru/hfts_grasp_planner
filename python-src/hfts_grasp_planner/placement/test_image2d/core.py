from PIL import Image
import numpy as np
import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
import matplotlib.pyplot as plt_module


class ImageGoalRegion(plcmnt_interfaces.PlacementHierarchy,
                      plcmnt_interfaces.PlacementSolutionConstructor,
                      plcmnt_interfaces.PlacementValidator,
                      plcmnt_interfaces.PlacementObjective):
    """
        This class provides a (placement) goal region that is defined through
        a 2d image. Technically, this is not good for any placement, but only to
        visualize/debug the sampling strategy of, for instance, the mcts_sampler.
    """
    class FakeManipulator(object):
        def GetName(self):
            return "fake_manip"

    def __init__(self, img_path, branching=2):
        """
            Create a new ImageGoalRegion.
            ---------
            Arguments
            ---------
            img_path, string - path to image file (png)
            branching, int - branching factor in x and y
        """
        self._image = np.array(Image.open(img_path)).transpose((1, 0, 2))
        # normalize image
        self._image = self._image / 255.0
        self._debug_image = np.zeros(self._image.shape[:2])
        if self._image.shape[0] != self._image.shape[1]:
            raise ValueError("X and Y dimension of the loaded are image are not identical.")
        self._branching = branching
        self._max_depth = np.int(np.floor(np.log(self._image.shape[0]) / np.log(branching)))

    def get_child_key_gen(self, key):
        if len(key) == self._max_depth:
            return None
        if len(key) == 0:
            return ((x, y) for x in xrange(self._branching) for y in xrange(self._branching))
        return ((key[0] + (x,), key[1] + (y,)) for x in xrange(self._branching)
                                               for y in xrange(self._branching))

    def get_random_child_key(self, key):
        if len(key) == 0:
            return ((np.random.randint(self._branching),), (np.random.randint(self._branching),))
        if len(key[0]) == self._max_depth:
            return None
        return (key[0] + (np.random.randint(self._branching),), key[1] + (np.random.randint(self._branching),))

    def get_minimum_depth_for_construction(self):
        return 1

    def construct_solution(self, key, b_optimize_constraints, b_optimize_objective):
        solution = plcmnt_interfaces.PlacementGoalSampler.PlacementGoal(ImageGoalRegion.FakeManipulator(), None, None, key, 0.0, None, None)
        idx = self._get_index(solution.key)
        # self._debug_image[idx, 3] = np.clip(self._debug_image[idx, 2] - 1.0, 0.0, 1.0)
        self._debug_image[idx] = 1.0
        # print "Constructing solution at pixel ", idx
        return solution

    def is_valid(self, solution):
        idx = self._get_index(solution.key)
        return self._image[idx][0] == 1.0

    def get_constraint_relaxation(self, solution):
        idx = self._get_index(solution.key)
        return self._image[idx][0]

    def evaluate(self, solution):
        idx = self._get_index(solution.key)
        return self._image[idx][1]

    def _get_index(self, key):
        min_x, max_x = self._get_index_range(key, 0)
        min_y, max_y = self._get_index_range(key, 1)
        x = (min_x + max_x) / 2
        y = (min_y + max_y) / 2
        return (x, y)

    def _get_index_range(self, key, axis):
        interval = np.array((0, self._image.shape[axis]))
        for i in xrange(len(key[axis])):
            cell_width = (interval[1] - interval[0]) / self._branching
            interval[0] += key[axis][i] * cell_width
            interval[1] = interval[0] + cell_width
        return interval

    def show(self):
        """
            Open a window showing the image
        """
        # plt_module.imshow(self._image.transpose((1, 0, 2)))
        # plt_module.imshow(self._debug_image.transpose((1, 0)))
        render_image = np.array(self._image)
        render_image[:, :] += self._debug_image[:, :, np.newaxis]
        plt_module.imshow(render_image.transpose((1, 0, 2)))
        plt_module.show(block=False)
