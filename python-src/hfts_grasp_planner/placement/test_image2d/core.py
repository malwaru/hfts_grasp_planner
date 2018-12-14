from PIL import Image
import numpy as np
import pygame
import itertools
import random
import hfts_grasp_planner.placement.goal_sampler.interfaces as plcmnt_interfaces
import scipy.ndimage.morphology


class ImageVisualizer(object):
    def __init__(self, width, height):
        """
            Create a new visualizer.
            ---------
            Arugments
            ---------
            width, int - width of window
            height, int - height of window
        """
        pygame.init()
        self._screen = pygame.display.set_mode([width, height])
        self._width = pygame.display.Info().current_w
        self._height = pygame.display.Info().current_h
        self._screen.fill([0, 0, 0])
        self._images = []

    def render_image(self, image_file_name):
        """
            Renders the given image.
            ---------
            Arguments
            ---------
            image_file_name, string - path to image
        """
        self._images.append(pygame.image.load(image_file_name).convert())
        self.refresh()

    def mark_pixel(self, idx):
        """
            Set the given pixel white.
            ---------
            Arguments
            ---------
            idx, (x, y) - x, y coordinate of pixel
        """
        np_surface = pygame.surfarray.pixels3d(self._screen)
        np_surface[idx] = (255, 255, 255)
        pygame.display.update()

    def refresh(self):
        """
            Refresh the view.
        """
        for image in self._images:
            self._screen.blit(image, (0, 0))
        pygame.display.update()


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

    def __init__(self, img_path, branching=4):
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
        self._distance_img = scipy.ndimage.morphology.distance_transform_edt(1.0 - self._image[:, :, 0])
        self._distance_img = np.exp(10 * (1.0 - self._distance_img / np.max(self._distance_img)))
        self._distance_img = self._distance_img / np.max(self._distance_img)
        self._visualizer = ImageVisualizer(self._image.shape[1], self._image.shape[0])
        self._visualizer.render_image(img_path)
        if self._image.shape[0] != self._image.shape[1]:
            raise ValueError("X and Y dimension of the loaded are image are not identical.")
        self._branching = branching
        self._max_depth = np.int(np.floor(np.log(self._image.shape[0]) / np.log(branching)))
        # number of calls for solution construction, validity check, relaxation and evaluation
        self._stats = np.array([0, 0, 0, 0])

    def get_child_key_gen(self, key):
        if self.is_leaf(key):
            return None
        return self.get_random_child_key_gen(key)
        # if self.is_leaf(key):
        #     return None
        # if len(key) == 0:
        #     return (((x,), (y,)) for x in xrange(self._branching) for y in xrange(self._branching))
        # return ((key[0] + (x,), key[1] + (y,)) for x in xrange(self._branching)
        #         for y in xrange(self._branching))

    def get_random_child_key_gen(self, key):
        child_options = list(itertools.product(range(self._branching), repeat=2))
        random.shuffle(child_options)
        for child_key in child_options:
            if len(key) == 0:
                yield ((child_key[0],), (child_key[1],))
            else:
                yield (key[0] + (child_key[0],), key[1] + (child_key[1],))

    def get_random_child_key(self, key):
        if len(key) == 0:
            return ((np.random.randint(self._branching),), (np.random.randint(self._branching),))
        if self.is_leaf(key):
            return None
        return (key[0] + (np.random.randint(self._branching),), key[1] + (np.random.randint(self._branching),))

    def get_minimum_depth_for_construction(self):
        return 1

    def can_construct_solution(self, key):
        return len(key) != 0

    def construct_solution(self, key, b_optimize_constraints, b_optimize_objective):
        solution = plcmnt_interfaces.PlacementGoalSampler.PlacementGoal(
            ImageGoalRegion.FakeManipulator(), None, None, key, 0.0, None, None)
        idx = self._get_index(solution.key)
        self._stats[0] += 1
        self._visualizer.mark_pixel(idx)
        return solution

    def is_valid(self, solution):
        idx = self._get_index(solution.key)
        self._stats[1] += 1
        return self._image[idx][0] == 1.0

    def is_leaf(self, key):
        if len(key) == 0:
            return False
        return len(key[0]) == self._max_depth

    def get_num_children(self, key):
        if self.is_leaf(key):
            return 0
        return self._branching * self._branching

    def get_constraint_relaxation(self, solution):
        idx = self._get_index(solution.key)
        # return 1.0 - self._distance_img[idx] / self._max_distance
        return self._distance_img[idx]
        # return self._image[idx][0]

    def evaluate(self, solution):
        idx = self._get_index(solution.key)
        self._stats[2] += 1
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

    def get_num_construction_calls(self, b_reset=True):
        val = self._stats[0]
        if b_reset:
            self._stats[0] = 0
        return val

    def get_num_validity_calls(self, b_reset=True):
        val = self._stats[1]
        if b_reset:
            self._stats[1] = 0
        return val

    def get_num_relaxation_calls(self, b_reset=True):
        val = self._stats[2]
        if b_reset:
            self._stats[2] = 0
        return val

    def get_num_evaluate_calls(self, b_reset=True):
        val = self._stats[3]
        if b_reset:
            self._stats[3] = 0
        return val
