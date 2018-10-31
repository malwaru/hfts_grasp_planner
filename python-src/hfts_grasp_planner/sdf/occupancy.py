import math
import operator
import itertools
import numpy as np
import openravepy as orpy
import scipy.ndimage.morphology


class OccupancyGridBuilder(object):
    """
        An OccupancyGridBuilder builds a occupancy grid (a binary collision map)
        in R^3. If you intend to construct multiple occupancy grids with the same cell size,
        it is recommended to use a single OccupancyGridBuilder as this saves resources generation.
        Note that the occupancy grid is constructed only considering collisions with the enabled bodies
        in a given OpenRAVE environment.
    """
    class BodyManager(object):
        """
            Internal helper class for managing box-shaped bodies for collision checking.
        """

        def __init__(self, env, cell_size):
            """
                Create a new BodyManager
                @param env - OpenRAVE environment
                @param cell_size - size of a cell
            """
            self._env = env
            self._cell_size = cell_size
            self._bodies = {}
            self._active_body = None

        def get_body(self, dimensions):
            """
                Get a kinbody that covers the given number of cells.
                @param dimensions - numpy array (wx, wy, wz)
            """
            new_active_body = None
            if tuple(dimensions) in self._bodies:
                new_active_body = self._bodies[tuple(dimensions)]
            else:
                new_active_body = orpy.RaveCreateKinBody(self._env, '')
                new_active_body.SetName("CollisionCheckBody" + str(dimensions[0]) +
                                        str(dimensions[1]) + str(dimensions[2]))
                physical_dimensions = self._cell_size * dimensions
                new_active_body.InitFromBoxes(np.array([[0, 0, 0,
                                                         physical_dimensions[0] / 2.0,
                                                         physical_dimensions[1] / 2.0,
                                                         physical_dimensions[2] / 2.0]]),
                                              True)
                self._env.AddKinBody(new_active_body)
                self._bodies[tuple(dimensions)] = new_active_body
            if new_active_body is not self._active_body and self._active_body is not None:
                self._active_body.Enable(False)
                self._active_body.SetVisible(False)
            self._active_body = new_active_body
            self._active_body.Enable(True)
            self._active_body.SetVisible(True)
            return self._active_body

        def disable_bodies(self):
            if self._active_body is not None:
                self._active_body.Enable(False)
                self._active_body.SetVisible(False)
                self._active_body = None

        def clear(self):
            """
                Remove and destroy all bodies.
            """
            for body in self._bodies.itervalues():
                self._env.Remove(body)
                body.Destroy()
            self._bodies = {}
    ################################### Methods ###################################

    def __init__(self, env, cell_size):
        """
            Creates a new OccupancyGridBuilder object.
            @param env - OpenRAVE environment this builder operates on.
            @param cell_size - The cell size of the signed distance field.
        """
        self._env = env
        self._cell_size = cell_size
        self._body_manager = OccupancyGridBuilder.BodyManager(env, cell_size)

    def __del__(self):
        self._body_manager.clear()

    def _compute_bcm_rec(self, min_idx, max_idx, grid, covered_volume):
        """
            Computes a binary collision map recursively.
            INVARIANT: This function is only called if there is a collision for a box ranging from min_idx to max_idx
            @param min_idx - numpy array [min_x, min_y, min_z] cell indices
            @param max_idx - numpy array [max_x, max_y, max_z] cell indices (the box excludes these)
            @param grid - the grid to operate on. should be of type bool
        """
        # Base case, we are looking at only one cell
        if (min_idx + 1 == max_idx).all():
            grid.set_cell_value(min_idx, True)
            return covered_volume + 1
        # else we need to split this cell up and see which child ranges are in collision
        box_size = max_idx - min_idx  # the number of cells along each axis in this box
        half_sizes = np.zeros((2, 3))
        half_sizes[0] = map(math.floor, box_size / 2)  # we split this box into 8 children by dividing along each axis
        half_sizes[1] = box_size - half_sizes[0]  # half_sizes stores the divisions for each axis
        # now we create the actual ranges for each of the 8 children
        children_dimensions = itertools.product(half_sizes[:, 0], half_sizes[:, 1], half_sizes[:, 2])
        # and the position offsets
        offset_matrix = np.zeros((2, 3))
        offset_matrix[1] = half_sizes[0]
        rel_min_indices = itertools.product(offset_matrix[:, 0], offset_matrix[:, 1], offset_matrix[:, 2])
        for (rel_min_idx, child_dim) in itertools.izip(rel_min_indices, children_dimensions):
            volume = reduce(operator.mul, child_dim)
            if volume != 0:
                child_min_idx = min_idx + np.array(rel_min_idx)
                child_max_idx = child_min_idx + np.array(child_dim)
                child_physical_dimensions = grid.get_cell_size() * np.array(child_dim)
                cell_body = self._body_manager.get_body(np.array(child_dim))
                transform = cell_body.GetTransform()
                transform[0:3, 3] = grid.get_cell_position(child_min_idx, b_center=False)
                transform[0:3, 3] += child_physical_dimensions / 2.0  # the center of our big box
                cell_body.SetTransform(transform)
                if self._env.CheckCollision(cell_body):
                    covered_volume = self._compute_bcm_rec(child_min_idx, child_max_idx, grid, covered_volume)
                else:
                    grid.fill(child_min_idx, child_max_idx, False)
                    covered_volume += volume
        # total_volme = reduce(operator.mul, self._grid.get_num_cells())
        # print("Covered %i / %i cells" % (covered_volume, total_volme))
        return covered_volume

    def compute_grid(self, grid):
        """
            Fill the given grid with bool values. Each cell in the grid will be set to True, if there this
            cell is in collision with an obstacle in the current state of the environment.
            Otherwise, a cell has value False.
            *NOTE*: If you do not intend to continue creating more SDFs using this builder, call clear() afterwards.

            Arguments
            ---------
            grid - VoxelGrid with elements of type bool, will be modified
        """
        # compute for each cell whether it collides with anything
        covered_volume = self._compute_bcm_rec(np.array([0, 0, 0]), grid.get_num_cells(), grid, 0)
        assert(covered_volume == np.multiply.reduce(grid.get_num_cells()))
        # The above function can not detect the interior of meshes, therefore we need to fill holes
        filled_holes = scipy.ndimage.morphology.binary_fill_holes(grid.get_raw_data().astype(bool))
        grid.set_raw_data(filled_holes)
        self._body_manager.disable_bodies()

    def clear(self):
        """
            Clear all cached resources.
        """
        self._body_manager.clear()
