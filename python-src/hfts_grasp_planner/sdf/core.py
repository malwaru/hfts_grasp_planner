"""
    This module contains the core components for signed distance fields.
"""
from __future__ import print_function
import math
import time
import yaml
import rospy
import os
import operator
import itertools
import skfmm
import numpy as np
import openravepy as orpy
import scipy.ndimage.morphology
from itertools import izip, product
from hfts_grasp_planner.utils import inverse_transform
# from mayavi import mlab
# from openravepy.misc import ComputeBoxMesh


class VoxelGrid(object):
    _rel_grid_neighbor_mask = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], dtype=int)
    """"
        A voxel grid is a 3D discretization of a robot's workspace.
        For each voxel in this grid, this voxel grid saves a single floating point number.
    """
    class VoxelCell(object):
        """
            A voxel cell is a cell in a voxel grid and represents a voxel.
        """

        def __init__(self, grid, idx):
            self._grid = grid
            self._idx = idx

        def get_idx(self):
            """
                Return the index of this cell.
            """
            return self._idx

        def get_position(self):
            """
                Return the position in R^3 of the center of this cell.
            """
            return self._grid.get_cell_position(self._idx)

        def get_size(self):
            """
                Returns the edge length of this cell.
            """
            return self._grid.get_cell_size()

        def get_value(self):
            """
                Return the value stored in this cell
            """
            return self._grid.get_cell_value(self._idx)

        def set_value(self, value):
            """
                Set the value of this cell
            """
            self._grid.set_cell_value(self._idx, value)

    def __init__(self, workspace_aabb, cell_size=0.02, base_transform=None, dtype=np.float_):
        """
            Creates a new voxel grid covering the specified workspace volume.
            @param workspace_aabb - bounding box of the workspace as numpy array of form
                                    [min_x, min_y, min_z, max_x, max_y, max_z] or a tuple
                                    ([x, y, z], [wx, wy, wz]), where x,y,z are the position of the center
                                    and wx, wy, wz are the dimensions of the box
            @param cell_size - cell size of the voxel grid (in meters)
            @param base_transform - if not None, any query point is transformed by base_transform
        """
        self._cell_size = cell_size
        if isinstance(workspace_aabb, tuple):
            dimensions = workspace_aabb[1]
            pos = workspace_aabb[0] - dimensions / 2.0
            self._aabb = np.empty(6)
            self._aabb[:3] = pos
            self._aabb[3:] = pos + dimensions
        else:
            dimensions = workspace_aabb[3:] - workspace_aabb[:3]
            pos = workspace_aabb[:3]
            self._aabb = workspace_aabb
        self._num_cells = np.ceil(dimensions / cell_size).astype(int)
        # self._num_cells = np.array([int(math.ceil(x)) for x in dimensions / cell_size])
        self._base_pos = pos  # position of the bottom left corner with respect the local frame
        self._transform = np.eye(4)
        if base_transform is not None:
            self._transform = base_transform
        self._inv_transform = inverse_transform(self._transform)
        # first and last element per dimension is a dummy element for trilinear interpolation
        self._cells = np.zeros(self._num_cells + 2, dtype=dtype)
        self._homogeneous_point = np.ones(4)

    def __iter__(self):
        return self.get_cell_generator()

    def _fill_border_cells(self):
        """
            Fills the border cells with the values at the border of the interior of the cells.
        """
        self._cells[0, :, :] = self._cells[1, :, :]
        self._cells[-1, :, :] = self._cells[-2, :, :]
        self._cells[:, 0, :] = self._cells[:, 1, :]
        self._cells[:, -1, :] = self._cells[:, -2, :]
        self._cells[:, :, 0] = self._cells[:, :, 1]
        self._cells[:, :, -1] = self._cells[:, :, -2]

    def save(self, file_name):
        """
            Saves this grid under the given filename.
            Note that this function creates multiple files with different endings.
            @param file_name - filename
        """
        data_file_name = file_name + '.data.npy'
        meta_file_name = file_name + '.meta.npy'
        # first and last element per dimension are dummy elements
        np.save(data_file_name, self._cells[1:-1, 1:-1, 1:-1])
        np.save(meta_file_name, np.array([self._base_pos, self._cell_size, self._aabb, self._transform]))

    @staticmethod
    def load(file_name, b_restore_transform=False):
        """
            Load a grid from the given file.
            - :file_name: - as the name suggests
            - :b_restore_transform: (optional) - If true, the transform is loaded as well, else identity transform is set
        """
        data_file_name = file_name + '.data.npy'
        meta_file_name = file_name + '.meta.npy'
        if not os.path.exists(data_file_name) or not os.path.exists(meta_file_name):
            raise IOError("Could not load grid for filename prefix " + file_name)
        grid = VoxelGrid(np.array([0, 0, 0, 0, 0, 0]))
        interior_cells = np.load(data_file_name)
        grid._num_cells = np.array(np.array(interior_cells.shape))
        grid._cells = np.empty(grid._num_cells + 2)
        grid._cells[1:-1, 1:-1, 1:-1] = interior_cells
        grid._fill_border_cells()
        meta_data = np.load(meta_file_name)
        grid._base_pos = meta_data[0]
        grid._cell_size = meta_data[1]
        grid._aabb = meta_data[2]
        if b_restore_transform:
            grid._transform = meta_data[3]
        else:
            grid._transform = np.eye(4)
        return grid

    def get_index_generator(self):
        """
            Returns a generator that generates all indices of this grid.
        """
        return ((ix, iy, iz) for ix in xrange(self._num_cells[0])
                for iy in xrange(self._num_cells[1])
                for iz in xrange(self._num_cells[2]))

    def get_cell_generator(self):
        """
            Returns a generator that generates all cells in this grid
        """
        index_generator = self.get_index_generator()
        return (VoxelGrid.VoxelCell(self, idx) for idx in index_generator)

    def get_cell_idx(self, pos):
        """
            Returns the index triple of the voxel in which the specified position lies
            Returns None if the given position is out of bounds.
        """
        self._homogeneous_point[:3] = pos
        local_pos = np.dot(self._inv_transform, self._homogeneous_point)[:3]
        if (local_pos < self._aabb[:3]).any() or (local_pos >= self._aabb[3:]).any():
            return None
        local_pos -= self._base_pos
        local_pos /= self._cell_size
        return map(int, local_pos)

    def get_workspace(self):
        """
            Return the workspace that this VoxelGrid represents.
            DO NOT MODIFY!
        """
        return self._aabb

    def map_to_grid(self, pos, index_type=np.int):
        """
            Maps the given global position to local frame and returns both the local point
            and the index (None if out of bounds).
            @param index_type - Denotes what type the returned index should be. Should either be
                np.int or np.float_. By default integer indices are returned, if np.float_ is passed
                the returned index is a real number, which allows trilinear interpolation between grid points.
        """
        self._homogeneous_point[:3] = pos
        local_pos = np.dot(self._inv_transform, self._homogeneous_point)[:3]
        idx = None
        if (local_pos >= self._aabb[:3]).all() and (local_pos < self._aabb[3:]).all():
            idx = local_pos - self._base_pos
            idx /= self._cell_size
            if index_type != np.float_:
                idx = idx.astype(index_type)
        return local_pos, idx

    def map_to_grid_batch(self, positions, index_type=np.int):
        """
            Map the given global positions to local frame and return both the local points
            and the indices (None if out of bounds).
            @param positions is assumed to be a numpy matrix of shape (n, 4) where n is the number of query
                    points and the last row are 1s.
            @param index_type - Denotes what type the returned index should be. Should either be
                np.int or np.float_. By default integer indices are returned, if np.float_ is passed
                the returned index is a real number, which allows trilinear interpolation between grid points.
            @return (local_positions, indices, mask) where
                local_positions are the transformed points in a numpy array of shape (n, 4) where n
                    is the number of query points
                indices is a numpy array of shape (m, 3) containing indices for the m <= n valid local points,
                    or None if all points are out of bounds (in this case mask.any() is False)
                mask is a 1D numpy array of length n where mask[i] is True iff local_positions[i, :3] is within
                    bounds and indices contains an index for this position

        """
        local_positions = np.dot(positions, self._inv_transform.transpose())
        in_bounds_lower = (local_positions[:, :3] >= self._aabb[:3]).all(axis=1)
        in_bounds_upper = (local_positions[:, :3] < self._aabb[3:]).all(axis=1)
        mask = np.logical_and(in_bounds_lower, in_bounds_upper)
        if mask.any():
            indices = (local_positions[mask, :3] - self._base_pos) / self._cell_size
            if index_type != np.float_:
                indices = indices.astype(index_type)
        else:
            indices = None
        return local_positions, indices, mask

    def get_cell_position(self, idx, b_center=True):
        """
            Returns the position in R^3 of the center or min corner of the voxel with index idx
            @param idx - a tuple/list of length 3 (ix, iy, iz) specifying the voxel
            @param b_center - if true, it return the position of the center, else of min corner
            @return numpy.array representing the center or min corner position of the voxel
        """
        rel_pos = np.array(idx) * self._cell_size
        if b_center:
            rel_pos += np.array([self._cell_size / 2.0,
                                 self._cell_size / 2.0,
                                 self._cell_size / 2.0])
        local_pos = self._base_pos + rel_pos
        return np.dot(self._transform, np.array([local_pos[0], local_pos[1], local_pos[2], 1]))[:3]

    def get_cell_value(self, idx):
        """
            Returns the value of the specified cell.
            If the elements of idx are of type float, the cell value is computed
            using trilinear interpolation.
        """
        idx = self.sanitize_idx(idx)
        if (isinstance(idx[0], np.float_) or isinstance(idx[0], float)) and self._cells.dtype.type == np.float_:
            return self.get_interpolated_values(np.array([idx]))[0]
        return self._cells[idx[0] + 1, idx[1] + 1, idx[2] + 1]

    def get_cell_values(self, indices):
        """
            Returns value of the cells with the specified indices.
            If the elements of each index are of type float, the cell values are computed
            using trilinear interpolation. The indices are not checked for bounds!
            @param indices - a numpy array of type int or float with shape (n, 3), where n is the number of query indices.
        """
        if indices.dtype.type == np.float_ and self._cells.dtype.type == np.float_:
            return self.get_interpolated_values(indices)
        return self._cells[indices[:, 0] + 1, indices[:, 1] + 1, indices[:, 2] + 1]

    def get_interpolated_values(self, indices):
        """
            Return grid values for the given floating point indices.
            The values are interpolated using trilinear interpolation.
            @param indices - numpy array of type np.float_ with shape (n, 3), where n is the number of query indices
            @return values - numpy array of type np.float_ and shape (n,).
        """
        # first shift indices so that integer coordinates correspond to the center of the interior cells (substract 0.5)
        centered_coords = indices - 0.5
        # we are going to compute trilinear interpolation for each interior point
        floor_coords = np.floor(centered_coords)  # rounded down coordinates
        w = centered_coords - floor_coords  # fractional part of index, i.e. distance to rounded down integer
        wm = 1.0 - w  # 1 - fractional part of index
        # compute the indices of each corner
        corner_indices = (floor_coords[:, np.newaxis, :] + VoxelGrid._rel_grid_neighbor_mask).astype(int) + 1
        assert(corner_indices.shape == (indices.shape[0], 8, 3))
        # retrieve values for each corner - dummy + 1 is already added to the indices
        corner_values = self._cells[corner_indices[:, :, 0], corner_indices[:, :, 1], corner_indices[:, :, 2]]
        assert(corner_values.shape == (indices.shape[0], 8))
        # weights for bilinear interpolation - each row of this matrix has the weights for the 4 corners
        bi_weight_matrix = np.array((wm[:, 1] * wm[:, 0], wm[:, 1] * w[:, 0],
                                     w[:, 1] * wm[:, 0], w[:, 1] * w[:, 0])).transpose()
        assert(bi_weight_matrix.shape == (indices.shape[0], 4))
        # combine the bilinear interpolation results to trilinearly interpolated values
        return wm[:, 2] * np.sum(bi_weight_matrix * corner_values[:, :4], axis=1) + \
            w[:, 2] * np.sum(bi_weight_matrix * corner_values[:, 4:], axis=1)

    def get_num_cells(self):
        """
            Returns the number of cells this grid has in each dimension.
            @return (nx, ny, nz)
        """
        return tuple(self._num_cells)

    def get_raw_data(self):
        """
            Returns a reference to the underlying cell data structure.
            Use with caution!
        """
        return self._cells[1:-1, 1:-1, 1:-1]

    def set_raw_data(self, data):
        """
            Overwrites the underlying cell data structure with the provided one.
            Use with caution!
            @param data - a numpy array with the same shape as returned by get_raw_data()
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('The type of the provided data is invalid.' +
                             ' Must be numpy.ndarray, but it is %s' % str(type(data)))
        if data.shape != tuple(self._num_cells):
            raise ValueError("The shape of the provided data differs from this grid's shape." +
                             " Input shape is %s, required shape %s" % (str(data.shape), str(self._num_cells)))
        self._cells[1:-1, 1:-1, 1:-1] = data
        self._fill_border_cells()

    def get_cell_size(self):
        """
            Returns the cell size
        """
        return self._cell_size

    def sanitize_idx(self, idx, cast_type=None):
        """
            Ensures that the provided index is a valid index type.
            @param idx - index to sanitize
            @param cast_type - may be either None, float or int. If not None, the index is cast
                to this type.
        """
        if len(idx) != 3:
            raise ValueError("Provided index has invalid length (%i)" % len(idx))

        def check_type(x):
            return isinstance(x, float) or isinstance(x, int) or isinstance(x, np.float_)
        valid_type = reduce(operator.and_, map(check_type, idx), True)
        if not valid_type:
            raise ValueError("Indices must either be int or float. Instead, the types are: %s, %s, %s" %
                             tuple(map(type, idx)))
        if cast_type is not None:
            if cast_type == int:
                idx = map(int, idx)
            elif cast_type == float or cast_type == np.float_:
                idx = map(float, idx)
        return np.array(idx)

    def set_cell_value(self, idx, value):
        """
            Sets the value of the cell with given index.
            @param idx - tuple (ix, iy, iz)
            @param value - value to set the cell to
        """
        idx = self.sanitize_idx(idx, cast_type=int)
        self._cells[idx[0] + 1, idx[1] + 1, idx[2] + 1] = value

    def fill(self, min_idx, max_idx, value):
        """
            Fills all cells in the block min_idx, max_idx with value
        """
        min_idx = self.sanitize_idx(min_idx, cast_type=int) + 1
        max_idx = self.sanitize_idx(max_idx, cast_type=int) + 1
        self._cells[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] = value

    def get_max_value(self, min_idx=None, max_idx=None):
        """
            Return the maximal value in this grid.
            Optionally, a subset of the grid can be specified by setting min_idx and max_idx.
            @param min_idx - minimal cell index to check for maximum value (default 0, 0, 0)
            @param max_idx - exclusive maximal cell index to check for maximum value (default get_num_cells())
        """
        if min_idx is None:
            min_idx = np.zeros(3, dtype=int)
        if max_idx is None:
            max_idx = self._num_cells
        return np.max(self._cells[min_idx[0] + 1:max_idx[0] + 1, min_idx[1] + 1:max_idx[1] + 1, min_idx[2] + 1:max_idx[2] + 1])

    def get_min_value(self, min_idx=None, max_idx=None):
        """
            Return the minimal value in this grid.
            Optionally, a subset of the grid can be specified by setting min_idx and max_idx.
            @param min_idx - minimal cell index to check for minimal value (default 0, 0, 0)
            @param max_idx - exclusive maximal cell index to check for minimal value (default get_num_cells())
        """
        if min_idx is None:
            min_idx = np.zeros(3, dtype=int)
        if max_idx is None:
            max_idx = self._num_cells
        return np.min(self._cells[min_idx[0] + 1:max_idx[0] + 1, min_idx[1] + 1:max_idx[1] + 1, min_idx[2] + 1:max_idx[2] + 1])

    def get_aabb(self):
        """
            Returns the local axis aligned bounding box of this grid.
            This is essentially the bounding box passed to the constructor.
        """
        return np.array(self._aabb)

    def set_transform(self, transform):
        """
            Sets the transform for this grid.
        """
        if not np.equal(self._transform, transform).all():
            self._transform = transform
            self._inv_transform = inverse_transform(self._transform)

    def get_inverse_transform(self):
        """
            Returns the inverse transform of this grid.
        """
        return self._inv_transform

    def get_transform(self):
        """
            Returns the transform of this grid.
        """
        return self._transform


class ORVoxelGridVisualization(object):
    """
        This class allows to visualize a voxel grid using an OpenRAVE environment.
    """

    def __init__(self, or_env, voxel_grid):
        """
            Creates a new visualization of a voxel grid using openrave.
        """
        self._env = or_env
        self._voxel_grid = voxel_grid
        self._handles = []

    def update(self, min_sat_value=None, max_sat_value=None, style=0):
        """
            Updates this visualization to reflect the latest state of the underlying voxel grid.
            The voxels are colored according to their values. By default the color of a voxel
            is computed using linear interpolation between two colors min_color and max_color.
            The voxel with maximum value is colored with max_color and the voxel with minimum value
            is colored with min_color. This behaviour can be changed by providing min_sat_value
            and max_sat_value. If these values are provided, any cell with value <= min_sat_value is
            colored min_color and any cell with value >= max_sat_value is colored with max_color.
            Cells with values in range (min_sat_value, max_sat_value) are colored
            using linear interpolation.

            @param min_sat_value (optional) - minimal saturation value
            @param max_sat_value (optional) - maximal saturation value
            @param style (optional) - if 0, renders cells using 2d sprites, if 1, renders cells using 3d balls
                                      WARNING: Rendering many balls(cells) will crash OpenRAVE
        """
        self._handles = []
        values = [x.get_value() for x in self._voxel_grid]
        if min_sat_value is None:
            min_sat_value = min(values)
        if max_sat_value is None:
            max_sat_value = max(values)

        blue_color = np.array([0.0, 0.0, 1.0, 0.05])
        red_color = np.array([1.0, 0.0, 0.0, 0.05])
        positions = np.array([cell.get_position() for cell in self._voxel_grid])

        def compute_color(value):
            """
                Computes the color for the given value
            """
            rel_value = np.clip((value - min_sat_value) / (max_sat_value - min_sat_value), 0.0, 1.0)
            return (1.0 - rel_value) * red_color + rel_value * blue_color
        colors = np.array([compute_color(v) for v in values])
        # TODO we should read the conversion from pixels to workspace size from somwhere
        # and convert true cell size to it
        if style == 0:
            handle = self._env.plot3(positions, 20, colors)  # size is in pixel
        else:
            handle = self._env.plot3(positions, self._voxel_grid.get_cell_size / 2.0, colors, 1)
        self._handles.append(handle)

    def clear(self):
        """
            Clear visualization
        """
        self._handles = []


class SDF(object):
    """
        This class represents a signed distance field.
    """

    def __init__(self, grid, approximation_box=None):
        """
            Creates a new signed distance field.
            You may either create an SDF using a SDFBuilder or by loading it from file.
            In neither case you will have to call this constructor yourself.
            - :grid: a VoxelGrid storing all signed distances - used by SDFBuilder
            - :approximation_box: a box used for approximating distances outside of the grid
        """
        self._grid = grid
        self._approximation_box = approximation_box
        self._or_visualization = None
        if self._approximation_box is None and self._grid:
            self._approximation_box = self._grid.get_aabb()

    def set_transform(self, transform):
        """
            Set the transform for this sdf
            @param transform - numpy 4x4 transformation matrix
        """
        self._grid.set_transform(transform)

    def get_transform(self):
        """
            Returns the current transformation matrix.
        """
        return self._grid.get_transform()

    def _get_heuristic_distance_local(self, local_point):
        """
            Returns a heuristical shortest distance of the given point the closest obstacle surface.
            @param local_point - point as numpy array (x, y, z), assumed to be in local frame
        """
        projected_point = np.clip(local_point, self._approximation_box[:3], self._approximation_box[3:])
        rel_point = local_point - projected_point
        return np.linalg.norm(rel_point)

    def get_heuristic_distance(self, point):
        """
            Returns a heuristical shortest distance of the given point the closest obstacle surface.
            @param point - point as numpy array (x, y, z)
        """
        local_point, idx = self._grid.map_to_grid(point)
        return self._get_heuristic_distance_local(local_point)

    def get_distance(self, point):
        """
            Returns the shortest distance of the given point to the closest obstacle surface.
            @param point - point as a numpy array (x, y, z).
        """
        local_point, idx = self._grid.map_to_grid(point)
        if idx is not None:
            return self._grid.get_cell_value(idx)
        # the point is out of range of our grid, we need to approximate the distance
        return self._get_heuristic_distance_local(local_point)

    def get_distances(self, positions, b_interpolate=True):
        """
            Returns the shortest distance of the given points to the closest obstacle surface respectively.

            Arguments:
            positions - a numpy matrix of shape (n, 4), where n is the number of query points.
                        The positions are expected to be given in homogeneous world coordinates,
                        i.e. the last column is expected to be 1s.
            b_interpolate - if true, values are interpolated, otherwise nearest neighbor lookup is used
        """
        distances = np.zeros(positions.shape[0])
        index_type = np.float_ if b_interpolate else np.int
        local_points, grid_indices, valid_mask = self._grid.map_to_grid_batch(positions, index_type=index_type)
        # retrieve the distances for which we have a valid index
        if grid_indices is not None:  # in case we have any valid points
            distances[valid_mask] = self._grid.get_cell_values(grid_indices)
            # for the rest, apply heuristic
            inverted_mask = np.logical_not(valid_mask)
            # TODO we might be able to optimize this step a bit more by using more numpy batch operations
            distances[inverted_mask] = map(self._get_heuristic_distance_local, local_points[inverted_mask, :3])
        else:
            distances = map(self._get_heuristic_distance_local, local_points[:, :3])
        return distances

    def clear_visualization(self):
        """
            Clear the visualization of this distance field
        """
        self._or_visualization.clear()

    def save(self, file_name):
        """
            Save this distance field to a file.
            Note that this function may create several files with different endings attached to file_name
            @param file_name - file to store sdf in
        """
        grid_file_name = file_name + '.grid'
        self._grid.save(grid_file_name)
        meta_file_name = file_name + '.meta'
        meta_data = np.array([self._approximation_box])
        np.save(meta_file_name, meta_data)

    @staticmethod
    def load(filename):
        """
            Loads an sdf from file.
            :return: the loaded sdf or None, if loading failed
        """
        meta_data_filename = filename + '.meta.npy'
        if not os.path.exists(meta_data_filename):
            rospy.logwarn("Could not load SDF because meta data file " + meta_data_filename + " does not exist")
            return None
        meta_data = np.load(meta_data_filename)
        approximation_box = meta_data[0]
        try:
            grid = VoxelGrid.load(filename + '.grid')
        except IOError as io_err:
            rospy.logwarn("Could not load SDF because:" + str(io_err))
            return None
        return SDF(grid=grid, approximation_box=approximation_box)

    def visualize(self, env, safe_distance=None):
        """
            Visualizes this sdf in the given openrave environment.
            @param env - OpenRAVE environment to visualize the SDF in.
            @param safe_distance (optional) - if provided, the visualization colors cells that are more than
                    safe_distance away from any obstacle in the same way as obstacles that are infinitely far away.
        """
        if not self._or_visualization or self._or_visualization._env != env:
            self._or_visualization = ORVoxelGridVisualization(env, self._grid)
            self._or_visualization.update(max_sat_value=safe_distance)
        else:
            self._or_visualization.update(max_sat_value=safe_distance)

    def set_approximation_box(self, box):
        """
            Sets an approximation box. If get_distance is queried for a point outside of the underlying grid,
            the distance of the query point to this approximation box is computed.
        """
        self._approximation_box = box


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
            grid - 3d numpy array of type bool, will be modified
        """
        # compute for each cell whether it collides with anything
        self._compute_bcm_rec(np.array([0, 0, 0]), grid.get_num_cells(), grid, 0)
        # The above function can not detect the interior of meshes, therefore we need to fill holes
        filled_holes = scipy.ndimage.morphology.binary_fill_holes(grid.get_raw_data().astype(bool))
        grid.set_raw_data(filled_holes)
        self._body_manager.disable_bodies()

    def clear(self):
        """
            Clear all cached resources.
        """
        self._body_manager.clear()


class SDFBuilder(object):
    """
        An SDF builder builds a signed distance field for a given environment.
        If you intend to construct multiple SDFs with the same cell size, it is recommended to use a single
        SDFBuilder as this saves resource generation. It only checks collisions with enabled bodies.
    """

    def __init__(self, env, cell_size):
        """
            Creates a new SDFBuilder object.
            @param env - OpenRAVE environment this builder operates on.
            @param cell_size - The cell size of the signed distance field.
        """
        self._cell_size = cell_size
        self._occupancy_builder = OccupancyGridBuilder(env, cell_size)

    def _compute_sdf(self, grid):
        """
            Compute a signed distance field from the given occupancy grid.

            Arguments
            ---------
            grid - VoxelGrid that is an occupancy map. The cell type must be bool

            Return
            ---------
            distance grid - VoxelGrid with cells of type float. Each cell contains the signed
                distance to the closest obstacle surface point
        """
        # the grid is a binary collision map: 1 - collision, 0 - no collision
        # before calling skfmm we need to transform this map
        # raw_grid = grid.get_raw_data()
        # raw_grid *= -2.0
        # raw_grid += 1.0
        # min_value = grid.get_min_value()
        # if min_value > 0:  # there is no collision
        #     raw_grid[:, :, :] = float('Inf')
        # else:
        #     grid.set_raw_data(skfmm.distance(raw_grid, dx=grid.get_cell_size()))
        raw_grid = grid.get_raw_data()
        inverse_collision_map = np.invert(raw_grid)
        # compute the distances to surface of collision space
        outside_distances = scipy.ndimage.morphology.distance_transform_edt(inverse_collision_map,
                                                                            sampling=grid.get_cell_size())
        # compute the interior of the collision space (remove the surface, because that's where the distance shall be 0)
        interior_collision_map = scipy.ndimage.morphology.binary_erosion(raw_grid)
        inside_distances = scipy.ndimage.morphology.distance_transform_edt(interior_collision_map,
                                                                           sampling=grid.get_cell_size())
        print('min inside', np.min(inside_distances))
        print('max inisde', np.max(inside_distances))
        print('min outside', np.min(outside_distances))
        print('max outside', np.max(outside_distances))
        sdf_grid = VoxelGrid(grid.get_workspace(), cell_size=self._cell_size, dtype=np.float_)
        sdf_grid.set_raw_data(outside_distances - inside_distances)
        return sdf_grid

    def create_sdf(self, workspace_aabb):
        """
            Creates a new sdf for the current state of the OpenRAVE environment provided on construction.
            The SDF is created in world frame of the environment. You can later change its transform.
            NOTE: If you do not intend to continue creating more SDFs using this builder, call clear() afterwards.
            @param workspace_aabb - bounding box of the sdf in form of [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        occupancy_grid = VoxelGrid(workspace_aabb, cell_size=self._cell_size, dtype=bool)
        # First compute binary collision map
        start_time = time.time()
        self._occupancy_builder.compute_grid(occupancy_grid)
        print ('Computation of collision binary map took %f s' % (time.time() - start_time))
        # next compute sdf
        start_time = time.time()
        distance_grid = self._compute_sdf(occupancy_grid)
        print ('Computation of sdf took %f s' % (time.time() - start_time))
        return SDF(grid=distance_grid)

    def clear(self):
        """
            Clear all used resources.
        """
        self._occupancy_builder.clear()

    @staticmethod
    def compute_sdf_size(aabb, approx_error, radius=0.0):
        """
            Computes the required size of an sdf for a movable kinbody such
            that at the boundary of the sdf the relative error in distance estimation to the body's
            surface is bounded by approx_error.
            - :aabb: OpenRAVE bounding box of the object
            - :approx_error: Positive floating point number in (0, 1] denoting the maximal relative error
            - :radius: (optional) a positive floating point number that is the radius of an inscribing ball
                    centered at the object center
            - :return: a bounding box in the shape [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        scaling_factor = (1.0 - (1.0 - approx_error) * radius / np.linalg.norm(aabb.extents())) / approx_error
        scaled_extents = scaling_factor * aabb.extents()
        upper_point = aabb.pos() + scaled_extents
        lower_point = aabb.pos() - scaled_extents
        return np.array([lower_point[0], lower_point[1], lower_point[2],
                         upper_point[0], upper_point[1], upper_point[2]])


class SceneSDF(object):
    """
        A scene sdf is a signed distance field for a motion planning scene that contains
        multiple movable kinbodies (including a robot) and a potentially empty set of static obstacles.
        A scene sdf creates separate sdfs for the static obstacles and the movable objects.
        When querying a scene sdf, the returned distance takes the current state of the environment,
        i.e. the current poses of all movable kinbodies into account. Kinbodies with more degrees of freedom
        are currently not supported, i.e. robots should always be excluded if they are expected to change their
        configuration.
    """

    def __init__(self, env, movable_body_names, excluded_bodies=None, sdf_paths=None, radii=None):
        """
            Constructor for SceneSDF. In order to actually use a SceneSDF you need to
            either call load_sdf or create_sdf.
            @param env - OpenRAVE environment to use
            @param movable_body_names - list of names of kinbodies that can move
            @param excluded_bodies - a list of kinbody names to exclude, i.e. completely ignore for instance a robot
            @param sdf_paths - optionally a dictionary that maps kinbody name to filename to
                               load an sdf from for that body
            @param radii - optionally a dictionary that maps kinbody name to a radius
                           of an inscribing ball
        """
        self._env = env
        self._movable_body_names = list(movable_body_names)
        if excluded_bodies is None:
            excluded_bodies = []
        self._ignored_body_names = list(excluded_bodies)
        self._static_sdf = None
        self._body_sdfs = {}
        self._sdf_paths = sdf_paths
        self._radii = radii

    def _enable_body(self, name, b_enabled):
        """
            Disable/Enable the body with the given name
        """
        body = self._env.GetKinBody(name)
        if body is None:
            raise ValueError("Could not retrieve body with name %s from OpenRAVE environment" % body)
        body.Enable(b_enabled)

    def _compute_sdf_size(self, aabb, approx_error, body_name):
        """
            Computes the required size of an sdf for a movable kinbody such
            that at the boundary of the sdf the relative error in distance estimation to the body's
            surface is bounded by approx_error.
        """
        radius = 0.0
        if self._radii is not None and body_name in self._radii:
            radius = self._radii[body_name]
        return SDFBuilder.compute_sdf_size(aabb, approx_error, radius)

    def create_sdf(self, workspace_bounds, static_resolution=0.02, moveable_resolution=0.02,
                   approx_error=0.1):
        """
            Creates a new scene sdf. This process takes time!
            @param workspace_bounds - the volume of the environment this sdf should cover
            @param static_resolution - the resolution of the sdf for the static part of the world
            @param moveable_resolution - the resolution of sdfs for movable kinbodies
            @param approx_error - a relativ error between 0 and 1 that is allowed to occur
                                  at boundaries of movable kinbody sdfs
        """
        # before we do anything, save which bodies are enabled
        body_enable_status = {}
        for body in self._env.GetBodies():
            body_enable_status[body.GetName()] = body.IsEnabled()
        # now first we build a sdf for the static obstacles
        builder = SDFBuilder(self._env, static_resolution)
        for body_name in self._ignored_body_names:
            self._enable_body(body_name, False)
        # it only makes sense to build a static sdf, if we have static obstacles
        if len(self._ignored_body_names) + len(self._movable_body_names) < len(self._env.GetBodies()):
            # for that disable all movable objects
            for body_name in self._movable_body_names:
                self._enable_body(body_name, False)
            self._static_sdf = builder.create_sdf(workspace_bounds)
        # Now we build SDFs for all movable object
        # if we have different resolutions for static and movables, we need a new builder
        if static_resolution != moveable_resolution:
            builder.clear()
            builder = SDFBuilder(self._env, moveable_resolution)
        # first disable everything in the scene
        for body in self._env.GetBodies():
            body.Enable(False)
        # Next create for each movable body an individual sdf
        for body_name in self._movable_body_names:
            body_sdf = None
            body = self._env.GetKinBody(body_name)
            if self._sdf_paths is not None and body_name in self._sdf_paths:  # we have a path for a body sdf
                # load an sdf
                body_sdf = SDF.load(self._sdf_paths[body_name])
            # if we do not have an sdf to load for this body or failed at doing so, create a new
            if body_sdf is None:
                # Prepare sdf creation for this body
                body.Enable(True)
                old_tf = body.GetTransform()
                body.SetTransform(np.eye(4))  # set it to the origin
                aabb = body.ComputeAABB()
                body_bounds = np.zeros(6)
                body_bounds[:3] = aabb.pos() - aabb.extents()
                body_bounds[3:] = aabb.pos() + aabb.extents()
                # Compute the size of the sdf that we need to ensure the maximum relative error
                sdf_bounds = self._compute_sdf_size(aabb, approx_error, body_name)
                # create the sdf
                body_sdf = builder.create_sdf(sdf_bounds)
                body_sdf.set_approximation_box(body_bounds)  # set the actual body aabb as approx box
                body.SetTransform(old_tf)  # restore transform
                body.Enable(False)  # disable body again
                # finally, if we have a body path for this body, store it
                if self._sdf_paths is None:
                    self._sdf_paths = {}
                if body_name in self._sdf_paths:
                    body_sdf.save(self._sdf_paths[body_name])
            self._body_sdfs[body_name] = (body, body_sdf)
        builder.clear()
        # Finally enable all bodies
        for body in self._env.GetBodies():
            body.Enable(body_enable_status[body.GetName()])

    def get_distance(self, position):
        """
            Returns the signed distance from the specified position to the closest obstacle surface
        """
        min_distance = float('inf')
        for (body, body_sdf) in self._body_sdfs.itervalues():
            body_sdf.set_transform(body.GetTransform())
            min_distance = min(min_distance, body_sdf.get_distance(position))
        if self._static_sdf is not None:
            min_distance = min(self._static_sdf.get_distance(position), min_distance)
        return min_distance

    def get_distances(self, positions):
        """
            Returns the signed distance from the given positions to the closest obstacle surface
            @param positions - a numpy matrix of shape (n, 4) where n is the number of query positions.
                                Each position is assumed to be given in homogenous world coordinates, i.e.
                                the last column is assumend to be all 1s.
        """
        min_distances = np.full(positions.shape[0], float('inf'))
        for (body, body_sdf) in self._body_sdfs.itervalues():
            body_sdf.set_transform(body.GetTransform())
            min_distances = np.minimum(min_distances, body_sdf.get_distances(positions))
        if self._static_sdf is not None:
            min_distances = np.minimum(min_distances, self._static_sdf.get_distances(positions))
        return min_distances

    def save(self, filename, body_dir=None):
        """
            Saves this scene sdf under the specified path.
            @param filename - absolute filename to save this sdf in (must be a path to a file)
            @param body_dir - optionally a relative path w.r.t dir(filename) to save the body sdfs in
                              if not provided, the body sdfs are saved in the same directory as filename points to
        """
        base_name = os.path.basename(filename)
        if not base_name:
            raise ValueError("The provided filename %s is invalid. The filename must be a valid path to a file" % filename)
        dir_name = os.path.dirname(filename)
        if body_dir is None:
            body_dir = '.'
        if self._sdf_paths is None:
            self._sdf_paths = {}
        # now build a dictionary mapping body name to sdf (static for static sdf) and save sdfs
        sdf_paths = {}
        rel_paths = {}
        if self._static_sdf:
            static_file_name = base_name + '.static.sdf'
            static_file_path = dir_name + '/' + static_file_name
            # TODO We could have a name collision here, if the environment contains a kinbody called static
            sdf_paths['__static_sdf__'] = static_file_path
            rel_paths['__static_sdf__'] = './' + static_file_name
            if '__static_sdf__' not in self._sdf_paths:
                self._static_sdf.save(static_file_path)
        for (key, value) in self._sdf_paths:  # reuse the filenames we loaded things from
            sdf_paths[key] = value
        for (body, body_sdf) in self._body_sdfs.itervalues():
            if body.GetName() in sdf_paths:  # no need to save body sdfs we loaded
                continue
            # we need to save the body sdfs for which we don't have an sdf path
            body_sdf_filename = str(body.GetName()) + '.sdf'
            body_sdf_filepath = dir_name + '/' + body_dir + '/' + body_sdf_filename
            sdf_paths[body.GetName()] = body_sdf_filepath
            body_sdf.save(body_sdf_filepath)
            rel_paths[str(body.GetName())] = body_dir + '/' + body_sdf_filename
        with open(filename, 'w') as meta_file:
            yaml.dump(rel_paths, meta_file)
        self._sdf_paths = sdf_paths

    def load(self, filename):
        """
            Loads a scene sdf from the specified path.
        """
        dir_name = os.path.dirname(filename)
        with open(filename, 'r') as meta_file:
            self._body_sdfs = {}
            self._static_sdf = None
            # first read in realtive paths and make them absolute
            rel_paths = yaml.load(meta_file)
            self._sdf_paths = {}
            for (name, rel_path) in rel_paths.iteritems():
                self._sdf_paths[name] = dir_name + '/' + rel_path
            # next read in sdfs
            available_sdfs = {}
            for (name, path) in self._sdf_paths.iteritems():
                if name != '__static_sdf__':
                    body = self._env.GetKinBody(name)
                    if body is None:
                        rospy.logerr("Could not find kinbody %s" % name)
                        continue
                    self._body_sdfs[name] = (body, SDF.load(path))
                    available_sdfs[name] = True
                else:
                    self._static_sdf = SDF.load(path)
                    available_sdfs['__static_sdf__'] = True
            # verify we have all movable bodies
            for name in self._movable_body_names:
                if name not in available_sdfs:
                    raise IOError("Could not load sdf for kinbody %s" % name)
            if '__static_sdf__' not in available_sdfs and len(self._movable_body_names) < len(self._env.GetBodies()):
                raise IOError("Could not load sdf for static environment")


class ORSDFVisualization(object):
    """
        This class allows to visualize an SDF using an OpenRAVE environment.
    """

    def __init__(self, or_env):
        """
            Creates a new visualization of an SDF using openrave.
        """
        self._env = or_env
        self._handles = []

    def visualize(self, sdf, volume, resolution=0.1, min_sat_value=None, max_sat_value=None, alpha=0.1,
                  style='sprites'):
        """
            Samples the given sdf within the specified volume and visualizes the data.
            @param sdf - the signed distance field to visualize
            @param volume - the workspace volume in which the sdf should be visualized
            @param resolution (optional) - the resolution at which the sdf should be sampled.
            @param min_sat_value (optional) - all points with distance smaller than this will have the same color
            @param max_sat_value (optional) - all point with distance larger than this will have the same color
            @param alpha (optional) - alpha value for colors
            @param style (optional) - if sprites, renders cells using 2d sprites, if balls, renders cells using 3d balls,
                                     if boxes, renders cells using boxes.
                                      WARNING: Rendering many balls(cells) will crash OpenRAVE
        """
        # first sample values
        num_samples = (volume[3:] - volume[:3]) / resolution
        start_time = time.time()
        positions = np.array([np.array([x, y, z, 1.0]) for x in np.linspace(volume[0], volume[3], num_samples[0])
                              for y in np.linspace(volume[1], volume[4], num_samples[1])
                              for z in np.linspace(volume[2], volume[5], num_samples[2])])
        print ('Computation of positions took %f s' % (time.time() - start_time))
        start_time = time.time()
        # values = [sdf.get_distance(pos) for pos in positions]
        values = sdf.get_distances(positions)
        print ('Computation of distances took %f s' % (time.time() - start_time))
        # compute min and max
        # draw
        start_time = time.time()
        if min_sat_value is None:
            min_sat_value = min(values)
        if max_sat_value is None:
            max_sat_value = max(values)

        blue_color = np.array([0.0, 0.0, 1.0, alpha])
        red_color = np.array([1.0, 0.0, 0.0, alpha])
        zero_color = np.array([0.0, 0.0, 0.0, alpha])

        def compute_color(value):
            """
                Computes the color for the given value
            """
            # rel_value = np.clip((value - min_sat_value) / (max_sat_value - min_sat_value), 0.0, 1.0)
            # return (1.0 - rel_value) * red_color + rel_value * blue_color
            base_color = blue_color if value > 0.0 else red_color
            sat_value = max_sat_value if value > 0.0 else min_sat_value
            rel_value = np.clip(value / sat_value, 0.0, 1.0)
            return rel_value * base_color + (1.0 - rel_value) * zero_color

        colors = np.array([compute_color(v) for v in values])
        if style == 'sprites':
            handle = self._env.plot3(positions[:, :3], 10, colors)  # size is in pixel
            self._handles.append(handle)
        elif style == 'balls':
            # Instead we can also render balls, but this can easily crash OpenRAVE if done for many cells
            handle = self._env.plot3(positions[:, :3], resolution / 2.0, colors, 1)
            self._handles.append(handle)
        elif style == 'boxes':
            extents = [resolution / 2.0, resolution / 2.0, resolution / 2.0]
            for pos, color in izip(positions, colors):
                handle = self._env.drawbox(pos, extents, color)
                self._handles.append(handle)
        else:
            raise ValueError('Could not render sdf. Unknown style %s' % style)
        print ('Rendering took %f s' % (time.time() - start_time))
        # Alternatively, the following code would render the sdf using meshes, but this also kills openrave
        # colors = np.reshape([12 * [compute_color(v)] for v in values], (12 * values.shape[0], 4))
        # box_extents = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]) * resolution
        # vertices = np.zeros((8 * positions.shape[0], 3))
        # indices = np.zeros((12 * positions.shape[0], 3), dtype=np.int64)
        # box_vertices, box_indices = ComputeBoxMesh(box_extents)
        # for pos_idx in xrange(len(positions)):
        #     vertices[pos_idx * 8 : (pos_idx + 1) * 8] = box_vertices + positions[pos_idx, :3]
        #     indices[pos_idx * 12 : (pos_idx + 1) * 12] = box_indices + pos_idx * 12
        # handle = self._env.drawtrimesh(points=vertices,
        #                                indices=indices)
        #                             #    colors=colors)

    def clear(self):
        """
            Clear visualization
        """
        self._handles = []


# def visualize_grid(grid, min_sat_value=None, max_sat_value=None):
#     """
#         Visualize the given grid using mlab.
#         @param grid - the grid to visualize
#         @param min_sat_value (optional) - all points with distance smaller than this will have the same color
#         @param max_sat_value (optional) - all point with distance larger than this will have the same color
#     """
#     if min_sat_value is None:
#         min_sat_value = np.min(grid.get_raw_data())
#     if max_sat_value is None:
#         max_sat_value = np.max(grid.get_raw_data())
#     mlab.pipeline.volume(mlab.pipeline.scalar_field(grid.get_raw_data()), vmin=min_sat_value, vmax=max_sat_value)

# def clear_visualization():
#     """
#         Clear visualization.
#     """
#     mlab.clf()
