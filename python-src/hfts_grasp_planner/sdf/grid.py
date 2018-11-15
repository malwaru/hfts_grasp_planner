import os
import operator
import numpy as np
from hfts_grasp_planner.utils import inverse_transform
from scipy import ndimage


class VoxelGrid(object):
    _rel_grid_neighbor_mask = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], dtype=int)
    """"
        A voxel grid is a 3D discretization of a robot's workspace.
        For each voxel in this grid, this voxel grid saves a single floating point number.
        Additionally, it may store for each voxel optional data of any type.
    """
    class VoxelCell(object):
        """
            A voxel cell is a cell in a voxel grid and represents a voxel.
        """

        def __init__(self, grid, idx):
            self._grid = grid
            self._idx = idx

        def get_neighbor_index_iter(self):
            """
                Return an iterator over the valid neighbors of this cell.
            """
            for ix in xrange(max(0, self._idx[0] - 1), min(self._idx[0] + 2, self._grid._num_cells[0])):
                for iy in xrange(max(0, self._idx[1] - 1), min(self._idx[1] + 2, self._grid._num_cells[1])):
                    for iz in xrange(max(0, self._idx[2] - 1), min(self._idx[2] + 2, self._grid._num_cells[2])):
                        idx = (ix, iy, iz)
                        if idx != self._idx:
                            yield idx

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

        def get_additional_data(self):
            return self._grid.get_additional_data(self._idx)

        def set_additional_data(self, data):
            self._grid.set_additional_data(self._idx, data)

    def __init__(self, workspace_aabb,  cell_size=0.02, base_transform=None, dtype=np.float_, b_additional_data=False,
                 num_cells=None):
        """
            Creates a new voxel grid covering the specified workspace volume.
            @param workspace_aabb - bounding box of the workspace as numpy array of form
                                    [min_x, min_y, min_z, max_x, max_y, max_z] or a tuple
                                    ([x, y, z], [wx, wy, wz]), where x,y,z are the position of the center
                                    and wx, wy, wz are the dimensions of the box
            @param cell_size - cell size of the voxel grid (in meters)
            @param base_transform - if not None, any query point is transformed by base_transform
            b_additional_data - if True, each voxel can be associated with additional data of object type
            num_cells - if provided, the number of cells is not computed from the workspace aabb, but instead
                set to the given one. The actual workspace this grid covers spans then from the min position in
                workspace aabb to the point min_position + num_cells * cell_size.
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
        if num_cells is not None:
            self._num_cells = num_cells
            self._aabb[3:] = self._aabb[:3] + self._num_cells * cell_size
        else:
            self._num_cells = np.ceil(dimensions / cell_size).astype(int)
        # self._num_cells = np.array([int(math.ceil(x)) for x in dimensions / cell_size])
        self._base_pos = pos  # position of the bottom left corner with respect the local frame
        self._transform = np.eye(4)
        if base_transform is not None:
            self._transform = base_transform
        self._inv_transform = inverse_transform(self._transform)
        # first and last element per dimension is a dummy element for trilinear interpolation
        self._cells = np.zeros(self._num_cells + 2, dtype=dtype)
        self._additional_data = None
        if b_additional_data:
            self._additional_data = np.empty(self._num_cells, dtype=object)
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
        add_data_file_name = file_name + '.adddata.npy'
        meta_file_name = file_name + '.meta.npy'
        # first and last element per dimension are dummy elements
        np.save(data_file_name, self._cells[1:-1, 1:-1, 1:-1])
        np.save(meta_file_name, np.array([self._base_pos, self._cell_size, self._aabb, self._transform]))
        if self._additional_data is not None:
            np.save(add_data_file_name, self._additional_data)

    @staticmethod
    def load(file_name, b_restore_transform=False):
        """
            Load a grid from the given file.
            - :file_name: - as the name suggests
            - :b_restore_transform: (optional) - If true, the transform is loaded as well, else identity transform is set
        """
        data_file_name = file_name + '.data.npy'
        add_data_file_name = file_name + '.adddata.npy'
        meta_file_name = file_name + '.meta.npy'
        if not os.path.exists(data_file_name) or not os.path.exists(meta_file_name):
            raise IOError("Could not load grid for filename prefix " + file_name)
        grid = VoxelGrid(np.array([0, 0, 0, 0, 0, 0]))
        interior_cells = np.load(data_file_name)
        grid._num_cells = np.array(np.array(interior_cells.shape))
        grid._cells = np.empty(grid._num_cells + 2, dtype=interior_cells.dtype)
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
        if os.path.exists(add_data_file_name):
            grid._additional_data = np.load(add_data_file_name)
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

    def get_cell_idx(self, pos, b_pos_global=True):
        """
            Returns the index triple of the voxel in which the specified position lies
            Returns None if the given position is out of bounds.
            ---------
            Arguments
            ---------
            b_pos_global, bool - pass True if pos is in global frame. If it is in local frame, pass False
        """
        if b_pos_global:
            self._homogeneous_point[:3] = pos
            local_pos = np.dot(self._inv_transform, self._homogeneous_point)[:3]
        else:
            local_pos = pos
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

    def get_type(self):
        """
            Return the type of cell values.
        """
        return self._cells.dtype

    def get_subset(self, min_pos, max_pos):
        """
            Return a new VoxelGrid that only contains a subset of this VoxelGrid.
            The subset is identified by the provided arguments min_pos and max_pos.
            min_pos describes the lower corner and max_pos the upper corner of a bounding box.
            The returned VoxelGrid is a subset of this grid that is guaranteed to contain all elements
            that lie within the specified box. Since the axes of the provided box may be misaligned
            with this grid's local frame, the resulting grid may be larger than the queried box.
            ---------
            Arguments
            ---------
            min_pos, numpy array of shape (3,) - type float
            max_pos, numpy array of shape (3,) - type float
            -------
            Returns
            -------
            New voxel grid
        """
        # first map world positions to local frame
        local_a, _ = self.map_to_grid(min_pos, index_type=np.float_)
        local_b, _ = self.map_to_grid(max_pos, index_type=np.float_)
        # now make a bounding box of these positions in local frame
        min_pos = np.min((local_a, local_b), axis=0)
        max_pos = np.max((local_a, local_b), axis=0)
        # clip them to the valid range of this grid (no point in exceeding grid dimensions)
        min_pos = np.clip(min_pos, self._aabb[:3], self._aabb[3:] -
                          self._cell_size / 2.0)  # max value is outside of the grid
        max_pos = np.clip(max_pos, self._aabb[:3], self._aabb[3:] - self._cell_size / 2.0)
        # now retrieve the indices for these positions
        indices_a = np.array(self.get_cell_idx(min_pos, b_pos_global=False), dtype=int)
        indices_b = np.array(self.get_cell_idx(max_pos, b_pos_global=False), dtype=int)
        # compute how many cells there are in the new grid
        new_num_cells = indices_b - indices_a + 1  # including indices_b
        new_extents = new_num_cells * self._cell_size
        min_pos = self.get_cell_position(indices_a, b_center=False)  # get base position
        # create new grid
        new_grid = VoxelGrid((min_pos + new_extents / 2.0, new_extents),
                             cell_size=self._cell_size, base_transform=np.array(self._transform),
                             num_cells=new_num_cells)
        # set data of new grid
        assert((new_grid.get_num_cells() == new_num_cells).all())
        new_grid.set_raw_data(self._cells[indices_a[0] + 1:indices_b[0] + 2,
                                          indices_a[1] + 1:indices_b[1] + 2,
                                          indices_a[2] + 1:indices_b[2] + 2])
        return new_grid

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
            @param positions is assumed to be a numpy matrix of shape (n, 3) where n is the number of query
                    points.
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
        local_positions = np.dot(positions, self._inv_transform[:3, :3].transpose()) + self._inv_transform[:3, 3]
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

    def get_cell_position(self, idx, b_center=True, b_global_frame=True):
        """
            Returns the position in R^3 of the center or min corner of the voxel with index idx
            @param idx - a tuple/list of length 3 (ix, iy, iz) specifying the voxel
            @param b_center - if true, it return the position of the center, else of min corner
            @param b_global_frame - if true, return the position in global frame, else local frame
            @return numpy.array representing the center or min corner position of the voxel
        """
        rel_pos = np.array(idx) * self._cell_size
        if b_center:
            rel_pos += np.array([self._cell_size / 2.0,
                                 self._cell_size / 2.0,
                                 self._cell_size / 2.0])
        local_pos = self._base_pos + rel_pos
        if not b_global_frame:
            return local_pos
        return np.dot(self._transform, np.array([local_pos[0], local_pos[1], local_pos[2], 1]))[:3]

    def get_cell_positions(self, indices, b_center=True, b_global_frame=True):
        """
            Like get_cell_position but for a matrix of indices.
            ---------
            Arguments
            ---------
            indices, numpy array of shape (n, 3) and type int - query indices
            b_center, bool - if True (default) return positions of cell centers, else corners
            b_global_frame, bool - if True (default) return positions in global frame
            -------
            Returns
            -------
            positions, numpy array of shape (n, 3) and type float - requested cell positions
        """
        rel_pos = indices * self._cell_size
        if b_center:
            rel_pos = rel_pos + np.array([self._cell_size / 2.0,
                                          self._cell_size / 2.0,
                                          self._cell_size / 2.0])
        local_pos = self._base_pos + rel_pos
        if not b_global_frame:
            return local_pos
        return np.dot(local_pos, self._transform[:3, :3].transpose()) + self._transform[:3, 3]

    def get_grid_positions(self, b_center=True, b_global_frame=True):
        """
            Return arrays x, y, z of all cell positions.
            ---------
            Arguments
            ---------
            b_center, bool - if True (default), return positions of cell centers, else corners
            b_global_frame, bool - if True (default), return positions in global frame
            -------
            Returns
            -------
            x, numpy array of shape (n,m,l) with type float - x coordinates of cells
            y, numpy array of shape (n,m,l) with type float - y coordinates of cells
            z, numpy array of shape (n,m,l) with type float - z coordinates of cells
        """
        xs = np.linspace(self._aabb[0], self._aabb[3], self._num_cells[0] + 1)
        ys = np.linspace(self._aabb[1], self._aabb[4], self._num_cells[1] + 1)
        zs = np.linspace(self._aabb[2], self._aabb[5], self._num_cells[2] + 1)
        if b_center:
            xs += self._cell_size / 2.0
            ys += self._cell_size / 2.0
            zs += self._cell_size / 2.0
        xx, yy, zz = np.meshgrid(xs, ys, zs)
        if not b_global_frame:
            return xx, yy, zz
        # transform each coordinate separately: x' = r00 * x + r01 * y + r02 * z + b_x,...
        txx = self._transform[0, 0] * xx + self._transform[0, 1] * \
            yy + self._transform[0, 2] * zz + self._transform[0, 3]
        tyy = self._transform[1, 0] * xx + self._transform[1, 1] * \
            yy + self._transform[1, 2] * zz + self._transform[1, 3]
        tzz = self._transform[2, 0] * xx + self._transform[2, 1] * \
            yy + self._transform[2, 2] * zz + self._transform[2, 3]
        return txx, tyy, tzz

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

    def get_cell_values_pos(self, positions):
        """
            Returns the values of the cells with the specified positions (in world frame).
            If some positions are out of bounds, None is returned for these positions.
            ---------
            Arguments
            ---------
            positions, numpy array of shape (n, 3) - query positions
        """
        values = np.full((positions.shape[0],), None)
        _, grid_indices, valid_mask = self.map_to_grid_batch(positions, index_type=np.float_)
        if valid_mask.any():
            values[valid_mask] = self.get_cell_values(grid_indices)
        return values

    def get_additional_data(self, idx):
        """
            Return additional data for the given cell, if it exists.
            ---------
            Arguments
            ---------
            idx - a tuple/list of length 3 (ix, iy, iz) specifying the voxel
            -------
            Returns
            -------
            additoinal data for the cell, None if not available
        """
        if self._additional_data is not None:
            idx = self.sanitize_idx(idx)
            return self._additional_data[idx[0], idx[1], idx[2]]
        return None

    def set_additional_data(self, idx, data):
        """
            Set additional data for the given cell.
            ---------
            Arguments
            ---------
            idx - a tuple/list of length 3 (ix, iy, iz) specifying the voxel
            data, object - any pickable data to store
        """
        if self._additional_data is None:
            self._additional_data = np.empty(self._num_cells, dtype=object)
        idx = self.sanitize_idx(idx)
        self._additional_data[idx[0], idx[1], idx[2]] = data

    # def get_interpolated_values(self, indices):
    #     """
    #         Return grid values for the given floating point indices.
    #         The values are interpolated using trilinear interpolation.
    #         @param indices - numpy array of type np.float_ with shape (n, 3), where n is the number of query indices
    #         @return values - numpy array of type np.float_ and shape (n,).
    #     """
    #     # first shift indices so that integer coordinates correspond to the center of the interior cells (substract 0.5)
    #     centered_coords = indices - 0.5
    #     # we are going to compute trilinear interpolation for each interior point
    #     floor_coords = np.floor(centered_coords)  # rounded down coordinates
    #     w = centered_coords - floor_coords  # fractional part of index, i.e. distance to rounded down integer
    #     wm = 1.0 - w  # 1 - fractional part of index
    #     # compute the indices of each corner
    #     corner_indices = (floor_coords[:, np.newaxis, :] + VoxelGrid._rel_grid_neighbor_mask).astype(int) + 1
    #     assert(corner_indices.shape == (indices.shape[0], 8, 3))
    #     # retrieve values for each corner - dummy + 1 is already added to the indices
    #     corner_values = self._cells[corner_indices[:, :, 0], corner_indices[:, :, 1], corner_indices[:, :, 2]]
    #     assert(corner_values.shape == (indices.shape[0], 8))
    #     # weights for bilinear interpolation - each row of this matrix has the weights for the 4 corners
    #     bi_weight_matrix = np.array((wm[:, 1] * wm[:, 0], wm[:, 1] * w[:, 0],
    #                                  w[:, 1] * wm[:, 0], w[:, 1] * w[:, 0])).transpose()
    #     assert(bi_weight_matrix.shape == (indices.shape[0], 4))
    #     # combine the bilinear interpolation results to trilinearly interpolated values
    #     return wm[:, 2] * np.sum(bi_weight_matrix * corner_values[:, :4], axis=1) + \
    #         w[:, 2] * np.sum(bi_weight_matrix * corner_values[:, 4:], axis=1)

    def get_interpolated_values(self, indices):
        """
            Return grid values for the given floating point indices.
            The values are interpolated using trilinear interpolation.
            @param indices - numpy array of type np.float_ with shape (n, 3), where n is the number of query indices
            @return values - numpy array of type np.float_ and shape (n,).
        """
        return ndimage.map_coordinates(self._cells, indices.transpose() + 0.5, order=1)

    def get_num_cells(self):
        """
            Returns the number of cells this grid has in each dimension.
            @return (nx, ny, nz)
        """
        return tuple(self._num_cells)

    def get_raw_data(self, b_include_padding=False):
        """
            Returns a reference to the underlying cell data structure.
            Use with caution!
        """
        if not b_include_padding:
            return self._cells[1:-1, 1:-1, 1:-1]
        return self._cells

    def get_raw_additional_data(self):
        """
            Returns a reference to the underlying additional data data structure.
            This data structure is a numpy array of shape num_cells, but of type object.
        """
        return self._additional_data

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
        self._cells = self._cells.astype(data.dtype)
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

    def has_additional_data(self):
        """
            Return whether there is additional data stored in this grid.
        """
        return self._additional_data is not None


# if __name__ == "__main__":
#     import os

#     def test_interpolation():
#         grid = VoxelGrid.load(os.path.dirname(__file__) + '/../../../data/sdfs/placement_exp_0.sdf.static.sdf.grid')
#         query_indices = np.random.rand(100000, 3)
#         query_indices = query_indices * np.array(grid.get_num_cells())
#         values_a = grid.get_interpolated_values(query_indices)
#         values_b = grid.get_interpolated_values2(query_indices)
#         errors = np.abs(values_a - values_b)
#         bad_ones = np.where(errors > 0.1)
#         print grid.get_num_cells()
#         print bad_ones[0].shape
#         print query_indices[bad_ones[0]]
#         print np.min(errors), np.max(errors)
#         assert(np.allclose(values_a, values_b))
#         print "Passed!"
    # test_interpolation()
