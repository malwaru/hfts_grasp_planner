import os
import operator
import numpy as np
from hfts_grasp_planner.utils import inverse_transform
from hfts_grasp_planner.sdf.cuda_grid import CudaInterpolator, CudaStaticPositionsInterpolator
from scipy import ndimage

MIN_NUM_POINTS_CUDA = 20


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
                 num_cells=None, b_in_slices=False, b_use_cuda=False):
        """
            Creates a new voxel grid covering the specified workspace volume.
            @param workspace_aabb - bounding box of the workspace as numpy array of form
                                    [min_x, min_y, min_z, max_x, max_y, max_z] or a tuple
                                    ([x, y, z], [wx, wy, wz]), where x,y,z are the position of the center
                                    and wx, wy, wz are the dimensions of the box
            @param cell_size - cell size of the voxel grid (in meters)
            @param base_transform (optional) - transformation matrix from workspace frame to some world frame (default identity)
            b_additional_data - if True, each voxel can be associated with additional data of object type
            num_cells - if provided, the number of cells is not computed from the workspace aabb, but instead
                set to the given one. The actual workspace this grid covers spans then from the min position in
                workspace aabb to the point min_position + num_cells * cell_size.
            b_in_slices, bool - if True, the grid is considered to represent data that is independent from each
                other along the z-axis. This only matters for interpolation, where in the True-case interpolation
                is only performed between cell values that have the same z-index, i.e. bilinear interpolation in
                the x,y plane. Otherwise, the interpolation is trilinear.
            b_use_cuda, bool - if True, utilize Cuda-based interpolator.
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
        self._b_in_slices = b_in_slices
        if b_additional_data:
            self._additional_data = np.empty(self._num_cells, dtype=object)
        self._homogeneous_point = np.ones(4)
        self._b_use_cuda = b_use_cuda
        self._cuda_static_interpolators = []
        self._init_interpolator()

    def __iter__(self):
        return self.get_cell_generator()

    def _init_interpolator(self):
        """
            Update Cuda interpolators in case of changes to data.
        """
        if self.supports_cuda_queries():
            self._cuda_interpolator = CudaInterpolator(self._cells, self._b_in_slices)
            # if we have static interpolators, notify them that data has changed
            if len(self._cuda_static_interpolators) > 0:
                for interp in self._cuda_static_interpolators:
                    interp.reset_data(self._cells)
        else:
            self._cuda_interpolator = None
            self._cuda_static_interpolators = []

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
    def load(file_name, b_restore_transform=False, b_use_cuda=False):
        """
            Load a grid from the given file.
            ---------
            Arguments
            ---------
            file_name: - as the name suggests
            b_restore_transform: (optional) - If true, the transform is loaded as well, else identity transform is set
            b_use_cuda: (optional) - If true, enable Cuda-based interpolation
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
        grid._b_use_cuda = b_use_cuda
        grid._init_interpolator()
        return grid

    def supports_cuda_queries(self):
        """
            Return whether Cuda queries are supported.
        """
        return self._b_use_cuda and (self._cells.dtype == np.float64 or self._cells.dtype == np.float32)

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
            -------
            Returns
            -------
            np.array of shape (6,) where aabb[:3] is the min point and aabb[3:] the max point
        """
        return self._aabb

    def get_aabb(self, bWorld=False):
        """
            Return axis aligned bounding box.
            ---------
            Arguments
            ---------
            bWorld, bool - if True return it in world frame, else in local frame.
            -------
            Returns
            -------
            np.array of shape (6,) where aabb[:3] is the min point and aabb[3:] the max point
        """
        if bWorld:
            global_aabb = np.dot(self._transform[:3, :3], self._aabb.reshape(
                (2, 3)).transpose()).transpose() + self._transform[:3, 3]
            return global_aabb.reshape((6,))
        return np.array(self._aabb)

    def get_type(self):
        """
            Return the type of cell values.
        """
        return self._cells.dtype

    def set_type(self, dtype):
        """
            Attempts to cast underlying type to the given one.
        """
        self._cells = self._cells.astype(dtype)

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
                             num_cells=new_num_cells, b_in_slices=self._b_in_slices,
                             b_use_cuda=self._b_use_cuda)
        # set data of new grid
        assert((new_grid.get_num_cells() == new_num_cells).all())
        new_grid.set_raw_data(self._cells[indices_a[0] + 1:indices_b[0] + 2,
                                          indices_a[1] + 1:indices_b[1] + 2,
                                          indices_a[2] + 1:indices_b[2] + 2])
        return new_grid

    def enlarge(self, padding, fill_value):
        """
            Return an enlarged version of this VoxelGrid. The returned voxel grid contains
            a copy of this grid's data, padded with cells containing fill_value.
            The workspace of the enlarged VoxelGrid is
            this grid's workspace +- ceil(padding / self.get_cell_size())
            ---------
            Arguments
            ---------
            padding, float - padding value (minimal enlargement) in each direction
            fill_value, value type - value to fill outer new cells with
            -------
            Returns
            -------
            voxel_grid, VoxelGrid - a new VoxelGrid that for the positions that lie in
                this grid's workspace contains a copy of the values from this grid, and
                for positions outside of this grid's workspace fill_value.
            # TODO does not support additional data yet
        """
        new_workspace = np.array(self._aabb)
        new_offset_cells = np.ceil(padding / self._cell_size)
        new_num_cells = self._num_cells + 2 * new_offset_cells
        new_workspace[:3] -= new_offset_cells * self._cell_size

        new_voxel_grid = VoxelGrid(new_workspace, self._cell_size,
                                   num_cells=new_num_cells.astype(int), base_transform=self._transform,
                                   dtype=self._cells.dtype, b_in_slices=self._b_in_slices, b_use_cuda=self._b_use_cuda)
        # get raw_data
        new_data = new_voxel_grid.get_raw_data()
        new_data[:, :, :] = fill_value  # just initialize everything with fill value
        # next copy data from this grid
        min_idx = int(new_offset_cells)
        max_idx = (new_offset_cells + self._num_cells).astype(int)
        new_data[min_idx:max_idx[0], min_idx:max_idx[1], min_idx:max_idx[2]] = self._cells[1:-1, 1:-1, 1:-1]
        new_voxel_grid.set_raw_data(new_data)
        return new_voxel_grid

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

    def map_to_grid_batch(self, positions, index_type=np.int, b_global_frame=True):
        """
            Map the given global positions to local frame and return both the local points
            and the indices (None if out of bounds).
            ---------
            Arguments
            ---------
            positions, numpy matrix of shape (n, 3) where n is the number of query points.
            index_type, np.dtype - Denotes what type the returned index should be. Should either be
                np.int or np.float_. By default integer indices are returned, if np.float_ is passed
                the returned index is a real number, which allows trilinear interpolation between grid points.
                b_global_frame, bool
            b_global_frame, bool - if True, positions are assumed to be in global frame and transformed to local
                frame, else they are assumed to be in local frame.
            -------
            Returns
            -------
            (local_positions, indices, mask) where
                local_positions are the transformed points in a numpy array of shape (n, 4) where n
                    is the number of query points
                indices is a numpy array of shape (m, 3) containing indices for the m <= n valid local points,
                    or None if all points are out of bounds (in this case mask.any() is False)
                mask is a 1D numpy array of length n where mask[i] is True iff local_positions[i, :3] is within
                    bounds and indices contains an index for this position

        """
        if b_global_frame:
            local_positions = np.dot(positions, self._inv_transform[:3, :3].transpose()) + self._inv_transform[:3, 3]
        else:
            local_positions = positions
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
        xs = np.linspace(self._aabb[0], self._aabb[3], self._num_cells[0], endpoint=False)
        ys = np.linspace(self._aabb[1], self._aabb[4], self._num_cells[1], endpoint=False)
        zs = np.linspace(self._aabb[2], self._aabb[5], self._num_cells[2], endpoint=False)
        if b_center:
            xs += self._cell_size / 2.0
            ys += self._cell_size / 2.0
            zs += self._cell_size / 2.0
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
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
        query_indices = indices
        if indices.dtype == np.float:
            query_indices = indices.astype(int)
        return self._cells[query_indices[:, 0] + 1, query_indices[:, 1] + 1, query_indices[:, 2] + 1]

    def get_cell_values_pos(self, positions, b_global_frame=True):
        """
            Returns the values of the cells with the specified positions (in world frame).
            If some positions are out of bounds, None is returned for these positions.
            ---------
            Arguments
            ---------
            positions, numpy array of shape (n, 3) - query positions
            b_global_fame, bool - if True, positions are in global frame and accordingly first
                transformed into local frame.
        """
        if positions.shape[0] > MIN_NUM_POINTS_CUDA and self.supports_cuda_queries():
            return self.get_cell_gradients_pos_cuda(positions, b_global_frame)
        values = np.full((positions.shape[0],), None)
        _, grid_indices, valid_mask = self.map_to_grid_batch(positions, index_type=np.float_,
                                                             b_global_frame=b_global_frame)
        if valid_mask.any():
            values[valid_mask] = self.get_cell_values(grid_indices)
        return values

    def get_cell_values_pos_cuda(self, positions, b_global_frame=True):
        """
            Just like get_cell_values_pos(..) but utilizes a CUDA accelerated version.
            -----
            # TODO currently returns all True for valid flags
        """
        assert(self.supports_cuda_queries())
        tf = np.array(self._inv_transform)
        if not b_global_frame:
            tf = np.eye(4)
        tf[:3, 3] -= self._base_pos
        values = self._cuda_interpolator.interpolate(positions, tf_matrix=tf,
                                                     scale=1.0/self._cell_size)
        return values

    def get_cuda_position_interpolator(self, positions):
        """
            Return an object that allows to quickly retrieve values and gradients from this grid
            for a fixed set of positions that share a common transformation matrix to the world frame.
            ---------
            Arguments
            ---------
            positions, np.array of shape (n, 3), float - positions to query values for w.r.t to some frame
                that may change w.r.t the world frame.
            -------
            Returns
            -------
            CudaStaticPositionsInterpolator
        """
        assert(self.supports_cuda_queries())
        base_tf = np.array(self._inv_transform)
        base_tf[:3, 3] -= self._base_pos
        delta = self._cell_size / 4.0
        scale = 1.0 / self._cell_size
        self._cuda_static_interpolators.append(CudaStaticPositionsInterpolator(
            self._cells, positions, base_tf, scale, delta))
        return self._cuda_static_interpolators[-1]

    def get_cell_gradients_pos_cuda(self, positions, b_global_frame=True, b_return_values=True):
        """
            Just like get_cell_gradients_pos(..) but utilizes a CUDA accelerated version.
            -----
            # TODO currently returns all True for valid flags
        """
        assert(self.supports_cuda_queries())
        tf = np.array(self._inv_transform)
        if not b_global_frame:
            tf = np.eye(4)
        tf[:3, 3] -= self._base_pos
        delta = self._cell_size / 4.0
        values, gradients = self._cuda_interpolator.gradient(positions, tf_matrix=tf,
                                                             scale=1.0/self._cell_size, delta=delta)
        # values = self._cuda_interpolator.interpolate(positions, tf_matrix=tf, scale=1.0 / self._cell_size)
        valid_flags = np.full(positions.shape[0], True)  # TODO change Cuda interpolator to actually return valid flags
        if b_return_values:
            return valid_flags, values, gradients
        return valid_flags, gradients

    def get_cell_gradients_pos(self, positions, b_global_frame=True, b_return_values=True):
        """
            Return the gradients and optionally the values at the given positions
            NOTE: This function throws a ValueError if the stored data type is not float.
            ---------
            Arguments
            ---------
            positions, np array of shape (n, 3) - query positions
            b_global_frame, bool - if True, positions are expected to be in world frame, else local.
            b_return_values, bool - if True, also return values, else only gradients
            -------
            Returns
            -------
            valid_flags, np array of shape (n, 3) - bool array storing whether ith position was within bounds
            values(optional), np array of shape (m,) - values associated with m <= n valid positions if b_return_values == True
            gradients, np.array of shape (m, 3) - gradients of values associated with m <= n valid positions
        """
        if self._cells.dtype != np.float_:
            raise ValueError("Can not compute gradients for anything else than floats")
        num_points = positions.shape[0]
        if self._b_use_cuda and num_points > MIN_NUM_POINTS_CUDA:
            return self.get_cell_gradients_pos_cuda(positions, b_global_frame, b_return_values)
        query_width = 7 if b_return_values else 6
        valid_masks = np.empty((query_width, num_points), dtype=bool)
        grid_indices = np.zeros((query_width, num_points, 3))
        delta = self._cell_size / 4.0
        delta_x = np.array([delta, 0.0, 0.0])
        delta_y = np.array([0.0, delta, 0.0])
        delta_z = np.array([0.0, 0.0, delta])
        if b_return_values:
            deltas = np.array([delta_x, -delta_x, delta_y, -delta_y, delta_z, -delta_z, np.zeros(3)])
        else:
            deltas = np.array([delta_x, -delta_x, delta_y, -delta_y, delta_z, -delta_z])
        for i in xrange(query_width):
            _, tgrid_indices, valid_masks[i] = self.map_to_grid_batch(positions + deltas[i], index_type=np.float_,
                                                                      b_global_frame=b_global_frame)
            grid_indices[i, valid_masks[i]] = tgrid_indices

        valid_mask = np.logical_and.reduce(valid_masks, axis=0)
        num_valid = np.sum(valid_mask)
        if num_valid:
            # if we have valid points, retrieve their values
            valid_indices = grid_indices[:, valid_mask]
            values = self.get_cell_values(valid_indices.reshape((query_width * num_valid, 3)))
            values = values.reshape((query_width, num_valid))
            gradients = np.empty((num_valid, 3))
            gradients[:, 0] = (values[0, :] - values[1, :]) / (2.0 * delta)
            gradients[:, 1] = (values[2, :] - values[3, :]) / (2.0 * delta)
            gradients[:, 2] = (values[4, :] - values[5, :]) / (2.0 * delta)
            if b_return_values:
                return valid_mask, values[6], gradients
            return valid_mask, gradients
        else:
            if b_return_values:
                return valid_mask, np.empty((0,)), np.empty((0, 3))
            return valid_mask, np.empty((0, 3))

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
        # integer indices correspond to the center of cells!
        if not self._b_in_slices:  # trilinear interpolation
            # + 0.5 because of padding cells, else we would need to do -0.5
            return ndimage.map_coordinates(self._cells, indices.transpose() + 0.5, order=1)
        # bilinear interpolation
        # perpare output values
        values = np.empty((indices.shape[0],))
        # 1. get integer z coordinates
        z_integer = indices[:, 2].astype(int)
        # for each z layer
        for z in z_integer:
            indices_with_z = indices[:, 2].astype(int) == z
            # do bilinear interpolation
            values[indices_with_z] = ndimage.map_coordinates(
                self._cells[:, :, z + 1], indices[indices_with_z, :2].transpose() + 0.5, order=1)
        return values

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
        self._cells = self._cells.astype(data.dtype)
        self._cells[1:-1, 1:-1, 1:-1] = data
        self._fill_border_cells()
        self._init_interpolator()

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


class VectorGrid(object):
    """
        A VectorGrid is somehow similar to a VoxelGrid, except that it stores a 3D vector for each cell.
        It also supports trilinear interpolation. The interface is significantly slimmer though.
        You can directly access the underlying numpy array of shape (3, nx, ny, nz) through
        grid.vectors. To read vectors with interpolated values, call get_interpolated_vectors(..).
    """

    def __init__(self, workspace, cell_size=0.02, base_transform=None, num_cells=None, b_in_slices=False):
        """
            Create a new VectorGrid covering the given workspace.
            ---------
            Arguments
            ---------
            workspace, np.array of shape (6,) - (minx, miny, minz, maxx, maxy, maxz)
            cell_size, float - size of a single voxel
            base_transform, np.array of shape (4, 4) - transformation matrix from workspace
                frame to some global frame
            num_cells, tuple of length 3 - number of cells. If provided, the actual workspace is computed
                as (workspace[:3], workspace[3:] + num_cells * cell_size)
            b_in_slices, bool - if True, this grid stores a stack of 2D vector fields, else 3D Vectors
        """
        self._aabb = np.array(workspace)
        if base_transform is None:
            self._transform = np.eye(4)
        else:
            self._transform = np.array(base_transform)
        self._inv_transform = inverse_transform(self._transform)
        if num_cells is not None:
            self._num_cells = num_cells
            self._aabb[3:] = self._aabb[:3] + self._num_cells * cell_size
        else:
            self._num_cells = np.ceil((self._aabb[3:] - self._aabb[:3]) / cell_size).astype(int)
        if b_in_slices:
            self._vec_dim = 2
        else:
            self._vec_dim = 3
        shape = (self._vec_dim, self._num_cells[0], self._num_cells[1], self._num_cells[2])
        self.vectors = np.zeros(shape)
        self._cell_size = cell_size

    def get_aabb(self, bWorld=False):
        """
            Return axis aligned bounding box.
            ---------
            Arguments
            ---------
            bWorld, bool - if True return it in world frame, else in local frame.
            -------
            Returns
            -------
            np.array of shape (6,) where aabb[:3] is the min point and aabb[3:] the max point
        """
        if bWorld:
            global_aabb = np.dot(self._transform[:3, :3], self._aabb.reshape(
                (2, 3)).transpose()).transpose() + self._transform[:3, 3]
            return global_aabb.reshape((6,))
        return np.array(self._aabb)

    def save(self, filename):
        """
            Save this voxel grid to file.
            ---------
            Arguments
            ---------
            filename, string - path to file
        """
        # first and last element per dimension are dummy elements
        data_to_save = np.array([self.vectors, self._transform, self._aabb, self._cell_size, self._vec_dim])
        np.save(filename, data_to_save)

    @staticmethod
    def load(file_name, b_restore_transform=True):
        """
            Load a grid from the given file.
            - :file_name: - as the name suggests
            - :b_restore_transform: (optional) - If true, the transform is loaded as well, else identity transform is set
        """
        if not os.path.exists(file_name):
            raise IOError("Could not find file %s to load vector grid from." % file_name)
        grid_data = np.load(file_name)
        grid = VectorGrid(grid_data[2])
        grid.vectors = grid_data[0]
        grid._cell_size = grid_data[3]
        grid._transform = grid_data[1]
        grid._inv_transform = inverse_transform(grid._transform)
        grid._vec_dim = grid_data[4]
        grid._num_cells = np.ceil((grid._aabb[3:] - grid._aabb[:3]) / grid._cell_size).astype(int)
        return grid

    def get_interpolated_vectors(self, positions, b_world_frame=True):
        """
            Return interpolated vectors for the given positions.
            ---------
            Arguments
            ---------
            positions, numpy array of shape (n, 3) - query positions
            b_world_frame, bool - if True, positions are mapped to local frame first
            -------
            Returns
            -------
            valid_flags, np.array of shape (n,) of type bool - element i is True if position[i]
                lies within the workspace of this grid, else False
            vectors, np.array of shape (m, v) - interpolated vectors for the m<=n positions that
                lie inside this grid's workspace. v is either 3 (b_sliced=False) or 2 (b_sliced=True).
        """
        if b_world_frame:
            # map to local frame (recall that self._transform[:3, :3] = self._inv_transform[:3, :3].transpose())
            positions = np.dot(positions, self._transform[:3, :3]) + self._inv_transform[:3, 3]
        lower_bound_ok = np.logical_and.reduce(positions >= self._aabb[:3], axis=1)
        upper_bound_ok = np.logical_and.reduce(positions < self._aabb[3:], axis=1)
        valid_flags = np.logical_and(lower_bound_ok, upper_bound_ok)
        return_values = np.empty((self._vec_dim, 0))
        if valid_flags.any():
            grid_indices = (positions[valid_flags] - self._aabb[:3]) / self._cell_size
            return_values = np.empty((self._vec_dim, grid_indices.shape[0]))
            # - 0.5 for center of cells
            tindices = grid_indices.transpose() - 0.5
            return_values[0] = ndimage.map_coordinates(self.vectors[0], tindices, order=1, mode='nearest')
            return_values[1] = ndimage.map_coordinates(self.vectors[1], tindices, order=1, mode='nearest')
            if self._vec_dim == 3:
                return_values[2] = ndimage.map_coordinates(self.vectors[2], tindices, order=1, mode='nearest')
        if not np.isclose(self._transform[:3, :3], np.eye(3)).all():
            rotated_values = np.empty_like(return_values)
            rotated_values[0] = self._transform[0, 0] * return_values[0] + self._transform[0, 1] * return_values[1]
            rotated_values[1] = self._transform[1, 0] * return_values[0] + self._transform[1, 1] * return_values[1]
            if self._vec_dim == 3:
                rotated_values[0] += self._transform[0, 2] * return_values[2]
                rotated_values[1] += self._transform[1, 2] * return_values[2]
                rotated_values[2] = self._transform[2, 0] * return_values[0] + self._transform[2, 1] * return_values[1] + \
                    self._transform[2, 2] * return_values[2]
            return_values = rotated_values
        return valid_flags, return_values.transpose()
