import itertools
import numpy as np
import scipy.ndimage.morphology
import hfts_grasp_planner.sdf.occupancy as occupancy
import hfts_grasp_planner.sdf.grid as grid_module
import hfts_grasp_planner.sdf.core as sdf_core
import hfts_grasp_planner.clearance_utils as clearance_utils
from hfts_grasp_planner.placement.goal_sampler.interfaces import PlacementObjective

"""
    This module provides all functionality related to clearance maps.
    A clearance map is a scalar field (VoxelGrid) that maps a point to its clearance from obstacles:
        d_c(p) = d_xy(p) + d_xyz(p),
    where d_xy is the shortest distance to any obstacle within the x,y plane
    and d_xyz the shortest distance to any obstacle.
    A clearance map is created from an occupancy grid of the workspace, and it is assumed
    that the z-axis of the grid is pointing upwards, i.e. is antiparallel to gravity.
"""


def compute_clearance_map(occ_grid, buse_cuda=True):
    """
        Compute a clearance map from the given voxel grid.
        ---------
        Arguments
        ---------
        occ_grid, VoxelGrid - representing occupancy grid of the environment. The cell type must be bool!
        -------
        Returns
        -------
        clearance_map, VoxelGrid - grid representing the clearance map
    """
    clearance_map = grid_module.VoxelGrid(np.array(occ_grid.get_workspace()),
                                          num_cells=np.array(occ_grid.get_num_cells()),
                                          cell_size=occ_grid.get_cell_size(),
                                          b_use_cuda=True)
    clrm_data = clearance_map.get_raw_data()
    occ_data = occ_grid.get_raw_data()
    if occ_data.dtype != bool:
        occ_data = occ_data.astype(bool)
    inv_occ_data = np.invert(occ_data)
    adj_mask = np.ones((3, 3, 3), dtype=bool)
    adj_mask[:, :, 2] = 0  # only propagate distances downwards and to the sides
    an_arr = np.empty_like(clrm_data)
    clearance_utils.compute_df(inv_occ_data, adj_mask, an_arr)
    clrm_data = an_arr * occ_grid.get_cell_size()
    # max_distance = np.min(np.array(occ_data.shape[:2]) * occ_grid.get_cell_size()) / 2.0  # TODO what to set here?
    # clrm_data[clrm_data == np.inf] = max_distance
    # fix infinite distances by copying bottom layers up
    for z in range(1, clrm_data.shape[2]):
        inf_vals = clrm_data[:, :, z] == np.inf
        if (inf_vals).any():
            assert(inf_vals.all())
            clrm_data[:, :, z] = clrm_data[:, :, z - 1]
    clearance_map.set_raw_data(clrm_data)
    return clearance_map

    #     # run over each layer
    #     for idx in xrange(occ_transposed.shape[0]):
    #         if np.any(occ_transposed[idx]):
    #             inv_occ_layer = np.invert(occ_transposed[idx])
    #             xy_distance_field = scipy.ndimage.morphology.distance_transform_edt(
    #                 inv_occ_layer, sampling=sdf_grid.get_cell_size())
    #         else:
    #             xy_distance_field = np.full(occ_transposed[idx].shape, max_distance)
    #         clrm_data_t[idx] = xy_distance_field + sdf_data_t[idx]
    #     clearance_map.set_raw_data(clrm_data)
    #     return clearance_map

    # def compute_clearance_map(occ_grid, sdf_grid=None):
    #     """
    #         Compute a clearance map from the given voxel grid.
    #         ---------
    #         Arguments
    #         ---------
    #         occ_grid, VoxelGrid - representing occupancy grid of the environment. The cell type must be bool!
    #         sdf_grid, VoxelGrid (optional) - signed distance field along x,y,z computed from occ_grid
    #             If not provided, this function computes this field.
    #         -------
    #         Returns
    #         -------
    #         clearance_map, VoxelGrid - grid representing the clearance map
    #     """
    #     if sdf_grid is None:
    #         sdf_grid = sdf_core.SDFBuilder.compute_sdf(occ_grid)
    #     # next, compute sdf for each slice
    #     clearance_map = grid_module.VoxelGrid(np.array(occ_grid.get_workspace()),
    #                                           num_cells=np.array(occ_grid.get_num_cells()),
    #                                           cell_size=occ_grid.get_cell_size())
    #     occ_transposed = occ_grid.get_raw_data().transpose()  # need to transpose for distance_transform
    #     clrm_data = clearance_map.get_raw_data()
    #     clrm_data_t = clrm_data.transpose()
    #     sdf_data_t = sdf_grid.get_raw_data().transpose()
    #     max_distance = np.min(np.array(occ_transposed[0].shape) * occ_grid.get_cell_size()) / 2.0  # TODO what to set here?
    #     # run over each layer
    #     for idx in xrange(occ_transposed.shape[0]):
    #         if np.any(occ_transposed[idx]):
    #             inv_occ_layer = np.invert(occ_transposed[idx])
    #             xy_distance_field = scipy.ndimage.morphology.distance_transform_edt(
    #                 inv_occ_layer, sampling=sdf_grid.get_cell_size())
    #         else:
    #             xy_distance_field = np.full(occ_transposed[idx].shape, max_distance)
    #         clrm_data_t[idx] = xy_distance_field + sdf_data_t[idx]
    #     clearance_map.set_raw_data(clrm_data)
    #     return clearance_map


class ClearanceObjective(PlacementObjective):
    """
        An objective function to maximize clearance to obstacles on horizontal surfaces.
    """

    def __init__(self, occ_grid, body_grid, b_max=False):
        """
            Create a new ClearanceObjective.
            ---------
            Arguments
            ---------
            occ_grid, OccupancyGrid of target volume
            body_grid, RigidBodyOccupancyGrid - rigid body occupancy grid as volumetric model of the
                target object
            b_max, bool - If True, the objective is to maximize clearance, else minimize
        """
        self._clearance_map = compute_clearance_map(occ_grid)
        self._body_grid = body_grid
        self._cuda_id = self._body_grid.setup_cuda_grid_access(self._clearance_map)
        self._sign = 1.0 if b_max else -1.0
        self._call_count = 0

    def __call__(self, obj_tf):
        """
            Return the objective value for the given solution.
            ---------
            Arguments
            ---------
            obj_tf, np array of shape (4, 4) - object pose to evaluate
            --------
            Returns
            --------
            val, float - objective value (the larger the better)
        """
        if obj_tf is None:
            return float('-inf')
        val = self._body_grid.sum(None, obj_tf, cuda_id=self._cuda_id)
        self._call_count += 1
        return self._sign * val / self._body_grid.get_num_occupied_cells()

    def evaluate(self, obj_tf):
        """
            Return the objective value for the given solution.
            ---------
            Arguments
            ---------
            obj_tf, np array of shape (4, 4) - object pose to evaluate
            --------
            Returns
            --------
            val, float - objective value (the larger the better)
        """
        return self(obj_tf)

    def get_num_evaluate_calls(self, b_reset=True):
        """
            Return statistics on how many times evaluate has been called.
            ---------
            Arguments
            ---------
            b_reset, bool - if True, reset counter
            -------
            Returns
            -------
            num_calls, int - number of times the evaluate function has been called
        """
        num_calls = self._call_count
        if b_reset:
            self._call_count = 0
        return num_calls

    def get_gradient(self, obj_tf, state, to_ref_pose):
        """
            Return the gradient of the objective value w.r.t x, y, theta.
            given that some reference point on the object is currently in the given state (x, y, theta).
            ---------
            Arguments
            ---------
            obj_tf, np array of shape (4, 4) - pose of the object
            state, tuple (x, y, theta) - current state of the reference point
            to_ref_pose, np array of shape (4, 4) - transformation matrix mapping from object frame to reference frame
                for which the state is defined
            -------
            Returns
            -------
            np array of shape (3,) - gradient w.r.t x, y, theta
        """
        cart_grads, loc_positions = self._body_grid.compute_gradients(None, obj_tf, cuda_id=self._cuda_id)
        # translate local positions into positions relative to reference pose
        loc_positions = np.dot(loc_positions, to_ref_pose[:3, :3].T) + to_ref_pose[:3, 3]
        # filter zero gradients
        non_zero_idx = np.unique(np.nonzero(cart_grads[:, :2])[0])
        non_zero_grads, non_zero_pos = cart_grads[non_zero_idx], loc_positions[non_zero_idx]
        if non_zero_grads.shape[0] == 0:
            return np.zeros(3)
        # get object state
        x, y, theta = state
        # compute gradient w.r.t to state
        r = np.array([[-np.sin(theta), np.cos(theta)], [-np.cos(theta), -np.sin(theta)]])
        dxi_dtheta = np.matmul(non_zero_pos[:, :2], r)
        lcart_grads = np.empty((non_zero_grads.shape[0], 3))
        lcart_grads[:, :2] = non_zero_grads[:, :2]
        lcart_grads[:, 2] = np.sum(non_zero_grads[:, :2] * dxi_dtheta, axis=1)
        return self._sign * 1.0 / lcart_grads.shape[0] * np.sum(lcart_grads, axis=0)


if __name__ == "__main__":
    import os
    import IPython
    import mayavi.mlab
    base_path = os.path.dirname(__file__) + '/../../../'
    # sdf_file = base_path + 'data/sdfs/placement_exp_0_low_res'
    # sdf = sdf_core.SDF.load(sdf_file)
    occ_file = base_path + 'data/occupancy_grids/placement_exp_0_low_res'
    occ = grid_module.VoxelGrid.load(occ_file)
    # clearance_map = compute_clearance_map(occ, sdf.get_grid())
    clearance_map = compute_clearance_map(occ)
    # grid = sdf._grid
    # grid = sdf._grid.get_subset(np.array((0, 0, 0)), np.array((1, 1.5, 1)))
    # cell_gen = grid.get_index_generator()
    # positions = grid.get_cell_positions(np.array(list(cell_gen), dtype=int), b_center=False)
    # cell_gen = grid.get_index_generator()
    # values = grid.get_cell_values(np.array(list(cell_gen), dtype=int))
    # x_grad, y_grad, z_grad = np.gradient(grid._cells, grid.get_cell_size())
    # x_extrema = scipy.signal.argrelextrema(grid._cells, np.greater_equal, axis=0)
    # mayavi.mlab.contour3d(grid._cells, contours=[0.0], color=(0.3, 0.3, 0.3))
    # mayavi.mlab.contour3d(clearance_map._cells, contours=[0.0], color=(0.3, 0.3, 0.3))
    mayavi.mlab.volume_slice(clearance_map._cells, slice_index=2, plane_orientation="z_axes")
    # positions = grid.get_cell_positions(np.array(x_extrema))
    # mayavi.mlab.points3d(x_extrema[0], x_extrema[1], x_extrema[2],
    #                      mode="cube", scale_mode='none', line_width=4.0,
    #                      transparent=True, opacity=0.5)

    # mayavi.mlab.contour3d(x_grad, contours=[0.0], transparent=True, opacity=0.5, color=(1.0, 0.0, 0.0))
    # mayavi.mlab.contour3d(y_grad, contours=[0.0], transparent=True, opacity=0.5, color=(0.0, 1.0, 0.0))
    # mayavi.mlab.contour3d(z_grad, contours=[0.0], transparent=True, opacity=0.5, color=(0.0, 0.0, 1.0))

    # mayavi.mlab.points3d(positions[:, 0], positions[:, 1], positions[:, 2],
    #                      values, mode="cube", scale_mode='none', line_width=4.0,
    #                      transparent=True, opacity=0.5, mask_points=100)
    mayavi.mlab.show()
    # IPython.embed()
