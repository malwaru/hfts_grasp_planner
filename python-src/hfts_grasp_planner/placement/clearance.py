import itertools
import numpy as np
import scipy.ndimage.morphology
import hfts_grasp_planner.sdf.occupancy as occupancy
import hfts_grasp_planner.sdf.grid as grid_module
import hfts_grasp_planner.sdf.core as sdf_core

"""
    This module provides all functionality related to clearance maps.
    A clearance map is a scalar field (VoxelGrid) that maps a point to its clearance from obstacles:
        d_c(p) = d_xy(p) + d_xyz(p),
    where d_xy is the shortest distance to any obstacle within the x,y plane
    and d_xyz the shortest distance to any obstacle.
    A clearance map is created from an occupancy grid of the workspace, and it is assumed
    that the z-axis of the grid is pointing upwards, i.e. is antiparallel to gravity.
"""


def compute_clearance_map(occ_grid, sdf_grid=None):
    """
        Compute a clearance map from the given voxel grid.
        ---------
        Arguments
        ---------
        occ_grid, VoxelGrid - representing occupancy grid of the environment. The cell type must be bool!
        sdf_grid, VoxelGrid (optional) - signed distance field along x,y,z computed from occ_grid
            If not provided, this function computes this field.
        -------
        Returns
        -------
        clearance_map, VoxelGrid - grid representing the clearance map
    """
    if sdf_grid is None:
        sdf_grid = sdf_core.SDFBuilder.compute_sdf(occ_grid)
    # next, compute sdf for each slice
    clearance_map = grid_module.VoxelGrid(np.array(occ_grid.get_workspace()),
                                          num_cells=np.array(occ_grid.get_num_cells()),
                                          cell_size=occ_grid.get_cell_size())
    occ_transposed = occ_grid.get_raw_data().transpose()  # need to transpose for distance_transform
    clrm_data = clearance_map.get_raw_data()
    clrm_data_t = clrm_data.transpose()
    sdf_data_t = sdf_grid.get_raw_data().transpose()
    max_distance = np.min(np.array(occ_transposed[0].shape) * occ_grid.get_cell_size()) / 2.0  # TODO what to set here?
    # run over each layer
    for idx in xrange(occ_transposed.shape[0]):
        if np.any(occ_transposed[idx]):
            inv_occ_layer = np.invert(occ_transposed[idx])
            xy_distance_field = scipy.ndimage.morphology.distance_transform_edt(
                inv_occ_layer, sampling=sdf_grid.get_cell_size())
        else:
            xy_distance_field = np.full(occ_transposed[idx].shape, max_distance)
        clrm_data_t[idx] = xy_distance_field + sdf_data_t[idx]
    clearance_map.set_raw_data(clrm_data)
    return clearance_map


if __name__ == "__main__":
    import os
    import IPython
    import mayavi.mlab
    base_path = os.path.dirname(__file__) + '/../../../'
    sdf_file = base_path + 'data/sdfs/placement_exp_0_low_res'
    sdf = sdf_core.SDF.load(sdf_file)
    occ_file = base_path + 'data/occupancy_grids/placement_exp_0_low_res'
    occ = grid_module.VoxelGrid.load(occ_file)
    clearance_map = compute_clearance_map(occ, sdf.get_grid())
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
