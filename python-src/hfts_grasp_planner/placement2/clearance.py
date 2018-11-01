import numpy as np
import hfts_grasp_planner.sdf.occupancy as occupancy
import hfts_grasp_planner.sdf.core as sdf_core
import hfts_grasp_planner.placement2.placement_regions as plcmnt_regions


class ClearanceMap(object):
    """
        Stores a distance field that stores for each point how far it is
        from a local maxima of a distance field to obstacles.
        Thus by maximizing the distance stored in this field, 
        one can obtain minimal clearance from obstacles. By minimizing the
        distance stored in this field, on the other hand, one obtains
        locally maximal clearance from obstacles.
    """

    def __init__(self, sdf):
        self._distance_field = None


# def compute_clearance_map(sdf):
    # distance_field = sdf._grid
    # compute local maxima in sdf

def compute_runtime():
    import timeit
    setup_code = """import hfts_grasp_planner.sdf.grid as grid_module; import numpy as np;\
    import hfts_grasp_planner.placement2.placement_regions as plcmnt_regions;\
    path = \"/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/occupancy_grids/placement_exp_0\";\
    world_grid = grid_module.VoxelGrid.load(path);\
    grid = world_grid.get_subset(np.array((0, 0, 0)), np.array((1, 1.5, 1)));"""
    setup_code_gpu = """gpu_kit = plcmnt_regions.GPUExtractPlanarRegions();"""
    evaluate_code_gpu = """surface = gpu_kit.extract_planar_regions(world_grid)"""
    evaluate_code_cpu = """surface = plcmnt_regions.extract_planar_regions(world_grid)"""
    print "Evaluating CPU:"
    print timeit.timeit(evaluate_code_cpu, setup=setup_code, number=1000)
    print "Evaluating GPU:"
    print timeit.timeit(evaluate_code_gpu, setup=setup_code + setup_code_gpu, number=1000)


if __name__ == "__main__":
    import IPython
    import os
    import hfts_grasp_planner.sdf.grid as grid_module
    import openravepy as orpy
    # import hfts_grasp_planner.sdf.visualization as vis_module
    # import mayavi.mlab
    base_path = os.path.dirname(__file__) + '/../../../'
    world_grid = grid_module.VoxelGrid.load(base_path + 'data/occupancy_grids/placement_exp_0_low_res')
    # Or create a new grid
    env = orpy.Environment()
    env.Load(os.path.dirname(__file__) + '/../../../data/environments/placement_exp_0.xml')
    # aabb = np.array([-1.0, -1.0, 0.0, 1.0, 1.0, 1.5])
    # grid_builder = occupancy.OccupancyGridBuilder(env, 0.04)
    # grid = grid_module.VoxelGrid(aabb)
    # grid_builder.compute_grid(grid)
    # grid_builder.clear()
    # grid._cells = grid._cells.astype(bool)
    # vis = vis_module.MatplotLibGridVisualization()
    # vis.visualize_bool_grid(grid)
    # env.SetViewer('qtcoin')
    # vis = vis_module.ORVoxelGridVisualization(env, grid)
    # def binary_color_fn(value):
    #     if value > 0.0:
    #         return np.array([1.0, 0.0, 0.0, 0.5])
    #     return np.array([0.0, 0.0, 0.0, 0.0])
    # vis.update(style=2, color_fn=binary_color_fn)
    # xx, yy, zz = np.where(world_grid.get_raw_data() == 1.0)
    # indices = np.array((xx, yy, zz)).transpose()
    # positions = world_grid.get_cell_positions(indices, b_center=False)
    # mayavi.mlab.points3d(positions[:, 0], positions[:, 1], positions[:, 2],
    #  positions.shape[0] * [world_grid.get_cell_size()],
    #  mode="cube", color=(0, 1, 0), scale_factor=1, line_width=4.0,
    #  transparent=True, opacity=0.5)
    # show subset
    grid = world_grid.get_subset(np.array((0, 0, 0)), np.array((1, 1.5, 1)))
    # xx, yy, zz = np.where(grid.get_raw_data() == 1.0)
    # indices = np.array((xx, yy, zz)).transpose()
    # positions = grid.get_cell_positions(indices, b_center=False)
    # mayavi.mlab.points3d(positions[:, 0], positions[:, 1], positions[:, 2],
    #  positions.shape[0] * [grid.get_cell_size()],
    #  mode="cube", color=(1, 0, 0), scale_factor=1, line_width=4.0,
    #  transparent=True, opacity=0.5)

    gpu_kit = plcmnt_regions.PlanarRegionExtractor()
    labels, num_regions, regions = gpu_kit.extract_planar_regions(grid, max_region_size=0.2)
    # print "found %i regions" % len(regions)
    env.SetViewer('qtcoin')
    handles = []
    xx, yy, zz = np.where(grid.get_raw_data() > 0)
    colors = np.random.random((num_regions+1, 3))
    color = np.empty(4)
    tf = np.array(grid.get_transform())
    # IPython.embed()
    # for cell_id in xrange(len(xx)):
    #     color[:3] = colors[labels[xx[cell_id], yy[cell_id], zz[cell_id]]]
    #     color[:3] = [1.0, 0.0, 0.0]
    #     color[3] = 0.3
    #     tf[:3, 3] = grid.get_cell_position((xx[cell_id], yy[cell_id], zz[cell_id]))
    #     handles.append(env.drawbox(np.array((0, 0, 0)), np.array(3 * [grid.get_cell_size() / 2.0]), color, tf))
    handles.extend(plcmnt_regions.visualize_plcmnt_regions(env, regions, height=grid.get_cell_size()))
    IPython.embed()
    # print regions
    # for r in xrange(1, num_regions + 1):
    #     xx, yy, zz = np.where(surfaces == r)
    #     mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=tuple(np.random.random(3)), scale_factor=1)
    # mayavi.mlab.show()
    # print grid._cells.shape

    # compute_runtime()
