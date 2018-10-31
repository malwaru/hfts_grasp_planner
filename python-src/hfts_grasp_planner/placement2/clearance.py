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
    # import hfts_grasp_planner.sdf.visualization as vis_module
    import mayavi.mlab
    base_path = os.path.dirname(__file__) + '/../../../'
    world_grid = grid_module.VoxelGrid.load(base_path + 'data/occupancy_grids/placement_exp_0')
    # Or create a new grid
    # env = orpy.Environment()
    # env.Load(os.path.dirname(__file__) + '/../../../data/environments/placement_exp_0.xml')
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
    grid = world_grid.get_subset(np.array((0, 0, 0)), np.array((1, 1.5, 1)))
    xx, yy, zz = np.where(grid.get_raw_data() == 1.0)
    mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=(0, 1, 0), scale_factor=1)
    gpu_kit = plcmnt_regions.PlanarRegionExtractor()
    surfaces, regions = gpu_kit.extract_planar_regions(grid)
    print "found %i regions" % regions
    for r in xrange(1, regions + 1):
        xx, yy, zz = np.where(surfaces == r)
        mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=tuple(np.random.random(3)), scale_factor=1)
    mayavi.mlab.show()
    # print grid._cells.shape

    # compute_runtime()
