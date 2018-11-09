import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import collections
import numpy as np
import openravepy as orpy
import skimage.measure as skm
import scipy.ndimage.morphology as scipy_morph


class PlanarPlacementRegion(object):
    """
        Struct-like class that stores information about a planar placement region.
    """

    def __init__(self, x_indices, y_indices, tf, cell_size):
        self.base_tf = tf  # transform from local frame to world frame
        # the indices we have might not start at 0, 0, so let's shift them
        xshift, yshift = np.min((x_indices, y_indices), axis=1)
        self.x_indices = x_indices - xshift
        self.y_indices = y_indices - yshift
        # accordingly we need to shift the base frame
        shift_tf = np.eye(4)
        shift_tf[:2, 3] = np.array((xshift, yshift)) * cell_size
        self.base_tf = np.dot(self.base_tf, shift_tf)
        # width and depth in local frame
        self.dimensions = cell_size * (np.array((self.x_indices[-1], self.y_indices[-1])) + 1)
        assert(self.x_indices.shape[0] >= 1)
        assert(self.x_indices.shape[0] == self.y_indices.shape[0])
        self.cell_size = cell_size

    # TODO construct search hiearchy (quadtree) around this -> only need to look at positions in self.labels
    def get_subregions(self):
        """
            Return a list of subregions of this region.
            The returned list may contain 0 to 4 elements. If there are only 0 elements, this region
            consist only of a single voxel/pixel and can thus not be further subdivided.
            -------
            Returns
            -------
            subregions, list of PlanarPlacementRegion objects
        """
        subregions = []
        if self.x_indices.shape[0] == 1 and self.y_indices.shape[0] == 1:
            return subregions
        max_indices = (self.x_indices[-1], self.y_indices[-1])
        split_indices = ((max_indices[0] + 1) / 2, (max_indices[1] + 1) / 2)
        left_filter = self.x_indices < split_indices[0]
        bottom_filter = self.y_indices < split_indices[1]
        left_bottom_filter = np.logical_and(left_filter, bottom_filter)
        left_top_filter = np.logical_and(left_filter, np.logical_not(bottom_filter))
        right_bottom_filter = np.logical_and(np.logical_not(left_filter), bottom_filter)
        right_top_filter = np.logical_and(np.logical_not(left_filter), np.logical_not(bottom_filter))
        filters = [left_bottom_filter, left_top_filter, right_bottom_filter, right_top_filter]
        offsets = np.array([(0, 0), (0, split_indices[1]), (split_indices[0], 0), split_indices])
        for filt, off in zip(filters, offsets):
            if filt.any():
                xx, yy = self.x_indices[filt] - off[0], self.y_indices[filt] - off[1]
                tf = np.eye(4)
                tf[:2, 3] = off * self.cell_size
                tf = np.dot(self.base_tf, tf)
                subregions.append(PlanarPlacementRegion(xx, yy, tf, self.cell_size))

        # TODO remove me:
        children_cells = 0
        for region in subregions:
            children_cells += region.x_indices.shape[0]
        assert(children_cells == self.x_indices.shape[0])
        return subregions
        # def __str__(self):
        #     return "PlanarPlacementRegion: base_tf: " + str(self.base_tf) + ", dimensions: " +\
        #         str(self.dimensions) + ", height: " + str(self.height)

        # def __repr__(self):
        #     return "PlanarPlacementRegion:\n base_tf:\n" + str(self.base_tf) + "\n dimensions: " +\
        #         str(self.dimensions) + "\n height: " + str(self.height)


class PlanarRegionExtractor(object):
    """
        This class allows to extract planar placement regions from an occupancy grid.
    """

    def __init__(self):
        """
            Create a new PlanarRegionExtractor.
        """
        # TODO could/should make this a singleton
        self.mod = SourceModule("""
        # define MEM_IDX(X,Y,Z) (X)*data_y*data_z+(Y)*data_z+(Z)
        # define OCC_MEM_IDX(X,Y,Z) (X+1)*(data_y+2)*(data_z+2)+(Y+1)*(data_z+2)+Z+1

        __global__ void surface_detect(int* occ_grid, int* out, int data_x, int data_y, int data_z)
        {
            int global_x = blockIdx.x * blockDim.x + threadIdx.x;
            int global_y = blockIdx.y * blockDim.y + threadIdx.y;
            int global_z = blockIdx.z * blockDim.z + threadIdx.z;
            // check whether this thread should do anything at all
            if (global_x < data_x && global_y < data_y && global_z < data_z) {
                // bottom layer is always 0
                if (global_z == 0) {
                    out[MEM_IDX(global_x,global_y,global_z)] = 0;
                } else if (occ_grid[OCC_MEM_IDX(global_x,global_y,global_z)] == 0) {  // free cell
                    // look at the support underneath this cell
                    int support = 0;
                    for (int i = -1; i < 2; ++i) {
                        for (int j = -1; j < 2; ++j) {
                            // occ_grid is padded, so we do not need to worry about boundaries
                            support += occ_grid[OCC_MEM_IDX(global_x + i, global_y + j, global_z - 1)];
                        }
                    }
                    out[MEM_IDX(global_x,global_y,global_z)] = support == 9;
                } else{
                    out[MEM_IDX(global_x,global_y,global_z)] = 0;
                }
            }
        }
        """)
        self.surface_detect_fn = self.mod.get_function("surface_detect")

    def extract_planar_regions(self, grid, max_region_size=0.2):
        """
            Extract horizontal planar placement regions from the given grid.
            ---------
            Arguments
            ---------
            grid, Voxel grid storing an occupancy grid. Value type should be bool.
            max_region_size, float - maximum distance that one region is allowed to span along one axis
            -------
            Returns
            -------
            placement_regions, list - a list of PlanarPlacementRegions
        """
        placement_regions = []
        max_cells_region_dim = max(int(max_region_size / grid.get_cell_size()), 1)
        data = grid.get_raw_data(b_include_padding=True).astype(np.int32)
        # fist use Cuda kernel to extract placement surfaces
        inner_shape = np.array(data.shape) - 2  # substract padding
        output_grid = np.empty(inner_shape, dtype=np.int32)
        grid_shape = np.array(inner_shape) / 8 + 1
        v0 = np.int32(inner_shape[0])
        v1 = np.int32(inner_shape[1])
        v2 = np.int32(inner_shape[2])
        self.surface_detect_fn(cuda.In(data), cuda.Out(output_grid), v0, v1, v2,
                               grid=tuple(grid_shape), block=(8, 8, 8))
        # next cluster them
        label_offset = 0
        for layer in xrange(1, output_grid.shape[2]):
            labels, num_regions = skm.label(output_grid[:, :, layer], return_num=True, connectivity=2)
            if num_regions > 0:
                # This is technically not needed, but nice for visualization
                labels[labels.nonzero()] += label_offset
                output_grid[:, :, layer] = labels
                # now compute new plcmnt regions from these clusters
                for r in xrange(label_offset + 1, num_regions + label_offset + 1):
                    placement_regions.extend(PlanarRegionExtractor._compute_plcmnt_regions(
                        grid, output_grid, layer, r, max_cells_region_dim))
                label_offset += num_regions
        return output_grid, label_offset, placement_regions
        # return placement_regions

    @staticmethod
    def _compute_plcmnt_regions(grid, labels, layer, cluster, max_region_extent):
        """
            Compute the placement regions arising from the given surface cluster.
            ---------
            Arguments
            ---------
            grid, VoxelGrid that store occupancy grid (input grid)
            labels, numpy array of same shape as VoxelGrid that stores surface labels
            layer, int - which z-layer of the voxel grid to look at
            cluster, int - which surface cluster(label) to extract from labels
            max_region_extent, int - the maximal extent in voxels along any dimension for a placement region
            -------
            Returns
            -------
            placement regions, list - A list of all placement regions spanning the surface cluster
                identified by cluster
        """
        # first extract voxels belonging to the queried cluster
        xx, yy = np.where(labels[:, :, layer] == cluster)
        min_x, min_y = np.min((xx, yy), axis=1)
        max_x, max_y = np.max((xx, yy), axis=1)
        # compute the extents of the cluster
        total_dimensions = np.array((max_x - min_x, max_y - min_y), dtype=int)
        # compute in how many subregions we need to split it
        num_regions = np.ceil(total_dimensions.astype(float) / max_region_extent).astype(int)
        num_regions = np.max(((1, 1), num_regions), axis=0)
        # and what their dimensions will be (in voxel)
        x_dims = num_regions[0] * [max_region_extent]
        x_dims[-1] = total_dimensions[0] - (num_regions[0] - 1) * max_region_extent
        y_dims = num_regions[1] * [max_region_extent]
        y_dims[-1] = total_dimensions[1] - (num_regions[1] - 1) * max_region_extent
        # get the base position of this cluster, we will use it later
        # base_pos = grid.get_cell_position((min_x, min_y, layer), b_center=False, b_global_frame=False)
        base_idx = np.array((min_x, min_y), dtype=int)
        # TODO should the subregions be constructed such that they have equal size?
        # construct subregions
        regions = []
        for xi in xrange(num_regions[0]):
            for yi in xrange(num_regions[1]):
                # compute base index of subregion
                base_region_idx = np.array((xi, yi)) * max_region_extent + base_idx
                # extract indices of surface pointers that are relevant for this region
                x_filter = np.logical_and(xx >= base_region_idx[0], xx < base_region_idx[0] + x_dims[xi])
                y_filter = np.logical_and(yy >= base_region_idx[1], yy < base_region_idx[1] + y_dims[yi])
                xy_filter = np.logical_and(x_filter, y_filter)
                sxx = xx[xy_filter]
                syy = yy[xy_filter]
                if sxx.shape[0] > 0 and syy.shape[0] > 0:
                    # copute transform of this region relative to grid frame
                    rel_tf = np.eye(4)
                    rel_tf[:3, 3] = grid.get_cell_position(
                        (base_region_idx[0], base_region_idx[1], layer), b_center=False, b_global_frame=False)
                    # compute tf for min cell of this region in world frame
                    region_tf = np.dot(grid.get_transform(), rel_tf)
                    # create plcmnt region
                    new_region = PlanarPlacementRegion(
                        sxx - base_region_idx[0], syy - base_region_idx[1], region_tf, grid.get_cell_size())
                    regions.append(new_region)
        return regions


def visualize_plcmnt_regions(env, regions, alpha=0.3, height=0.01, level=0, b_cells=False):
    """
        Visualize the provided PlanarPlacementRegions in OpenRAVE.
        ---------
        Arguments
        ---------
        env, OpenRAVE environment
        regions, list of PlanarPlacementRegions
        alpha, float - alpha value to render placement regions with
        level, int - if b_cells is False, this parameter determines at which level of the hierarchy
            of each placement region, the bounding box should be drawn
        b_cells, bool - if true visualize actual plcmnt regions cells, else just bounding boxes on the given level
        -------
        Returns
        -------
        handles, list of OpenRAVE handles for the drawings
    """
    color = np.empty(4)
    handles = []
    for region in regions:
        color[:3] = np.random.random(3)
        color[3] = alpha
        if b_cells:
            xx, yy = region.x_indices, region.y_indices
            for c_id in range(len(xx)):
                tf = np.array(region.base_tf)
                rel_tf = np.eye(4)
                rel_tf[:3, 3] = np.array((xx[c_id], yy[c_id], 0)) * region.cell_size
                tf = np.dot(tf, rel_tf)
                handles.append(env.drawbox(np.array((region.cell_size / 2.0, region.cell_size / 2.0, height/2.0)),
                                           np.array([region.cell_size / 2.0, region.cell_size / 2.0, height / 2.0]),
                                           color, tf))
        else:
            subregions_to_draw = []
            if level > 0:
                subregions = collections.deque([region])
                for l in range(level):
                    lp1_subregions = collections.deque()
                    while subregions:
                        nregion = subregions.popleft()
                        children = nregion.get_subregions()
                        if len(children) > 0:
                            if l < level - 1:
                                lp1_subregions.extend(children)
                            else:
                                subregions_to_draw.extend(children)
                        else:
                            subregions_to_draw.append(nregion)
                    subregions = lp1_subregions
            else:
                subregions_to_draw = [region]
            for sregion in subregions_to_draw:
                pos = np.array((sregion.dimensions[0] / 2.0, sregion.dimensions[1] / 2.0, height / 2.0))
                extents = np.array([sregion.dimensions[0] / 2.0, sregion.dimensions[1] / 2.0, height / 2.0])
                handles.append(env.drawbox(pos, extents, color, sregion.base_tf))
    return handles


if __name__ == "__main__":
    import IPython
    import os
    import hfts_grasp_planner.sdf.grid as grid_module
    import openravepy as orpy

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

    gpu_kit = PlanarRegionExtractor()
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
    handles.extend(visualize_plcmnt_regions(env, regions, height=grid.get_cell_size(), level=10))
    # handles.extend(plcmnt_regions.visualize_plcmnt_regions(env, regions, height=grid.get_cell_size(), b_cells=True))
    IPython.embed()
    # print regions
    # for r in xrange(1, num_regions + 1):
    #     xx, yy, zz = np.where(surfaces == r)
    #     mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=tuple(np.random.random(3)), scale_factor=1)
    # mayavi.mlab.show()
    # print grid._cells.shape

    # compute_runtime()
