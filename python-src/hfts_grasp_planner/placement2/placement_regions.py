import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import openravepy as orpy
import skimage.measure as skm
import scipy.ndimage.morphology as scipy_morph


class PlanarPlacementRegion(object):
    """
        Struct-like class that stores information about a planar placement region.
    """

    def __init__(self, labels, tf, cell_size):
        self.base_tf = tf  # transform from local frame to world frame
        self.dimensions = cell_size * np.array(labels.shape)  # width and depth in local frame
        self.labels = labels
        self.cell_size = cell_size

    # TODO construct search hiearchy (quadtree) around this -> only need to look at positions in self.labels

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
        # construct subregions
        regions = []
        for xi in xrange(num_regions[0]):
            for yi in xrange(num_regions[1]):
                # compute base index of subregion
                base_region_idx = np.array((xi, yi)) * max_region_extent + base_idx
                # x_range = xx[np.where(np.logical_and(xx >= base_idx[0], xx < base_idx[0] + max_region_extent))]
                # y_range = yy[np.where(np.logical_and(yy >= base_idx[1], yy < base_idx[1] + max_region_extent))]
                # extract sub image that is relevant for this region
                sublabels = np.array(labels[base_region_idx[0]: base_region_idx[0] + x_dims[xi],
                                            base_region_idx[1]: base_region_idx[1] + y_dims[yi],
                                            layer])
                # make it binary and remove irrelevant cluster labels
                sublabels = sublabels == cluster
                if np.sum(sublabels) > 0:  # if this region contains any points
                    # copute transform of this region relative to grid frame
                    rel_tf = np.eye(4)
                    rel_tf[:3, 3] = grid.get_cell_position(
                        (base_region_idx[0], base_region_idx[1], layer), b_center=False, b_global_frame=False)
                    # compute tf for min cell of this region in world frame
                    region_tf = np.dot(grid.get_transform(), rel_tf)
                    # create plcmnt region
                    new_region = PlanarPlacementRegion(sublabels, region_tf, grid.get_cell_size())
                    regions.append(new_region)
        return regions


def visualize_plcmnt_regions(env, regions, alpha=0.3, height=0.01, b_cells=False):
    """
        Visualize the provided PlanarPlacementRegions in OpenRAVE.
        ---------
        Arguments
        ---------
        env, OpenRAVE environment
        regions, list of PlanarPlacementRegions
        alpha, float - alpha value to render placement regions with
        b_cells, bool - if true visualize actual plcmnt regions cells, else just bounding boxes
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
            xx, yy = np.where(region.labels)
            for c_id in range(len(xx)):
                tf = np.array(region.base_tf)
                rel_tf = np.eye(4)
                rel_tf[:3, 3] = np.array((xx[c_id], yy[c_id], 0)) * region.cell_size
                tf = np.dot(tf, rel_tf)
                handles.append(env.drawbox(np.array((region.cell_size / 2.0, region.cell_size / 2.0, height/2.0)),
                                           np.array([region.cell_size / 2.0, region.cell_size / 2.0, height / 2.0]),
                                           color, tf))
        else:
            handles.append(env.drawbox(np.array((region.dimensions[0] / 2.0, region.dimensions[1] / 2.0, height/2.0)),
                                       np.array([region.dimensions[0] / 2.0, region.dimensions[1] / 2.0, height / 2.0]),
                                       color, region.base_tf))
    return handles
