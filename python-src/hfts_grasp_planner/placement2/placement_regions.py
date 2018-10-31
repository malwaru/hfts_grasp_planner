import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import scipy.ndimage.morphology as scipy_morph
import skimage.measure as skm


class PlanarRegionExtractor(object):
    """
        This class allows to extract planar placement regions from an occupancy grid.
    """

    def __init__(self):
        """
            Create a new PlanarRegionExtractor.
        """
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

    def extract_planar_regions(self, grid):
        """
            Extract horizontal planar regions from the given grid.
            ---------
            Arguments
            ---------
            grid, Voxel grid storing an occupancy grid. Value type should be bool.
        """
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
        #    data.shape[0], data.shape[1], data.shape[2], grid=grid_shape, block=(8, 8, 8))
        # next cluster them
        label_offset = 0
        for layer in xrange(1, output_grid.shape[2]):
            # TODO save components here
            labels, num_regions = skm.label(output_grid[:, :, layer], return_num=True, connectivity=2)
            labels[labels.nonzero()] += label_offset
            output_grid[:, :, layer] = labels
            label_offset += num_regions
        return output_grid, label_offset
