import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import collections
import numpy as np
import openravepy as orpy
import skimage.measure as skm
import hfts_grasp_planner.sdf.grid as grid_mod
import scipy.ndimage.morphology as scipy_morph
import hfts_grasp_planner.external.transformations as tf_mod
import hfts_grasp_planner.utils as utils


class PlanarPlacementRegion(object):
    """
        Planar placement region described by contiguous region of
        free occupancy grid cells.
    """

    def __init__(self, x_indices, y_indices, tf, cell_size):
        """
            Create a new PlanarPlacementRegion.
            ---------
            Arguments
            ---------
            x_indices, numpy array of int - x positions of cells
            y_indices, numpy array of int - y positions of cells
            tf, numpy array of shape (4, 4) - transformation matrix describing the pose of front left cell
            cell_size, float - dimension of cells
        """
        self.base_tf = tf  # transform from local frame to world frame
        # the indices we have might not start at 0, 0, so let's shift them
        xshift, yshift = np.min((x_indices, y_indices), axis=1)
        self.x_indices = x_indices - xshift
        self.y_indices = y_indices - yshift
        # accordingly we need to shift the base frame
        shift_tf = np.eye(4)
        shift_tf[:2, 3] = np.array((xshift, yshift)) * cell_size
        self.base_tf = np.dot(self.base_tf, shift_tf)
        # dimensions in local frame
        self.dimensions = np.array([0, 0, cell_size])
        self.dimensions[:2] = cell_size * (np.max((self.x_indices, self.y_indices), axis=1) + 1)
        assert(self.x_indices.shape[0] >= 1)
        assert(self.x_indices.shape[0] == self.y_indices.shape[0])
        self.cell_size = cell_size
        # set the reference point for this region to its median
        median_point = np.percentile(np.column_stack((self.x_indices, self.y_indices)),
                                     q=50, axis=0, interpolation='nearest')
        self.contact_tf = np.array(self.base_tf)  # tf that a contact point should have
        self.contact_xy = median_point * cell_size  # xy position of reference contact point in local frame
        # also lift it by half the cell size
        self.contact_tf[:3, 3] += np.dot(self.base_tf[:3, :3],
                                         np.array((self.contact_xy[0], self.contact_xy[1], 0.5 * cell_size)))
        # center of bounding box
        self.center_tf = np.array(self.base_tf)
        self.center_tf[:3, 3] += np.dot(self.base_tf[:3, :3], self.dimensions / 2.0)
        # next, compute radius of encompassing circle with center in contact_tf
        self.radius = np.max((np.linalg.norm(self.contact_xy),  # distance to origin
                              # distance to bounding box's max point
                              np.linalg.norm(self.contact_xy - self.dimensions[:2]),
                              # distance to right bottom point
                              np.linalg.norm(self.contact_xy - (self.dimensions[0], 0.0)),
                              np.linalg.norm(self.contact_xy - (0.0, self.dimensions[1]))))  # distance to left top point
        self.normal = np.dot(self.base_tf[:3, :3], np.array((0, 0, 1.0)))
        self._subregions = None
        self.aabb_distance_field = None
        self.aabb_dist_gradient_field = None
        self._compute_aabb_distance_field()

    def _compute_aabb_distance_field(self):
        workspace = np.array([-self.cell_size, -self.cell_size, 0.0, self.dimensions[0] + self.cell_size,
                              self.dimensions[1] + self.cell_size, self.dimensions[2]])
        shape = np.array([0, 0, 1])
        shape[:2] = (self.dimensions[:2] / self.cell_size).astype(int) + 2  # add some buffer to the sides
        self.aabb_distance_field = grid_mod.VoxelGrid(workspace, cell_size=self.cell_size,
                                                      num_cells=shape,
                                                      base_transform=self.base_tf,
                                                      dtype=bool)
        raw_data = self.aabb_distance_field.get_raw_data()
        raw_data[:, :] = True
        raw_data[self.x_indices + 1, self.y_indices + 1] = False
        raw_data = scipy_morph.distance_transform_edt(raw_data, sampling=self.cell_size) - self.cell_size
        self.aabb_distance_field.set_raw_data(raw_data)
        self.aabb_dist_gradient_field = grid_mod.VectorGrid(workspace, cell_size=self.cell_size,
                                                            num_cells=shape,
                                                            base_transform=self.base_tf,
                                                            b_in_slices=True)
        grad_x, grad_y = np.gradient(raw_data.reshape(shape[:2]), self.cell_size)
        self.aabb_dist_gradient_field.vectors[0] = grad_x.reshape(shape)
        self.aabb_dist_gradient_field.vectors[1] = grad_y.reshape(shape)

    def clear_subregions(self):
        """
            Reset any prior computation done to create subregions. After
            calling this function, this region will behave exactly the same way as if it had just been
            created.
        """
        self._subregions = None

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
        if self._subregions is not None:
            return self._subregions
        self._subregions = []
        if self.x_indices.shape[0] == 1 and self.y_indices.shape[0] == 1:
            return self._subregions
        max_indices = np.max((self.x_indices, self.y_indices), axis=1)
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
                self._subregions.append(PlanarPlacementRegion(xx, yy, tf, self.cell_size))
        # sort subregions by area
        self._subregions.sort(key=lambda x: -x.get_area())
        return self._subregions

    def get_num_subregions(self):
        if self._subregions is None:
            self.get_subregions()  # initializes subregions
        return len(self._subregions)

    def get_leaf_key(self, position):
        """
            Return the key of the smallest subregion that contains the given position
            ---------
            Arguments
            ---------
            position, np.array of shape (3,) - (x, y, z)
            -------
            Returns
            -------
            key, tuple - key of the subregion, None if not within this region, () if this region is a leaf
        """
        # transform position into local frame
        inv_base_tf = utils.inverse_transform(self.base_tf)
        lpos = np.dot(inv_base_tf[:3, :3], position) + inv_base_tf[:3, 3]
        if not self.has_subregions():  # is this a leaf?
            if (lpos <= self.cell_size).all() and (lpos >= 0).all():  # position is in this leaf
                return ()
            else:
                return None
        if (lpos < 0).any() or (lpos > self.dimensions).any():  # is position out of bounding box?
            return None

        # since we sort children by area, we can't do much other than asking each child region whether
        # it contains the position
        childregions = self.get_subregions()
        for i, child in enumerate(childregions):
            key = child.get_leaf_key(position)
            if key is not None:
                return (i,) + key
        return None

    def get_subregion(self, key):
        """
            Return the subregion identified by key. The key may either be a single int and then
            the returned subregion is the key'th subregion, or the key is a tuple of int which defines
            the subregion recursively, i.e. (i, j) describes the jth subregion of the ith
            subregion of this region.
            If this region has no subregions, return self.
        """
        regions = self.get_subregions()
        if len(regions) == 0:
            return self
        if type(key) == int:
            return regions[key]
        assert(type(key) == tuple)
        if len(key) == 0:
            return self
        return regions[key[0]].get_subregion(key[1:])

    def has_subregions(self):
        """
            Return whether this region has subregions.
        """
        return self.get_num_subregions() > 0

    def get_area(self):
        """
            Return the approximate contact area covered by this region.
        """
        return self.x_indices.shape[0] * self.cell_size**2


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
            grid, VoxelGrid storing an occupancy grid. Value type should be bool.
            max_region_size, float - maximum distance that one region is allowed to span along one axis
            -------
            Returns
            -------
            contact_cells, VoxelGrid with bool values - a grid of same dimensions as the input
                grid that stores for each cell whether it is a valid position for a placement contact.
            labels, numpy array of the same shape as grid.get_num_cells(), where each entry stores to which
                placement region a cell belongs
            num_regions, int - the number of contiguous placement regions
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
        surface_grid = grid_mod.VoxelGrid(grid.get_workspace(), cell_size=grid.get_cell_size(),
                                          num_cells=np.array(grid.get_num_cells()),
                                          base_transform=grid.get_transform(),
                                          dtype=bool)
        surface_grid.set_raw_data(output_grid.astype(bool))
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
        # sort placement regions by area
        placement_regions.sort(key=lambda x: -x.get_area())
        return surface_grid, output_grid, label_offset, placement_regions
        # return placement_regions

    @staticmethod
    def compute_surface_distance_field(surface_grid, padding=0.0):
        """
            Compute a grid that stores in each cell the x,y distance to the contact surface
            described in surface_grid. In addition, also compute the gradient fields.
            ---------
            Arguments
            ---------
            surface_grid, VoxelGrid with bool values - a grid storing where valid positions of placement
                contacts are
            padding (optional), float - additional padding by which the returned grid should be larger than the
                input surface grid.
            -------
            Returns
            -------
            distance_surface_grid, VoxelGrid with float values - a grid storing x,y distance to closest valid position
                of placement contact within that plane.
            distance_gradient_grid, VectorGrid - sliced vector grid storing 2D gradients of distance_surface_grid
        """
        if padding > 0.0:
            # first enlarge surface grid
            surface_grid = surface_grid.enlarge(padding, False)
        cell_size = surface_grid.get_cell_size()
        distance_map = grid_mod.VoxelGrid(np.array(surface_grid.get_workspace()),
                                          cell_size=cell_size,
                                          num_cells=np.array(surface_grid.get_num_cells()),
                                          b_in_slices=True, b_use_cuda=True)
        gradient_grid = grid_mod.VectorGrid(np.array(surface_grid.get_workspace()),
                                            cell_size=cell_size,
                                            num_cells=np.array(surface_grid.get_num_cells()),
                                            b_in_slices=True)
        distance_data = distance_map.get_raw_data()
        distance_data_t = distance_data.transpose()  # transposed view on distance_data
        occ_transposed = surface_grid.get_raw_data().transpose().astype(bool)
        vectors_t = gradient_grid.vectors.transpose()
        # run over each layer
        for idx in xrange(occ_transposed.shape[0]):
            if np.any(occ_transposed[idx]):
                inv_occ_layer = np.invert(occ_transposed[idx])
                xy_distance_field = scipy_morph.distance_transform_edt(
                    inv_occ_layer, sampling=cell_size)
                grad_y, grad_x = np.gradient(xy_distance_field, cell_size)
            else:
                xy_distance_field = np.full(occ_transposed[idx].shape, float("inf"))
                grad_x = np.full(occ_transposed[idx].shape, 0.0)
                grad_y = np.full(occ_transposed[idx].shape, 0.0)
            distance_data_t[idx] = xy_distance_field
            vectors_t[idx, :, :, 0] = grad_x
            vectors_t[idx, :, :, 1] = grad_y
        distance_map.set_raw_data(distance_data)
        return distance_map, gradient_grid

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
        grid_tf = grid.get_transform()
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
                    region_tf = np.dot(grid_tf, rel_tf)
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
    # import openravepy as orpy

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

    def show_contact_grid(surface, show_type="distances"):
        surface_distance, gradients = PlanarRegionExtractor.compute_surface_distance_field(surface, 0.3)
        if show_type == "distances":
            # NOTE: surface_distance._cells should contain no ("inf") values, else you don't see anything
            surface_distance._cells[surface_distance._cells == float("inf")] = 0.0
            mayavi.mlab.volume_slice(surface_distance._cells, slice_index=2, plane_orientation="z_axes")
        else:
            grad_z = np.zeros((gradients.vectors[0].shape))
            src = mayavi.mlab.pipeline.vector_field(
                gradients.vectors[0], gradients.vectors[1], grad_z)
            mayavi.mlab.pipeline.vector_cut_plane(src, mask_points=2, plane_orientation="z_axes")
        mayavi.mlab.show()

    def show_local_sdf(region, show_type="distances"):
        if show_type == "distances":
            distance_field = region.aabb_distance_field.get_raw_data()
            mayavi.mlab.volume_slice(distance_field, slice_index=2, plane_orientation="z_axes")
        else:
            import matplotlib.pyplot as plt
            grad_x, grad_y = region.aabb_dist_gradient_field.vectors
            plt.quiver(grad_x.reshape(grad_x.shape[:2]).transpose(), grad_y.reshape(grad_y.shape[:2]).transpose())
            plt.show()
            # grad_z = np.zeros_like(grad_x)
            # src = mayavi.mlab.pipeline.vector_field(grad_x, grad_y, grad_z)
            # mayavi.mlab.pipeline.vector_cut_plane(src, plane_orientation="z_axes")
        mayavi.mlab.show()

    # import hfts_grasp_planner.sdf.visualization as vis_module
    import mayavi.mlab
    base_path = os.path.dirname(__file__) + '/../../../../'
    world_grid = grid_module.VoxelGrid.load(base_path + 'data/occupancy_grids/placement_exp_0_low_res')
    # Or create a new grid
    # env = orpy.Environment()
    # env.Load(base_path + 'data/environments/placement_exp_0.xml')
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
    surface, labels, num_regions, regions = gpu_kit.extract_planar_regions(grid, max_region_size=0.2)
    # show_contact_grid(surface)
    # show_contact_grid(surface, "gradients")
    # for rid, region in enumerate(regions):
    # print "Showing region ", rid
    # show_local_sdf(region, "gradients")
    show_local_sdf(regions[0], "gradients")
    # print "found %i regions" % len(regions)
    # env.SetViewer('qtcoin')
    # handles = []
    # surface_grid = surface.get_raw_data()
    # xx, yy, zz = np.where(surface_grid)
    # colors = np.random.random((num_regions+1, 3))
    # color = np.empty(4)
    # tf = np.array(grid.get_transform())
    # IPython.embed()
    # for cell_id in xrange(len(xx)):
    #     color[:3] = colors[labels[xx[cell_id], yy[cell_id], zz[cell_id]]]
    #     color[:3] = [1.0, 0.0, 0.0]
    #     color[3] = 0.3
    #     tf[:3, 3] = grid.get_cell_position((xx[cell_id], yy[cell_id], zz[cell_id]))
    #     handles.append(env.drawbox(np.array((0, 0, 0)), np.array(3 * [grid.get_cell_size() / 2.0]), color, tf))
    # handles.extend(visualize_plcmnt_regions(env, regions, height=grid.get_cell_size(), level=10))
    # handles.extend(plcmnt_regions.visualize_plcmnt_regions(env, regions, height=grid.get_cell_size(), b_cells=True))
    # IPython.embed()
    # print regions
    # for r in xrange(1, num_regions + 1):
    #     xx, yy, zz = np.where(surfaces == r)
    #     mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=tuple(np.random.random(3)), scale_factor=1)
    # mayavi.mlab.show()
    # print grid._cells.shape
    # mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=tuple(np.random.random(3)), scale_factor=1)

    # compute_runtime()
