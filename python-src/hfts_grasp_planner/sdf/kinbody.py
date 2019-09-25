"""
    This module provides functionalities to compute the degree of collisions
    between a kinbody and its environment. The algorithms within this module
    build on the availability of a Signed Distance Field of the environment.
"""
import itertools
import collections
# import rospy
import cPickle
import numpy as np
import openravepy as orpy
import hfts_grasp_planner.sdf.core as sdf_core
import hfts_grasp_planner.sdf.occupancy as occupancy_mod
import hfts_grasp_planner.sdf.grid as grid_mod
import hfts_grasp_planner.utils as utils


def construct_grid(links, cell_size):
    """
        Construct an occupancy grid of the given links with given cell_size.
        ---------
        Arguments
        ---------
        list of links, OpenRAVE Link - links to contruct grid for (all links are assumed to be within the same env)
        cell_size, float - size of a cell
        -------
        Returns
        -------
        list of grid, VoxelGrid - occupancy grid in the links' local frames
    """
    body = links[0].GetParent()
    env = body.GetEnv()
    collision_map_builder = occupancy_mod.OccupancyGridBuilder(env, cell_size)
    grids = []
    for link in links:
        with env:
            with orpy.KinBodyStateSaver(link.GetParent()):
                link.SetTransform(np.eye(4))  # set link to origin frame
                body_flags = []
                for body in env.GetBodies():
                    body_flags.append((body.IsEnabled(), body.IsVisible()))
                    body.Enable(False)
                    body.SetVisible(False)
                link.Enable(True)
                # construct a binary occupancy grid of the body
                local_aabb = link.ComputeAABB()
                grid = grid_mod.VoxelGrid((local_aabb.pos(), local_aabb.extents() * 2.0), cell_size, dtype=bool)
                collision_map_builder.compute_grid(grid)
                # restore flags # TODO do we need this? Doesn't with env solve this?
                for body, body_flag in itertools.izip(env.GetBodies(), body_flags):
                    body.Enable(body_flag[0])
                    body.SetVisible(body_flag[1])
                grids.append(grid)
    collision_map_builder.clear()
    return grids


class OccupancyOctreeCell(object):
    """
        Represents a cell of an OccupancyOctree
    """

    def __init__(self, idx_box, depth, grid):
        """
            Create a new cell.
            @param idx_box - numpy array describing the grid index range
                [min_x, min_y, min_z, max_x, max_y, max_z] (max_x, max_y, max_z) are exclusive.
            @param depth - depth of this node in the hierarchy
            @param grid - the grid this cell is defined for
        """
        self.idx_box = idx_box
        self.occupied = True
        self.children = []
        self.depth = depth
        self.num_occupied_leaves = 0
        self.cart_dimensions = (idx_box[3:] - idx_box[:3]) * grid.get_cell_size()
        self.radius = np.linalg.norm(self.cart_dimensions / 2.0)
        min_cell_pos = grid.get_cell_position(self.idx_box[:3], b_center=False)
        self.cart_center = min_cell_pos + 0.5 * self.cart_dimensions

    def is_leaf(self):
        return len(self.children) == 0

    def get_corners(self, tf):
        """
            Return the coordinates of the 8 corners of this cell.
            ---------
            Arguments
            ---------
            tf, numpy array of shape (4, 4) - transformation matrix to world frame
        """
        local_corners = self.cart_center + self.cart_dimensions / 2.0 * np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                                                                                  [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
        return np.dot(local_corners, tf[:3, :3].transpose()) + tf[:3, 3]


class OccupancyOctree(object):
    """
        This class represents an occupancy map for a rigid body
        using an octree representation. Each cell of the octree stores whether
        it is part of the volume of the rigid body or not. Thus the octree provides
        a hierarchical representation of the body's volume.
        This class allows to efficiently compute to what degree a rigid body collides with its
        environment, if a signed distance field for this environment exists.
    """

    def __init__(self, cell_size, link, grid=None):
        """
            Construct a new OccupancyOctree. If you intend to construct multiple occupancy octrees for different links,
            use the factory method create_occupancy_octrees(...) to save resources.
            ---------
            Arguments
            ---------
            cell_size, float - minimum edge length of a cell (all cells are cubes)
            link, Link to create the tree for
            grid(optional), OccupancyGrid - to save resources, you may pass a precomputed occupancy grid
                that needs to have the same cell_size
        """
        self._link = link
        self._body = link.GetParent()
        self._grid = None
        self._root = None
        self._total_volume = 0.0
        self._depth = 0
        self._cell_size = None
        if grid is None:
            grid = construct_grid([link], cell_size)[0]
        self._construct_octree(grid)
        self._num_cells = grid.get_num_cells()
        self._cell_size = grid.get_cell_size()

    def __getstate__(self):
        # Copy the object's state from self.__dict__, but remove OpenRAVE objects
        state = self.__dict__.copy()
        # remove OpenRAVE objects for pickling
        del state['_link']
        del state['_body']
        return state

    def __setstate__(self, state):
        # Restore instance from state
        self.__dict__.update(state)
        # OpenRAVE objects have to be set by load function

    def _construct_octree(self, grid):
        """
            Construct the octree. Assumes self._grid has been created.
        """
        # first construct the root
        root_aabb_idx = np.zeros((6,), dtype=np.int)  # stores indices of cells in grid covered by this node
        root_aabb_idx[3:] = grid.get_num_cells()  # it represents all cells
        self._root = OccupancyOctreeCell(root_aabb_idx, 0, grid)
        # construct tree
        nodes_to_refine = collections.deque([self._root])
        while nodes_to_refine:  # as long as there are nodes to refine
            current_node = nodes_to_refine.popleft()
            # handle = self.draw_cell(current_node)
            min_idx = current_node.idx_box[:3]
            max_idx = current_node.idx_box[3:]
            idx_range = max_idx - min_idx  # the number of cells along each axis in this box
            volume = np.multiply.reduce(idx_range)
            min_value = grid.get_min_value(min_idx, max_idx)
            max_value = grid.get_max_value(min_idx, max_idx)
            assert(volume >= 1)
            assert(min_value == 0.0 or min_value == 1.0)
            assert(max_value == 0.0 or max_value == 1.0)
            if volume == 1 or max_value == 0.0:
                # we are either at the bottom of the hierarchy or all children are free
                current_node.occupied = min_value == 1
                current_node.num_occupied_leaves = volume if current_node.occupied else 0
            else:
                # we want to refine nodes that are mixed, or in collision and
                # do not have minimal size
                assert((idx_range > 0).all())
                # compute sizes of children
                half_sizes = np.zeros((2, 3), dtype=int)
                half_sizes[0] = idx_range / 2  # we split this box into 8 children by dividing along each axis
                half_sizes[1] = idx_range - half_sizes[0]  # half_sizes stores the divisions for each axis
                # now we create the actual ranges for each of the 8 children
                children_dimensions = itertools.product(half_sizes[:, 0], half_sizes[:, 1], half_sizes[:, 2])
                child_combinations = itertools.product(range(2), repeat=3)
                for (child_id, child_dim) in itertools.izip(child_combinations, children_dimensions):
                    child_volume = np.multiply.reduce(child_dim)
                    if child_volume != 0:
                        child_idx_box = np.zeros(6, dtype=int)
                        child_idx_box[:3] = min_idx + child_id * half_sizes[0]
                        child_idx_box[3:] = child_idx_box[:3] + child_dim
                        child_node = OccupancyOctreeCell(child_idx_box, current_node.depth + 1, grid)
                        current_node.children.append(child_node)
                        nodes_to_refine.append(child_node)
                        self._depth = max(self._depth, child_node.depth)

        def update_bottom_up(node):
            # helper function to recursively compute the number of occupied leaves and penetration distances
            if not node.children:
                return node.num_occupied_leaves
            if not node.occupied:
                assert(node.num_occupied_leaves == 0)
                return 0, 0.0
            for child in node.children:
                node.num_occupied_leaves += update_bottom_up(child)
            return node.num_occupied_leaves
        # lastly, update num_occupied_leaves flags
        update_bottom_up(self._root)
        self._total_volume = self._root.num_occupied_leaves * np.power(grid.get_cell_size(), 3)

    @staticmethod
    def create_occupancy_octrees(cell_size, links):
        """
            Factory method to construct a sequence of occupancy octrees.
            ---------
            Arguments
            ---------
            cell_size, float - minimal cell size to use for all trees
            links, list of OpenRAVE links - links to create cells for.
        """
        grids = construct_grid(links, cell_size)
        return [OccupancyOctree(cell_size, link, grid) for link, grid in zip(links, grids)]

    @staticmethod
    def load(filename, link):
        """
            Load an OccupancyOctree from file. There is no sanity check performed whether the loaded
            OccupancyOctree really matches the given link.
            ---------
            Arguments
            ---------
            filename, string - filename
            link, OpenRAVE link - link this OccupancyOctree is supposed to represent
            -------
            Returns
            -------
            OccupancyOctree
        """
        with open(filename, 'r') as pickle_file:
            obj = cPickle.load(pickle_file)
        if type(obj) != OccupancyOctree:
            raise IOError("Could not load OccupancyOctree from %s" % filename)
        obj._link = link
        obj._body = link.GetParent()
        return obj

    def save(self, filename):
        """
            Save this object to the given file.
            ---------
            Arguments
            ---------
            filename, string - filename
        """
        with open(filename, 'w') as pickle_file:
            cPickle.dump(self, pickle_file)

    def draw_cell(self, cell):
        """
            Draw the specified cell.
            @param cell - OccupancyOctreeCell to render
            @return handle - OpenRAVE handle
        """
        tf = self._link.GetTransform()
        env = self._body.GetEnv()
        color = np.array([cell.max_int_distance / self.get_maximal_peneration_depth(), 0.0, 0.0, 1.0])
        return env.drawbox(cell.cart_center, cell.cart_dimensions / 2.0, color, tf)

    def get_depth(self):
        """
            Return the maximal depth of the hierarchy.
        """
        return self._depth

    def get_volume(self):
        """
            Return the total volume of occupied cells.
        """
        return self._total_volume

    def get_num_occupied_cells(self):
        """
            Get the total number of cells that are occupied.
        """
        return self._root.num_occupied_leaves

    def get_maximal_peneration_depth(self):
        """
            Return the maximal depth any obstacle can penetrate this body.
        """
        return self._max_possible_penetration

    def get_root_cell(self):
        """
            Return root cell of occupancy octree.
        """
        return self._root

    def get_cell_size(self):
        """
            Return the cell size.
        """
        return self._cell_size

    def visualize(self, level):
        """
            Visualize the octree for the given level.
            If the level is larger than the maximum depth, the octree is visualized for the maximum depth.
            @param level - level to draw
            @return handles - list of handles for the drawings
        """
        handles = []
        level = min(level, self._depth)
        cells_to_render = collections.deque([self._root])
        position = np.empty((4,))
        position[3] = 1
        while cells_to_render:
            cell = cells_to_render.popleft()
            if cell.depth == level or cell.is_leaf():
                if cell.occupied:
                    # render this box
                    handles.append(self.draw_cell(cell))
            else:
                cells_to_render.extend(cell.children)
        return handles

    def compute_intersection(self, scene_sdf, tf=None, bvolume_in_cells=False):
        """
            Computes the intersection between an octree and the geometry
            of the scene described by the provided scene sdf.
            ---------
            Arguments
            ---------
            scene_sdf - signed distance field of the environment that the body
                this map belongs to resides in
            tf (optional), np.array - pose of the object. If not provided, current pose is used
            bvolume_in_cells (optional), bool - if True, return the intersecting volume as number of cells,
                else the actual volume of these cells
            -------
            Returns
            -------
            (v, rv, dc, adc, mid, med) -
                v is the total volume that is intersecting
                rv is this volume relative to the body's total volume, i.e. in range [0, 1]
                dc is a cost that is computed by (approximately) summing up all signed
                    distances of intersecting cells
                adc is this cost divided by the number of intersecting cells, i.e. the average
                    signed distance of the intersecting cells
                max_ext_distance, float - maximal exterior distance of a colliding cell
        """
        if not self._root.occupied:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if tf is None:
            tf = self._link.GetTransform()
        num_intersecting_leaves = 0
        distance_cost = 0.0
        layer_idx = 0
        max_ext_distance = 0.0  # maximal exterior penetration distance
        # current_layer = [self._root]  # invariant current_layer items are occupied
        current_layer = collections.deque([self._root])  # invariant current_layer items are occupied
        next_layer = collections.deque()
        # iterate through hierarchy layer by layer (bfs) - this way we can perform more efficient batch distance queries
        while current_layer:
            # first get the positions of all cells on the current layer
            query_positions = np.array([cell.cart_center for cell in current_layer])
            query_positions = np.dot(query_positions, tf.transpose()[:3, :3]) + tf[:3, 3]
            # query distances for all cells on this layer
            distances = scene_sdf.get_distances(query_positions)
            for dist, cell in itertools.izip(distances, current_layer):
                # handle = self.draw_cell(cell)
                if dist > cell.radius:  # none of the points in this cell can be in collision
                    continue
                elif dist < -1.0 * cell.radius:
                    # the cell lies so far inside of an obstacle, that it is completely in collision
                    num_intersecting_leaves += cell.num_occupied_leaves
                    # regarding the distance cost, assume the worst case, i.e. add the maximum distance
                    # that any child might have for all children
                    ext_distance = (dist - cell.radius)
                    distance_cost += cell.num_occupied_leaves * ext_distance
                    max_ext_distance = max(max_ext_distance, abs(ext_distance))
                else:  # boundary, partly in collision
                    if cell.children:  # as long as there are children, we can descend
                        next_layer.extend([child for child in cell.children if child.occupied])
                    else:
                        num_intersecting_leaves += cell.num_occupied_leaves
                        # subtracting the radius here guarantees that distance_cost is always negative
                        ext_distance = dist - cell.radius
                        distance_cost += ext_distance
                        max_ext_distance = max(max_ext_distance, abs(ext_distance))
            # switch to next layer
            current_layer, next_layer = next_layer, current_layer
            next_layer.clear()
            layer_idx += 1
        if bvolume_in_cells:
            intersection_volume = num_intersecting_leaves
        else:
            intersection_volume = num_intersecting_leaves * np.power(self._cell_size, 3)
        relative_volume = num_intersecting_leaves / float(self._root.num_occupied_leaves)
        normalized_distance_cost = distance_cost / float(self._root.num_occupied_leaves)
        return intersection_volume, relative_volume, distance_cost, normalized_distance_cost, max_ext_distance

    def compute_max_penetration(self, scene_sdf, b_compute_dir=False):
        """
            Computes the maximum penetration of this link with the given sdf.
            The maximum penetration is the minimal signed distance in this link's volume
            in the given scene_sdf.
            ---------
            Arguments
            ---------
            scene_sdf, SceneSDF
            b_compute_dir, bool - If True, also retrieve the direction from the maximally penetrating
                cell to the closest free cell.
            -------
            Returns
            -------
            penetration distance, float - minimum in scene_sdf in the volume covered by this link.
            v_to_border, numpy array of shape (3,) - translation vector to move the cell with maximum penetration
                out of collision (None if b_compute_dir is False)
            pos, numpy array of shape (3,) - position of the cell with maximum penetration (in world frame)
            DISABLED: dist_to_surface, float - if b_compute_dir is True, the distance from the point of maximum
                        penetration towards the body's surface along -v_to_border
        """
        max_penetration = 0.0
        # dist_to_surface = 0.0  # distance to object surface along the direction of penetration
        direction = np.array([0.0, 0.0, 0.0]) if b_compute_dir else None
        max_pen_position = np.array([0.0, 0.0, 0.0])
        if not self._octree._root.occupied:
            return max_penetration, direction
        tf = self._octree._link.GetTransform()
        current_layer = collections.deque([self._octree._root])  # invariant current_layer items are occupied
        next_layer = collections.deque()
        # iterate through hierarchy layer by layer (bfs) - this way we can perform more efficient batch distance queries
        while current_layer:
            # first get the positions of all cells on the current layer
            query_positions = np.array([cell.cart_center for cell in current_layer])
            query_positions = np.dot(query_positions, tf.transpose()[:3, :3]) + tf[:3, 3]
            # query distances for all cells on this layer
            distances = scene_sdf.get_distances(query_positions)
            for dist, cell, pos in itertools.izip(distances, current_layer, query_positions):
                # handle = self.draw_cell(cell)
                if dist > cell.radius:  # none of the points in this cell can be in collision
                    continue
                else:
                    if cell.children:  # as long as there are children, we can descend
                        if max_penetration > dist - cell.radius:  # a child of this cell might have a larger penetration
                            next_layer.extend([child for child in cell.children if child.occupied])
                    else:
                        if max_penetration > dist - cell.radius:
                            max_penetration = dist - cell.radius
                            max_pen_position = pos[:3]
                            if b_compute_dir:  # retrieve direction
                                direction = np.array(scene_sdf.get_direction(pos[:3]))
            current_layer, next_layer = next_layer, current_layer
            next_layer.clear()
        return max_penetration, direction, max_pen_position  # , dist_to_surface


class RigidBodyOccupancyGrid(object):
    """
        This class represents a rigid body, i.e. a link, as an occupancy grid.
        This is, in principle, similar to the OccupancyOctree defined above, but here there is
        no octree. The class provides a numpy-based function to sum all cells in a VoxelGrid
        that overlap with the link given its current transform.
    """

    def __init__(self, cell_size, link, grid=None):
        """
            Create a new RigidBodyOccupancy Grid.
            ---------
            Arguments
            ---------
            cell_size, float - cell size of the grid
            link, OpenRAVE Link - link to represent
            grid (optional), VoxelGrid - You may provide the voxel grid yourself, e.g.
                if you want to load it from a file. There is no sanity check performed though,
                whether this grid really represents the link. In case you provide a grid,
                the argument cell_size is ignored.
        """
        self._link = link
        self._body = link.GetParent()
        if grid is None:
            self._grid = construct_grid([link], cell_size)[0]
        else:
            self._grid = grid
            cell_size = grid.get_cell_size()
        cells = self._grid.get_raw_data()
        xx, yy, zz = np.nonzero(cells)
        indices_mat = np.column_stack((xx, yy, zz))
        self._num_occupied_cells = xx.shape[0]
        self._cell_volume = pow(cell_size, 3)
        self._total_volume = self._num_occupied_cells * self._cell_volume
        self._locc_positions = self._grid.get_cell_positions(indices_mat, b_center=True, b_global_frame=False)
        self._cuda_sdf_interpolator = None
        self._cuda_interpolators = []

    def setup_cuda_sdf_access(self, scene_sdf):
        """
            Configures this class to be able to make use of Cuda accelerated value retrieval from the given
            scene sdf. This class can always only be used with at most one SceneSDF at a time in combination with Cuda.
        """
        self._cuda_sdf_interpolator = scene_sdf.get_cuda_interpolator(self._locc_positions)

    def setup_cuda_grid_access(self, grid):
        """
            Configures this class to be able to make use of Cuda accelerated value
            retrieval from the given grid.
            ---------
            Arguments
            ---------
            grid, VoxelGrid
            -------
            Returns
            -------
            interpolator id, int - id to identify the cuda interpolator. Provide this id to any of the
                functions below (that operator on VoxelGrid's) in order to use the cuda interpolator.
        """
        self._cuda_interpolators.append(grid.get_cuda_position_interpolator(self._locc_positions))
        return len(self._cuda_interpolators) - 1

    def sum(self, field, tf=None, default_value=0.0, cuda_id=None):
        """
            Sum up all cell values that are occupied by the link represented by this object.
            ---------
            Arguments
            ---------
            field, VoxelGrid - a voxel grid filled with values to sum up
            tf, numpy array of shape 4x4 - transformation matrix from this link's frame to the global frame that field
                is expecting. If None, link.GetTransform is used.
            default_value, float - Value to add if an occupied cell of this link is out of bounds of the field.
            cuda_id, int - if provided, ignore the value given for field and instead utilize the cuda interpolator
                with the given id.
            -------
            Returns
            -------
            result, float
        """
        if tf is None:
            tf = self._link.GetTransform()
        if cuda_id is not None:
            # TODO what about out of bounds values?
            return self._cuda_interpolators[cuda_id].sum(tf)
        query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
        _, indices, _ = field.map_to_grid_batch(query_pos, index_type=np.float_)
        if indices is not None:
            values_to_sum = field.get_cell_values(indices)
            return np.sum(values_to_sum) + (query_pos.shape[0] - indices.shape[0]) * default_value
        return query_pos.shape[0] * default_value

    def min(self, field, tf=None, cuda_id=None):
        """
            Return the minimal cell value occupied by this object.
            ---------
            Arguments
            ---------
            field, VoxelGrid - a voxel grid filled with values to get min from
            tf, numpy array of shape 4x4 - transformation matrix from this link's frame to the global frame that field
                is expecting. If None, link.GetTransform is used.
            cuda_id, int - if provided, ignore the value given for field and instead utilize the cuda interpolator
                with the given id.
            -------
            Returns
            -------
            result, float - minimal cell value, None if all positions are out of bounds
        """
        if tf is None:
            tf = self._link.GetTransform()
        if cuda_id is not None:
            # TODO what about out of bounds values?
            return self._cuda_interpolators[cuda_id].min(tf)
        query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
        _, indices, _ = field.map_to_grid_batch(query_pos, index_type=np.float_)
        if indices is not None:
            values = field.get_cell_values(indices)
            return np.min(values)
        return None

    def min_grad(self, field, tf=None, cuda_id=None):
        """
            Return the gradient at the cell with minimal value.
            ---------
            Arguments
            ---------
            field, VoxelGrid - a voxel grid filled with values to get min from
            tf, numpy array of shape 4x4 - transformation matrix from this link's frame to the global frame that field
                is expecting. If None, link.GetTransform is used.
            cuda_id, int - if provided, ignore the value given for field and instead utilize the cuda interpolator
                with the given id.
            -------
            Returns
            -------
            min_val, float - minimal cell value, None if out of bounds
            pos, np.array of shape (3, 3) - local position of the minimizing cell, None if out of bounds
            grad, np.array of shape (3, 3) - gradient at the minimizing cell, None if out of bounds
        """
        if tf is None:
            tf = self._link.GetTransform()
        if cuda_id is not None:
            # TODO what about out of bounds values?
            vals, grads = self._cuda_interpolators[cuda_id].gradient(tf)
            min_index = np.argmin(vals)
            return vals[min_index], self._locc_positions[min_index], grads[min_index]
        query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
        _, indices, _ = field.map_to_grid_batch(query_pos, index_type=np.float_)
        if indices is not None:
            mask, vals, grads = field.get_cell_gradients_pos(query_pos, b_return_values=True)
            min_index = np.argmin(vals)
            pos = self._locc_positions[mask]
            return vals[min_index], pos[min_index], grads[min_index]
        return None, None, None

    def max(self, field, tf=None, cuda_id=None):
        """
            Return the maximal cell value occupied by this object.
            ---------
            Arguments
            ---------
            field, VoxelGrid - a voxel grid filled with values to get min from
            tf, numpy array of shape 4x4 - transformation matrix from this link's frame to the global frame that field
                is expecting. If None, link.GetTransform is used.
            cuda_id, int - if provided, ignore the value given for field and instead utilize the cuda interpolator
                with the given id.
            -------
            Returns
            -------
            result, float - minimal cell value, None if all positions are out of bounds
        """
        if tf is None:
            tf = self._link.GetTransform()
        if cuda_id is not None:
            # TODO what about out of bounds values?
            return self._cuda_interpolators[cuda_id].max(tf)
        query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
        _, indices, _ = field.map_to_grid_batch(query_pos, index_type=np.float_)
        if indices is not None:
            values = field.get_cell_values(indices)
            return np.max(values)
        return None

    def max_grad(self, field, tf=None, cuda_id=None):
        """
            Return the gradient at the cell with maximal value.
            ---------
            Arguments
            ---------
            field, VoxelGrid - a voxel grid filled with values to get min from
            tf, numpy array of shape 4x4 - transformation matrix from this link's frame to the global frame that field
                is expecting. If None, link.GetTransform is used.
            cuda_id, int - if provided, ignore the value given for field and instead utilize the cuda interpolator
                with the given id.
            -------
            Returns
            -------
            max_val, float - minimal cell value, None if out of bounds
            pos, np.array of shape (3, 3) - local position of the minimizing cell, None if out of bounds
            grad, np.array of shape (3, 3) - gradient at the minimizing cell, None if out of bounds
        """
        if tf is None:
            tf = self._link.GetTransform()
        if cuda_id is not None:
            # TODO what about out of bounds values?
            vals, grads = self._cuda_interpolators[cuda_id].gradient(tf)
            max_index = np.argmax(vals)
            return vals[max_index], self._locc_positions[max_index], grads[max_index]
        query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
        _, indices, _ = field.map_to_grid_batch(query_pos, index_type=np.float_)
        if indices is not None:
            mask, vals, grads = field.get_cell_gradients_pos(query_pos, b_return_values=True)
            max_index = np.argmax(vals)
            pos = self._locc_positions[mask]
            return vals[max_index], pos[max_index], grads[max_index]
        return None, None, None

    def compute_gradients(self, field, tf=None, cuda_id=None):
        """
            Compute the gradients at the positions of this grid's cells.
            ---------
            Arguments
            ---------
            field, VoxelGrid - a voxel grid filled with values to compute gradient for
            tf, numpy array of shape 4x4 - transformation matrix from this link's frame to the global frame that field
                is expecting. If None, link.GetTransform is used.
            cuda_id, int - if provided, ignore the value given for field and instead utilize the cuda interpolator
                with the given id.
            -------
            Returns
            -------
            gradients, np.array of shape (n, 3) - gradients at all n cell positions (gradient is 0 if outside of field)
            loc_positions, np.array of shape (n, 3) - local positions of cells
        """
        if tf is None:
            tf = self._link.GetTransform()
        if cuda_id is not None:
            # TODO what about out of bounds values?
            _, grads = self._cuda_interpolators[cuda_id].gradient(tf)
            return grads, self._locc_positions
        query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
        rgradients = np.zeros_like(self._locc_positions)
        valid_mask, gradients = field.get_cell_gradients_pos(query_pos, b_return_values=False)
        rgradients[valid_mask] = gradients
        return gradients, self._locc_positions

    def count(self, field, val, comp=np.greater_equal, tf=None):
        """
            Count how many cell values in field occupied by this link are >, >=, ==, <= or < than val.
            ---------
            Arguments
            ---------
            field, VoxelGrid - a voxel grid filled with values to compare val to
            val, float - reference value to compare to
            comp, np.ufunc - may either be np.greater_equal, np.greater, np.equal, np.less or np.less_equal
            tf, numpy array of shape 4x4 - transformation matrix from this link's frame to the global frame that field
                is expecting. If None, link.GetTransform() is used.
        """
        if tf is None:
            tf = self._link.GetTransform()
        query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
        _, indices, _ = field.map_to_grid_batch(query_pos, index_type=np.float_)
        if indices is not None:
            return np.count_nonzero(comp(field.get_cell_values(indices), val))
        return 0

    def compute_intersection(self, scene_sdf, tf=None):
        """
            Return how many cells of this occupancy grid correspond to positions
            with negative or zero distance in the given scene_sdf.
            ---------
            Arguments
            ---------
            scene_sdf, SceneSDF - signed distance field of a scene
            tf (optional), np.array of shape 4,4 - transformation matrix for this link,
                if not provided current tf of link is used.
            -------
            Returns
            -------
            num_intersecting_cells, int - number of cells with negative or zero distance
            percentage, float - percentage of cells that have negative or zero distance
            volume, float - the total volume of the occupied cells
        """
        if tf is None:
            tf = self._link.GetTransform()
        query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
        distances = scene_sdf.get_distances(query_pos)
        num_intersecting_cells = np.sum(distances <= 0.0)
        return num_intersecting_cells, float(num_intersecting_cells) / self._num_occupied_cells, num_intersecting_cells * self._cell_volume

    def compute_obstacle_cost(self, scene_sdf, tf=None, bgradients=False, eps=None):
        """
            Compute obstacle penetration cost from CHOMP paper C(x) = sum_i(c(x_i)) where x_i are the voxel positions
            and:
                    -d(x) + eps / 2             if d(x) < 0.0
            c(x) =  1/(2eps)(d(x) - eps)^2      if 0 <= d(x) <= eps
                    0                           else
            with d(x) being the minimal distance to an obstacle (i.e. what is given by scene_sdf)
            ---------
            Arguments
            ---------
            scene_sdf, SceneSDF - to retrieve d(x)
            tf(optional), np array of shape (4, 4) - link pose, if not provided use link's current pose
            bgradients(optional), bool - if true, also return gradients of c(x_i) w.r.t x_i (one for each voxel)
            eps(optional), float - eps in above formula = minimal distance a collision-free link should have.
                If not provided cell size of underlying occupancy grid is used.
            -------
            Returns
            -------
            values, np.array of shape (n, 3) - individual c(x_i), to compute C(x) run np.sum(values)
            gradients(optional), numpy array of shape (n, 3) - gradients dc(x_i)/dx_i if bgradients is True
            local_positions(optional), np array of shape (n, 3) - positions of voxel positions in link frame if bgradients is True
        """
        if tf is None:
            tf = self._link.GetTransform()
        if eps is None:
            eps = self._grid.get_cell_size()
        if self._cuda_sdf_interpolator is None:
            query_pos = np.dot(self._locc_positions, tf[:3, :3].transpose()) + tf[:3, 3]
            if bgradients:
                distances, dist_gradients = scene_sdf.get_distances_grad(query_pos)
                return_values, gradients = utils.chomps_distance(distances, eps, dist_gradients)
                return return_values, gradients, self._locc_positions
            else:
                distances = scene_sdf.get_distances(query_pos)
                return_values = utils.chomps_distance(distances, eps)
                return return_values
        else:
            if bgradients:
                # distances, dist_gradients = self._cuda_sdf_interpolator.gradient(tf)
                # return_values, gradients = utils.chomps_distance(distances, eps, dist_gradients)
                return_values, gradients = self._cuda_sdf_interpolator.chomps_smooth_distance(tf, eps)
                return return_values, gradients, self._locc_positions
            else:
                distances = self._cuda_sdf_interpolator.interpolate(tf)
                return utils.chomps_distance(distances, eps)

    def get_link(self):
        """
            Return the link this grid represents.
        """
        return self._link

    def get_volume(self):
        """
            Return approximate volume of the link.
        """
        return self._total_volume

    def get_num_occupied_cells(self):
        """
            Return the number of cells in the underlying grid that are occupied.
        """
        return self._num_occupied_cells

    def get_cell_size(self):
        """
            Return cell size of underlying grid.
        """
        return self._grid.get_cell_size()

    def save(self, filename):
        """
            Save the underlying grid to file.
            ---------
            Arguments
            ---------
            filename, string - path where to store grid
        """
        self._grid.save(filename)

    @staticmethod
    def load(filename, link):
        """
            Construct a RigidBodyOccupancyGrid from a grid stored on disk.
            ---------
            Arguments
            ---------
            filename, string - the filename where to load grid from
            link, OpenRAVE link - the link this grid is for
        """
        grid = grid_mod.VoxelGrid.load(filename)
        return RigidBodyOccupancyGrid(None, link, grid)

    @staticmethod
    def create_occupancy_grids(cell_size, links):
        grids = construct_grid(links, cell_size)
        return [RigidBodyOccupancyGrid(None, link, grid) for link, grid in zip(links, grids)]


if __name__ == '__main__':
    def mtest_vis_occtree():
        env = orpy.Environment()
        import os
        env.Load(os.path.dirname(__file__) + '/../../../data/bunny/objectModel.ply')
        env.SetViewer('qtcoin')
        body = env.GetBodies()[0]
        # body.SetVisible(False)
        octree = OccupancyOctree(0.005, body.GetLinks()[0])
        # octree._grid.save('/tmp/test_grid')
        handles = octree.visualize(10)
        import IPython
        IPython.embed()

    def mtest_sum_occgrid():
        env = orpy.Environment()
        import os
        base_path = os.path.dirname(__file__) + '/../../../'
        env.Load(base_path + 'data/environments/placement_exp_0.xml')
        env.SetViewer('qtcoin')
        body = env.GetKinBody('crayola')
        occupancy_grid = RigidBodyOccupancyGrid(0.01, body.GetLinks()[0])
        sdf = sdf_core.SDF.load(base_path + 'data/sdfs/placement_exp_0_low_res')
        sdf_grid = sdf.get_grid()
        print occupancy_grid.sum(sdf_grid)
        import IPython
        IPython.embed()

    def mtest_count_occgrid():
        env = orpy.Environment()
        import os
        base_path = os.path.dirname(__file__) + '/../../../'
        env.Load(base_path + 'data/environments/placement_exp_0.xml')
        env.SetViewer('qtcoin')
        body = env.GetKinBody('crayola')
        occupancy_grid = RigidBodyOccupancyGrid(0.01, body.GetLinks()[0])
        sdf = sdf_core.SDF.load(base_path + 'data/sdfs/placement_exp_0_low_res')
        sdf_grid = sdf.get_grid()
        print occupancy_grid.count(sdf_grid, 0.0)
        import IPython
        IPython.embed()

    def mtimeit_sum_occgrid():
        setup_code = """import openravepy as orpy;\
        from __main__ import RigidBodyOccupancyGrid;\
        import hfts_grasp_planner.sdf.core as sdf_core;\
        env = orpy.Environment();\
        base_path = \'/home/joshua/projects/placement_planning/src/hfts_grasp_planner/\';\
        env.Load(base_path + \'data/environments/placement_exp_0.xml\');\
        body = env.GetKinBody(\'crayola\');\
        occupancy_grid = RigidBodyOccupancyGrid(0.01, body.GetLinks()[0]);\
        sdf = sdf_core.SDF.load(base_path + \'data/sdfs/placement_exp_0_low_res\');\
        sdf_grid = sdf.get_grid();\
        """
        run_code = """occupancy_grid.sum(sdf_grid);"""
        import timeit
        num_runs = 10000
        total = timeit.timeit(run_code, setup=setup_code, number=num_runs)
        print "Total:",  total, "per run: ", total / num_runs

    def mtimeit_count_occgrid():
        setup_code = """import openravepy as orpy;\
        from __main__ import RigidBodyOccupancyGrid;\
        import hfts_grasp_planner.sdf.core as sdf_core;\
        env = orpy.Environment();\
        base_path = \'/home/joshua/projects/placement_planning/src/hfts_grasp_planner/\';\
        env.Load(base_path + \'data/environments/placement_exp_0.xml\');\
        body = env.GetKinBody(\'crayola\');\
        occupancy_grid = RigidBodyOccupancyGrid(0.01, body.GetLinks()[0]);\
        sdf = sdf_core.SDF.load(base_path + \'data/sdfs/placement_exp_0_low_res\');\
        sdf_grid = sdf.get_grid();\
        """
        run_code = """occupancy_grid.count(sdf_grid, 0.0);"""
        import timeit
        num_runs = 10000
        total = timeit.timeit(run_code, setup=setup_code, number=num_runs)
        print "Total:",  total, "per run: ", total / num_runs

    def mtimeit_intersection():
        setup_code = """import openravepy as orpy;\
        from __main__ import OccupancyOctree;\
        import hfts_grasp_planner.sdf.core as sdf_core;\
        env = orpy.Environment();\
        base_path = \'/home/joshua/projects/placement_planning/src/hfts_grasp_planner/\';\
        env.Load(base_path + \'data/environments/placement_exp_0.xml\');\
        body = env.GetKinBody(\'crayola\');\
        occupancy_tree = OccupancyOctree(0.01, body.GetLinks()[0]);\
        scene_sdf = sdf_core.SceneSDF(env, [], [\'Yumi\', \'crayola\']);\
        scene_sdf.load(base_path + \'data/sdfs/placement_exp_0.sdf\');\
        """
        run_code = """occupancy_tree.compute_intersection(scene_sdf);"""
        import timeit
        num_runs = 10000
        total = timeit.timeit(run_code, setup=setup_code, number=num_runs)
        print "Total:",  total, "per run: ", total / num_runs

    def mtimeit_grid_intersection():
        setup_code = """import openravepy as orpy;\
        from __main__ import RigidBodyOccupancyGrid;\
        import hfts_grasp_planner.sdf.core as sdf_core;\
        env = orpy.Environment();\
        base_path = \'/home/joshua/projects/placement_planning/src/hfts_grasp_planner/\';\
        env.Load(base_path + \'data/environments/placement_exp_0.xml\');\
        body = env.GetKinBody(\'crayola\');\
        occupancy_grid = RigidBodyOccupancyGrid(0.01, body.GetLinks()[0]);\
        scene_sdf = sdf_core.SceneSDF(env, [], [\'Yumi\', \'crayola\']);\
        scene_sdf.load(base_path + \'data/sdfs/placement_exp_0.sdf\');\
        """
        run_code = """occupancy_grid.compute_intersection(scene_sdf);"""
        import timeit
        num_runs = 10000
        total = timeit.timeit(run_code, setup=setup_code, number=num_runs)
        print "Total:",  total, "per run: ", total / num_runs

    # mtest_count_occgrid()
    # mtest_sum_occgrid()
    # mtimeit_sum_occgrid()
    # mtimeit_count_occgrid()
    # mtimeit_intersection()
    mtimeit_grid_intersection()
