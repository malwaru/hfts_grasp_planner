"""
    This module provides functionalities to compute the degree of collisions
    between a kinbody and its environment. The algorithms within this module
    build on the availability of a Signed Distance Field of the environment.
"""
import itertools
import collections
import numpy as np
import openravepy as orpy
import hfts_grasp_planner.sdf.core as sdf_core


class OccupancyOctree(object):
    """
        This class represents an occupancy map for a rigid body
        using an octree representation. Each cell of the octree stores whether
        it is part of the volume of the rigid body or not. Thus the octree provides
        a hierarchical representation of the body's volume. 
        This class allows to efficiently compute to what degree a kinbody collides with its 
        environment, if a signed distance field for this environment exists.
    """
    class OccupancyOctreeCell(object):
        """
            Represents a cell of an OccupancyOctree
        """

        def __init__(self, idx_box, depth, tree):
            """
                Create a new cell.
                @param idx_box - numpy array describing the grid index range
                    [min_x, min_y, min_z, max_x, max_y, max_z] (max_x, max_y, max_z) are exclusive.
                @param depth - depth of this node in the hierarchy
                @param tree - the tree this cell belongs to
            """
            self.idx_box = idx_box
            self.occupied = True
            self.children = []
            self.depth = depth
            self.num_occupied_leaves = 0
            self.tree = tree
            self.cart_dimensions = (idx_box[3:] - idx_box[:3]) * tree._grid.get_cell_size()
            self.radius = np.linalg.norm(self.cart_dimensions / 2.0)
            min_cell_pos = tree._grid.get_cell_position(self.idx_box[:3], b_center=False)
            self.cart_center = min_cell_pos + 0.5 * self.cart_dimensions

    def __init__(self, cell_size, body):
        """
            Construct a new OccupancyOctree.
            @param cell_size - minimum edge length of a cell (all cells are cubes)
        """
        self._body = body
        self._grid = None
        self._root = None
        self._total_volume = 0.0
        self._depth = 0
        self._construct_grid(cell_size)
        self._construct_octree(cell_size)

    def _construct_grid(self, cell_size):
        """
            Construct the occupancy grid that this octree represents.
        """
        env = self._body.GetEnv()
        with env:
            original_tf = self._body.GetTransform()
            self._body.SetTransform(np.eye(4))  # set body to origin frame
            body_flags = []
            for body in env.GetBodies():
                body_flags.append((body.IsEnabled(), body.IsVisible()))
                body.Enable(False)
                body.SetVisible(False)
            self._body.Enable(True)
            # construct a binary occupancy grid of the body
            local_aabb = self._body.ComputeAABB()
            self._grid = sdf_core.VoxelGrid((local_aabb.pos(), local_aabb.extents() * 2.0), cell_size)
            collision_map_builder = sdf_core.OccupancyGridBuilder(env, cell_size)
            collision_map_builder.compute_grid(self._grid)
            collision_map_builder.clear()
            # set the body back to its original pose
            self._body.SetTransform(original_tf)
            # restore flags
            for body, body_flag in itertools.izip(env.GetBodies(), body_flags):
                body.Enable(body_flag[0])
                body.SetVisible(body_flag[1])

    def _construct_octree(self, cell_size):
        """
            Construct the octree. Assumes self._grid has been created.
        """
        # first construct the root
        root_aabb_idx = np.zeros((6,), dtype=np.int)  # stores indices of cells in grid covered by this node
        root_aabb_idx[3:] = self._grid.get_num_cells()  # it represents all cells
        self._root = OccupancyOctree.OccupancyOctreeCell(root_aabb_idx, 0, self)
        # construct tree
        nodes_to_refine = collections.deque([self._root])
        while nodes_to_refine:  # as long as there are nodes to refine
            current_node = nodes_to_refine.popleft()
            # handle = self.draw_cell(current_node)
            min_idx = current_node.idx_box[:3]
            max_idx = current_node.idx_box[3:]
            idx_range = max_idx - min_idx  # the number of cells along each axis in this box
            volume = np.multiply.reduce(idx_range)
            min_value = self._grid.get_min_value(min_idx, max_idx)
            max_value = self._grid.get_max_value(min_idx, max_idx)
            assert(volume >= 1)
            assert(min_value == 0.0 or min_value == 1.0)
            assert(max_value == 0.0 or max_value == 1.0)
            if volume == 1 or max_value == 0.0:
                # we are either at the bottom of the hierarchy or all children are free
                current_node.occupied = min_value == 1.0
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
                        child_node = OccupancyOctree.OccupancyOctreeCell(child_idx_box, current_node.depth + 1, self)
                        current_node.children.append(child_node)
                        nodes_to_refine.append(child_node)
                        self._depth = max(self._depth, child_node.depth)
        # lastly, update num_occupied_leaves flags

        def compute_num_occupied_leaves(node):
            # helper function to recursively compute the number of occupied leaves
            if not node.children:
                return node.num_occupied_leaves
            if not node.occupied:
                assert(node.num_occupied_leaves == 0)
                return 0
            for child in node.children:
                node.num_occupied_leaves += compute_num_occupied_leaves(child)
            return node.num_occupied_leaves
        compute_num_occupied_leaves(self._root)
        self._total_volume = self._root.num_occupied_leaves * np.power(self._grid.get_cell_size(), 3)

    def compute_intersection(self, scene_sdf):
        """
            Computes the intersection between this octree and the geometry
            of the scene described by the provided scene sdf.
            @param scene_sdf - signed distance field of the environment that the body
                this map belongs to resides in
            @return (v, rv, dc, adc) -
                v is the total volume that is intersecting
                rv is this volume relative to the body's total volume, i.e. in range [0, 1]
                dc is a cost that is computed by (approximately) summing up all signed
                    distances of intersecting cells
                adc is this cost divided by the number of intersecting cells, i.e. the average
                    signed distance of the intersecting cells
        """
        if not self._root.occupied:
            return 0.0, 0.0, 0.0, 0.0
        tf = self._body.GetTransform()
        num_intersecting_leaves = 0
        distance_cost = 0.0
        layer_idx = 0
        # current_layer = [self._root]  # invariant current_layer items are occupied
        current_layer = collections.deque([self._root])  # invariant current_layer items are occupied
        next_layer = collections.deque()
        # iterate through hierarchy layer by layer (bfs) - this way we can perform more efficient batch distance queries
        while current_layer:
            # first get the positions of all cells on the current layer
            query_positions = np.ones((len(current_layer), 4))
            query_positions[:, :3] = np.array([cell.cart_center for cell in current_layer])
            query_positions = np.dot(query_positions, tf.transpose())
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
                    distance_cost += cell.num_occupied_leaves * (dist - cell.radius)
                else:  # boundary, partly in collision
                    if cell.children:  # as long as there are children, we can descend
                        next_layer.extend([child for child in cell.children if child.occupied])
                    else:
                        num_intersecting_leaves += cell.num_occupied_leaves
                        # subtracting the radius here guarantees that distance_cost is always negative
                        distance_cost += dist - cell.radius
            # switch to next layer
            current_layer, next_layer = next_layer, current_layer
            next_layer.clear()
            layer_idx += 1
        intersection_volume = num_intersecting_leaves * np.power(self._grid.get_cell_size(), 3)
        relative_volume = num_intersecting_leaves / float(self._root.num_occupied_leaves)
        normalized_distance_cost = distance_cost / float(self._root.num_occupied_leaves)
        return intersection_volume, relative_volume, distance_cost, normalized_distance_cost

    def draw_cell(self, cell):
        """
            Draw the specified cell.
            @param cell - OccupancyOctreeCell to render
            @return handle - OpenRAVE handle
        """
        tf = self._body.GetTransform()
        env = self._body.GetEnv()
        color = np.array([1.0, 0.0, 0.0, 1.0])
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


if __name__ == '__main__':
    env = orpy.Environment()
    import os
    env.Load(os.path.dirname(__file__) + '/../../../data/bunny/objectModel.ply')
    env.SetViewer('qtcoin')
    body = env.GetBodies()[0]
    octree = OccupancyOctree(0.005, body)
    # octree._grid.save('/tmp/test_grid')
    handles = octree.visualize(10)
    import IPython
    IPython.embed()
