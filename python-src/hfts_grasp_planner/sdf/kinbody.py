"""
    This module provides functionalities to compute the degree of collisions
    between a kinbody and its environment. The algorithms within this module
    build on the availability of a Signed Distance Field of the environment.
"""
import itertools
import collections
# import rospy
import numpy as np
import openravepy as orpy
import hfts_grasp_planner.sdf.core as sdf_core


class OccupancyOctree(object):
    """
        This class represents an occupancy map for a rigid body
        using an octree representation. Each cell of the octree stores whether
        it is part of the volume of the rigid body or not. Thus the octree provides
        a hierarchical representation of the body's volume.
        This class allows to efficiently compute to what degree a rigid body collides with its
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
            self.max_int_distance = 0.0
            self.tree = tree
            self.cart_dimensions = (idx_box[3:] - idx_box[:3]) * tree._grid.get_cell_size()
            self.radius = np.linalg.norm(self.cart_dimensions / 2.0)
            min_cell_pos = tree._grid.get_cell_position(self.idx_box[:3], b_center=False)
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

    def __init__(self, cell_size, link):
        """
            Construct a new OccupancyOctree.
            ---------
            Arguments
            ---------
            cell_size, float - minimum edge length of a cell (all cells are cubes)
            link, Link to create the tree for
        """
        self._ray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._link = link
        self._body = link.GetParent()
        # self._object_diameter = 2.0 * np.linalg.norm(self._link.ComputeLocalAABB().extents())
        self._grid = None
        self._root = None
        self._total_volume = 0.0
        self._depth = 0
        self._construct_grid(cell_size)
        self._sdf = sdf_core.SDF(sdf_core.SDFBuilder.compute_sdf(self._grid))
        self._max_possible_penetration = np.abs(self._sdf.min())
        self._construct_octree(cell_size)

    def _construct_grid(self, cell_size):
        """
            Construct the occupancy grid that this octree represents.
        """
        env = self._body.GetEnv()
        with env:
            original_tf = self._link.GetTransform()
            self._link.SetTransform(np.eye(4))  # set link to origin frame
            body_flags = []
            for body in env.GetBodies():
                body_flags.append((body.IsEnabled(), body.IsVisible()))
                body.Enable(False)
                body.SetVisible(False)
            self._link.Enable(True)
            # construct a binary occupancy grid of the body
            local_aabb = self._link.ComputeAABB()
            self._grid = sdf_core.VoxelGrid((local_aabb.pos(), local_aabb.extents() * 2.0), cell_size, dtype=bool)
            collision_map_builder = sdf_core.OccupancyGridBuilder(env, cell_size)
            collision_map_builder.compute_grid(self._grid)
            collision_map_builder.clear()
            # set the body back to its original pose
            self._link.SetTransform(original_tf)
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
                current_node.occupied = min_value == 1
                current_node.num_occupied_leaves = volume if current_node.occupied else 0
                # NOTE there is an error of up to the cell radius here
                current_node.max_int_distance = abs(self._sdf.get_distance(
                    current_node.cart_center)) if current_node.occupied else 0.0
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

        def update_bottom_up(node):
            # helper function to recursively compute the number of occupied leaves and penetration distances
            if not node.children:
                return node.num_occupied_leaves, node.max_int_distance
            if not node.occupied:
                assert(node.num_occupied_leaves == 0)
                assert(node.max_int_distance == 0.0)
                return 0, 0.0
            for child in node.children:
                child_leaves, child_int_dist = update_bottom_up(child)
                node.num_occupied_leaves += child_leaves
                node.max_int_distance = max(child_int_dist, node.max_int_distance)
            return node.num_occupied_leaves, node.max_int_distance
        update_bottom_up(self._root)
        self._total_volume = self._root.num_occupied_leaves * np.power(self._grid.get_cell_size(), 3)

    def compute_intersection(self, scene_sdf):
        """
            Computes the intersection between this octree and the geometry
            of the scene described by the provided scene sdf.
            @param scene_sdf - signed distance field of the environment that the body
                this map belongs to resides in
            @return (v, rv, dc, adc, mid, med) -
                v is the total volume that is intersecting
                rv is this volume relative to the body's total volume, i.e. in range [0, 1]
                dc is a cost that is computed by (approximately) summing up all signed
                    distances of intersecting cells
                adc is this cost divided by the number of intersecting cells, i.e. the average
                    signed distance of the intersecting cells
                max_int_distance, float - maximal interior distance of a colliding cell
                max_ext_distance, float - maximal exterior distance of a colliding cell
        """
        if not self._root.occupied:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        tf = self._link.GetTransform()
        num_intersecting_leaves = 0
        distance_cost = 0.0
        layer_idx = 0
        max_int_distance = 0.0  # maximal interior penetration distance
        max_ext_distance = 0.0  # maximal exterior penetration distance
        # current_layer = [self._root]  # invariant current_layer items are occupied
        current_layer = collections.deque([self._root])  # invariant current_layer items are occupied
        next_layer = collections.deque()
        # iterate through hierarchy layer by layer (bfs) - this way we can perform more efficient batch distance queries
        while current_layer:
            # first get the positions of all cells on the current layer
            query_positions = np.array([cell.cart_center for cell in current_layer])
            # TODO we also don't need this additional 1 for this operation (decompose transformation in rotation and translation)
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
                    max_int_distance = max(max_int_distance, cell.max_int_distance)
                else:  # boundary, partly in collision
                    if cell.children:  # as long as there are children, we can descend
                        next_layer.extend([child for child in cell.children if child.occupied])
                    else:
                        num_intersecting_leaves += cell.num_occupied_leaves
                        # subtracting the radius here guarantees that distance_cost is always negative
                        ext_distance = dist - cell.radius
                        distance_cost += ext_distance
                        max_ext_distance = max(max_ext_distance, abs(ext_distance))
                        max_int_distance = max(max_int_distance, cell.max_int_distance)
            # switch to next layer
            current_layer, next_layer = next_layer, current_layer
            next_layer.clear()
            layer_idx += 1
        intersection_volume = num_intersecting_leaves * np.power(self._grid.get_cell_size(), 3)
        relative_volume = num_intersecting_leaves / float(self._root.num_occupied_leaves)
        normalized_distance_cost = distance_cost / float(self._root.num_occupied_leaves)
        return intersection_volume, relative_volume, distance_cost, normalized_distance_cost, max_int_distance, max_ext_distance

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
        env = self._body.GetEnv()
        max_penetration = 0.0
        # dist_to_surface = 0.0  # distance to object surface along the direction of penetration
        direction = np.array([0.0, 0.0, 0.0]) if b_compute_dir else None
        max_pen_position = np.array([0.0, 0.0, 0.0])
        if not self._root.occupied:
            return max_penetration, direction
        tf = self._link.GetTransform()
        current_layer = collections.deque([self._root])  # invariant current_layer items are occupied
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
                                # compute the distance from this cell to the object's surface along the direction
                                # self._ray[:3] = pos[:3]
                                # self._ray[3:] = -self._object_diameter * direction / np.linalg.norm(direction)
                                # handle = env.drawarrow(self._ray[:3], self._ray[:3] + self._ray[3:], linewidth=0.002)
                                # collisions, contacts = env.CheckCollisionRays(
                                #     np.array([self._ray]), self._body)
                                # rospy.logdebug("Cell with id " + str(cell.idx_box) + " is new maximal penetrator.")
                                # rospy.logdebug("Distance is " + str(dist) + "dir is " + str(direction))
                                # rospy.logdebug("Ray is " + str(self._ray) + "dir is " + str(direction))
                                # rospy.logdebug("Collision result is " + str(collisions) +
                                #                " Found contacts " + str(contacts))
                                # if collisions[0]:
                                #     dist_to_surface = np.linalg.norm(contacts[0, :3] - pos[:3])
                                # else:
                                #     dist_to_surface = cell.radius
                                # switch to next layer
            current_layer, next_layer = next_layer, current_layer
            next_layer.clear()
        return max_penetration, direction, max_pen_position  # , dist_to_surface

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
    # body.SetVisible(False)
    octree = OccupancyOctree(0.005, body.GetLinks()[0])
    # octree._grid.save('/tmp/test_grid')
    handles = octree.visualize(10)
    import IPython
    IPython.embed()
