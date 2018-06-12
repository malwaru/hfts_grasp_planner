"""
    This module provides functionalities to compute the degree of collisions
    between a kinbody and its environment. The algorithms within this module
    build on the availability of a Signed Distance Field of the environment.
"""
import itertools
import collections
import numpy as np
import openravepy as orpy

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
        def __init__(self, aabb, depth):
            """
                Create a new cell.
                @param aabb - numpy array describing the bounding box 
                    [min_x, min_y, min_z, max_x, max_y, max_z]
                @param depth - depth of this node in the hierarchy
            """
            self.aabb = aabb
            self.dimensions = aabb[3:] - aabb[:3]
            self.occupied = True
            self.children = []
            self.depth = depth
            self.position = aabb[3:] + 0.5 * self.dimensions

        def is_leaf(self):
            """
                Return whether this cell is a leaf.
            """
            return len(self.children) == 0


    class BodyManager(object):
        """
            Internal helper class for creating binary collision maps
        """
        def __init__(self, env):
            """
                Create a new BodyManager
                @param env - OpenRAVE environment
            """
            self._env = env
            self._bodies = {}
            self._active_body = None

        def get_body(self, dimensions):
            """
                Get a cell kinbody with the given dimensions.
                @param dimensions - numpy array (wx, wy, wz)
            """
            new_active_body = None
            key = tuple(dimensions)
            if key in self._bodies:
                new_active_body = self._bodies[key]
            else:
                new_active_body = orpy.RaveCreateKinBody(self._env, '')
                new_active_body.SetName("CollisionCheckBody" + str(dimensions[0]) +
                                        str(dimensions[1]) + str(dimensions[2]))
                new_active_body.InitFromBoxes(np.array([[0, 0, 0,
                                                         dimensions[0] / 2.0,
                                                         dimensions[1] / 2.0,
                                                         dimensions[2] / 2.0]]),
                                              True)
                self._env.AddKinBody(new_active_body)
                self._bodies[key] = new_active_body
            if new_active_body is not self._active_body and self._active_body is not None:
                self._active_body.Enable(False)
                self._active_body.SetVisible(False)
            self._active_body = new_active_body
            self._active_body.Enable(True)
            self._active_body.SetVisible(True)
            return self._active_body

        def disable_bodies(self):
            """
                Disable and hide all bodies
            """
            if self._active_body is not None:
                self._active_body.Enable(False)
                self._active_body.SetVisible(False)
                self._active_body = None

        def clear(self):
            """
                Remove and destroy all bodies.
            """
            for body in self._bodies.itervalues():
                self._env.Remove(body)
                body.Destroy()
            self._bodies = {}


    def __init__(self, min_cell_size, body):
        """
            Construct a new OccupancyOctree.
            @param min_cell_size - minimum volume of a cell
                Cells are not split apart if their children would be smaller in volume than this.
        """
        self._min_cell_size = min_cell_size
        self._body = body  
        self._root = None
        self._total_volume = 0.0
        self._depth = 0
        self._construct_octree()

    def _construct_octree(self):
        """
            Construct the octree. 
        """
        env = self._body.GetEnv()
        with env:
            # prepare construction
            body_manager = OccupancyOctree.BodyManager(env)
            original_tf = self._body.GetTransform()
            self._body.SetTransform(np.eye(4))  # set body to origin frame
            tf = np.eye(4)  # tf that we use later
            # construct root element
            local_aabb = self._body.ComputeAABB()
            root_aabb = np.empty((6,))
            root_aabb[:3] = local_aabb.pos() - local_aabb.extents()
            root_aabb[3:] = local_aabb.pos() + local_aabb.extents()
            self._root = OccupancyOctree.OccupancyOctreeCell(root_aabb, 0) 
            # construct tree
            cells_to_refine = collections.deque([self._root])
            while cells_to_refine:  # as long as there are cells to refine
                current_cell = cells_to_refine.popleft()
                cell_body = body_manager.get_body(current_cell.dimensions)
                # set the cell body to its pose
                tf[:3, 3] = current_cell.position
                cell_body.SetTransform(tf)
                current_cell.occupied = env.CheckCollision(self._body, cell_body)
                if current_cell.occupied:  # do we need to refine its children?
                    child_dimensions = current_cell.dimensions / 2.0
                    if np.multiply.reduce(child_dimensions) > self._min_cell_size:
                        child_combinations = itertools.product(range(2), repeat=3)
                        for child_id in child_combinations:
                            child_aabb = np.empty((6,))
                            child_aabb[:3] = current_cell.aabb[:3] + child_id * child_dimensions
                            child_aabb[3:] = child_aabb[:3] + child_dimensions
                            new_child = OccupancyOctree.OccupancyOctreeCell(child_aabb, current_cell.depth + 1)
                            current_cell.children.append(new_child)
                            cells_to_refine.append(new_child)
                            self._depth = max(self._depth, new_child.depth)
                    else:
                        self._total_volume += np.multiply.reduce(current_cell.dimensions)
  
            body_manager.clear()
            self._body.SetTransform(original_tf)
    
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

    def compute_intersection(self, scene_sdf):
        """
            Computes the intersection between this octree and the geometry
            of the scene described by the provided scene sdf.
            @param scene_sdf - signed distance field of the environment that the body
                this map belongs to resides in
        """
        if not self._root.occupied:
            return 0.0
        tf = self._body.GetTransform()
        intersection_volume = 0.0
        layer_idx = 0
        current_layer = [self._root]  # invariant current_layer items are occupied
        next_layer = [] 
        # iterate through hierarchy layer by layer (bfs)
        while current_layer:
            # first get the positions of all cells on the current layer
            current_layer_positions = [cell.position for cell in current_layer]
            query_positions = np.ones((len(current_layer_positions), 4))
            query_positions[:, :3] = current_layer_positions
            # query distances for all cells on this layer
            distances = scene_sdf.get_distances(np.dot(tf, query_positions))
            # all cells on the same layer have the same dimensions and volume
            radius = np.linalg.norm(current_layer[0].dimensions / 2.0)  
            cell_volume = np.multiply.reduce(current_layer[0].dimensions)
            for idx, cell in enumerate(current_layer):
                if distances[idx] > radius: # none of the points in this cell can be in collision
                    continue
                elif distances[idx] < -1.0 * radius:  
                    # the cell lies so far inside of an obstacle, that it is completely in collision
                    intersection_volume += cell_volume
                else:
                    if layer_idx < self._depth:  # as long as there are children, we can descend
                        next_layer.extend([child for child in cell.children if child.occupied])
                    else:
                        # TODO we should/could check the intersection volume more accurately here
                        intersection_volume += cell_volume
            # switch to next layer
            current_layer = next_layer
            next_layer = []
        return intersection_volume

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
        env = self._body.GetEnv()
        while cells_to_render:
            cell = cells_to_render.popleft()
            if cell.depth == level:
                if cell.occupied:
                    # render this box
                    extents = cell.dimensions / 2.0
                    handle = env.drawbox(cell.aabb[:3] + extents, extents)
                    handles.append(handle)
            else:
                cells_to_render.extend(cell.children)
        return handles


if __name__ == '__main__':
    env = orpy.Environment()
    import os
    env.Load(os.path.dirname(__file__) + '/../../../data/bunny/objectModel.ply')
    env.SetViewer('qtcoin')
    body = env.GetBodies()[0]
    octree = OccupancyOctree(10e-9, body)
    handles = octree.visualize(6)
    import IPython
    IPython.embed()
