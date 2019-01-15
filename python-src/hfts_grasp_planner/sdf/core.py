"""
    This module contains the core components for signed distance fields.
"""
from __future__ import print_function
import math
import time
import yaml
import rospy
import os
import operator
import itertools
import skfmm
import numpy as np
import openravepy as orpy
from itertools import izip, product
import scipy.ndimage.morphology
from hfts_grasp_planner.sdf.grid import VoxelGrid, VectorGrid
from hfts_grasp_planner.sdf.occupancy import OccupancyGridBuilder
from hfts_grasp_planner.sdf.visualization import ORVoxelGridVisualization
# from mayavi import mlab
# from openravepy.misc import ComputeBoxMesh


class SDF(object):
    """
        This class represents a signed distance field.
    """

    def __init__(self, grid, approximation_box=None):
        """
            Creates a new signed distance field.
            You may either create an SDF using a SDFBuilder or by loading it from file.
            In neither case you will have to call this constructor yourself.
            - :grid: a VoxelGrid storing all signed distances - used by SDFBuilder
            - :approximation_box: a box used for approximating distances outside of the grid
        """
        self._grid = grid
        self._approximation_box = approximation_box
        self._or_visualization = None
        if self._approximation_box is None and self._grid:
            self._approximation_box = self._grid.get_aabb()

    def set_transform(self, transform):
        """
            Set the transform for this sdf
            @param transform - numpy 4x4 transformation matrix
        """
        self._grid.set_transform(transform)

    def get_transform(self):
        """
            Returns the current transformation matrix.
        """
        return self._grid.get_transform()

    def _get_heuristic_distance_local(self, local_point):
        """
            Returns a heuristical shortest distance of the given point the closest obstacle surface.
            @param local_point - point as numpy array (x, y, z), assumed to be in local frame
        """
        projected_point = np.clip(local_point, self._approximation_box[:3], self._approximation_box[3:])
        rel_point = local_point - projected_point
        return np.linalg.norm(rel_point)

    def get_heuristic_distance(self, point):
        """
            Returns a heuristical shortest distance of the given point the closest obstacle surface.
            @param point - point as numpy array (x, y, z)
        """
        local_point, idx = self._grid.map_to_grid(point)
        return self._get_heuristic_distance_local(local_point)

    def get_distance(self, point, b_return_dir=False):
        """
            Returns the shortest distance of the given point to the closest obstacle surface.
            @param point - point as a numpy array (x, y, z).
            b_return_dir, bool - if True, also return the direction to the closest free point
        """
        v = None
        dist = None
        local_point, idx = self._grid.map_to_grid(point)
        if idx is not None:
            dist = self._grid.get_cell_value(idx)
        else:
            # point is outside of the grid, need to copmute heuristic value
            dist = self._get_heuristic_distance_local(local_point)
        if b_return_dir:
            if idx is not None:
                v = self._grid.get_additional_data(idx)
            if v is None:
                v = np.array([0.0, 0.0, 0.0])
            return dist, v
        return dist

    def get_distances(self, positions, b_interpolate=True):
        """
            Returns the shortest distance of the given points to the closest obstacle surface respectively.

            Arguments:
            positions - a numpy matrix of shape (n, 3), where n is the number of query points.
            b_interpolate - if true, values are interpolated, otherwise nearest neighbor lookup is used
        """
        distances = np.zeros(positions.shape[0])
        index_type = np.float_ if b_interpolate else np.int
        local_points, grid_indices, valid_mask = self._grid.map_to_grid_batch(positions, index_type=index_type)
        # retrieve the distances for which we have a valid index
        if grid_indices is not None:  # in case we have any valid points
            distances[valid_mask] = self._grid.get_cell_values(grid_indices)
            # for the rest, apply heuristic
            inverted_mask = np.logical_not(valid_mask)
            # TODO we might be able to optimize this step a bit more by using more numpy batch operations
            distances[inverted_mask] = map(self._get_heuristic_distance_local, local_points[inverted_mask, :3])
        else:
            distances = map(self._get_heuristic_distance_local, local_points[:, :3])
        return distances

    def get_distances_grad(self, positions):
        """
            Return the shortest distance to obstacles at the given points as well as the gradients.
            This method interpolates distances between grid points.
            TODO: If any query position is out of bounds, it currently throws an exception.
            ---------
            Arguments
            ---------
            positions, a numpy array of shape (n, 3) - n query positions in global frame
            -------
            Returns
            -------
            values, np array of shape (n,) - distances at the query positions
            gradients, np array of shape (n, 3) - gradients at the query positions
        """
        valid_mask, values, gradients = self._grid.get_cell_gradients_pos(positions)
        if np.sum(valid_mask) < values.shape[0]:
            raise NotImplementedError("Dealing with positions that are out of bounds is not implemented yet")
        return values, gradients

    def get_direction(self, point):
        """
            Return the direction to the closest non-penetrating point (this is approximate and up to the value of resolution wrong).
            ---------
            Arguments
            ---------
            points, numpy array of shape (3,) - query position
            -------
            Returns
            -------
            numpy array of shape (3,)
        """
        v = None
        _, idx = self._grid.map_to_grid(point)
        if idx is not None:
            v = self._grid.get_additional_data(idx)
        if v is None:  # means we are outside of collisions
            v = np.array([0.0, 0.0, 0.0])
        return v

    def has_directions(self):
        """
            Return whether this sdf supports direction queries.
        """
        return self._grid.has_additional_data()

    def clear_visualization(self):
        """
            Clear the visualization of this distance field
        """
        self._or_visualization.clear()

    def save(self, file_name):
        """
            Save this distance field to a file.
            Note that this function may create several files with different endings attached to file_name
            @param file_name - file to store sdf in
        """
        grid_file_name = file_name + '.grid'
        self._grid.save(grid_file_name)
        meta_file_name = file_name + '.meta'
        meta_data = np.array([self._approximation_box])
        np.save(meta_file_name, meta_data)

    @staticmethod
    def load(filename):
        """
            Loads an sdf from file.
            :return: the loaded sdf or None, if loading failed
        """
        meta_data_filename = filename + '.meta.npy'
        if not os.path.exists(meta_data_filename):
            rospy.logwarn("Could not load SDF because meta data file " + meta_data_filename + " does not exist")
            return None
        meta_data = np.load(meta_data_filename)
        approximation_box = meta_data[0]
        try:
            grid = VoxelGrid.load(filename + '.grid')
        except IOError as io_err:
            rospy.logwarn("Could not load SDF because:" + str(io_err))
            return None
        return SDF(grid=grid, approximation_box=approximation_box)

    def visualize(self, env, safe_distance=None):
        """
            Visualizes this sdf in the given openrave environment.
            @param env - OpenRAVE environment to visualize the SDF in.
            @param safe_distance (optional) - if provided, the visualization colors cells that are more than
                    safe_distance away from any obstacle in the same way as obstacles that are infinitely far away.
        """
        if not self._or_visualization or self._or_visualization._env != env:
            self._or_visualization = ORVoxelGridVisualization(env, self._grid)
            self._or_visualization.update(max_sat_value=safe_distance)
        else:
            self._or_visualization.update(max_sat_value=safe_distance)

    def set_approximation_box(self, box):
        """
            Sets an approximation box. If get_distance is queried for a point outside of the underlying grid,
            the distance of the query point to this approximation box is computed.
        """
        self._approximation_box = box

    def min(self):
        """
            Return minimal signed distance.
        """
        return self._grid.get_min_value()

    def max(self):
        """
            Return maximal signed distance.
        """
        return self._grid.get_max_value()

    def get_grid(self):
        """
            Return the underlying VoxelGrid.
            Use with caution!
        """
        return self._grid


class SDFBuilder(object):
    """
        An SDF builder builds a signed distance field for a given environment.
        If you intend to construct multiple SDFs with the same cell size, it is recommended to use a single
        SDFBuilder as this saves resource generation. It only checks collisions with enabled bodies.
    """

    def __init__(self, env, cell_size):
        """
            Creates a new SDFBuilder object.
            @param env - OpenRAVE environment this builder operates on.
            @param cell_size - The cell size of the signed distance field.
        """
        self._cell_size = cell_size
        self._occupancy_builder = OccupancyGridBuilder(env, cell_size)

    @staticmethod
    def compute_gradient_field(sdf):
        """
            Compute the gradient field of the given sdf.
            ---------
            Arguments
            ---------
            sdf, SDF to compute gradients for
            -------
            Returns
            -------
            gradf, VectorGrid
        """
        cell_size = sdf._grid.get_cell_size()
        raw_data = sdf._grid.get_raw_data()
        vector_field = VectorGrid(sdf._grid.get_aabb(), cell_size=cell_size,
                                  base_transform=sdf._grid.get_transform(), num_cells=np.array(raw_data.shape))
        vector_field.vectors[0], vector_field.vectors[1], vector_field.vectors[2] = np.gradient(raw_data, cell_size)
        return vector_field

    @staticmethod
    def compute_sdf(grid, b_compute_dirs=False):
        """
            Compute a signed distance field from the given occupancy grid.

            Arguments
            ---------
            grid - VoxelGrid that is an occupancy map. The cell type must be bool
            b_compute_dirs, bool - If true, the function also computes for each voxel
                a vector pointing to the closest collision free voxel. This vector is stored
                in the sdf as additional data.
            ---------
            distance grid - VoxelGrid with cells of type float. Each cell contains the signed
                distance to the closest obstacle surface point
        """
        # the grid is a binary collision map: 1 - collision, 0 - no collision
        # before calling skfmm we need to transform this map
        # raw_grid = grid.get_raw_data()
        # raw_grid *= -2.0
        # raw_grid += 1.0
        # min_value = grid.get_min_value()
        # if min_value > 0:  # there is no collision
        #     raw_grid[:, :, :] = float('Inf')
        # else:
        #     grid.set_raw_data(skfmm.distance(raw_grid, dx=grid.get_cell_size()))
        raw_grid = grid.get_raw_data()
        inverse_collision_map = np.invert(raw_grid)
        # compute the distances to surface of collision space
        outside_distances = scipy.ndimage.morphology.distance_transform_edt(inverse_collision_map,
                                                                            sampling=grid.get_cell_size())
        # compute the interior of the collision space (remove the surface, because that's where the distance shall be 0)
        interior_collision_map = scipy.ndimage.morphology.binary_erosion(raw_grid)
        inside_distances = scipy.ndimage.morphology.distance_transform_edt(interior_collision_map,
                                                                           sampling=grid.get_cell_size())
        # create sdf grid
        sdf_grid = VoxelGrid(grid.get_workspace(), cell_size=grid.get_cell_size(), dtype=np.float_,
                             b_additional_data=b_compute_dirs)
        # shift distances by half cell size to be conservative (i.e. 0 should be on the interface between free and non-free cells)
        sdf_grid.set_raw_data(outside_distances - inside_distances - grid.get_cell_size() / 2.0)
        # compute directions, if requested
        if b_compute_dirs:
            # we compute these also for the borders
            x_indices, y_indices, z_indices = scipy.ndimage.morphology.distance_transform_edt(raw_grid,
                                                                                              return_distances=False,
                                                                                              return_indices=True,
                                                                                              sampling=grid.get_cell_size())
            for cell in sdf_grid:
                idx = cell.get_idx()
                nearest_free_idx = (x_indices[idx], y_indices[idx], z_indices[idx])
                if idx != nearest_free_idx:
                    delta_idx = np.array((nearest_free_idx)) - np.array(idx)
                    dir_vec = delta_idx * grid.get_cell_size()
                    cell.set_additional_data(dir_vec)
        return sdf_grid

    def create_sdf(self, workspace_aabb, b_compute_dirs=False):
        """
            Creates a new sdf for the current state of the OpenRAVE environment provided on construction.
            The SDF is created in world frame of the environment. You can later change its transform.
            NOTE: If you do not intend to continue creating more SDFs using this builder, call clear() afterwards.
            @param workspace_aabb - bounding box of the sdf in form of [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        occupancy_grid = VoxelGrid(workspace_aabb, cell_size=self._cell_size, dtype=bool)
        # First compute binary collision map
        start_time = time.time()
        self._occupancy_builder.compute_grid(occupancy_grid)
        print ('Computation of collision binary map took %f s' % (time.time() - start_time))
        # next compute sdf
        start_time = time.time()
        distance_grid = SDFBuilder.compute_sdf(occupancy_grid, b_compute_dirs=b_compute_dirs)
        print ('Computation of sdf took %f s' % (time.time() - start_time))
        return SDF(grid=distance_grid)

    def clear(self):
        """
            Clear all used resources.
        """
        self._occupancy_builder.clear()

    @staticmethod
    def compute_sdf_size(aabb, approx_error, radius=0.0):
        """
            Computes the required size of an sdf for a movable kinbody such
            that at the boundary of the sdf the relative error in distance estimation to the body's
            surface is bounded by approx_error.
            - :aabb: OpenRAVE bounding box of the object
            - :approx_error: Positive floating point number in (0, 1] denoting the maximal relative error
            - :radius: (optional) a positive floating point number that is the radius of an inscribing ball
                    centered at the object center
            - :return: a bounding box in the shape [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        scaling_factor = (1.0 - (1.0 - approx_error) * radius / np.linalg.norm(aabb.extents())) / approx_error
        scaled_extents = scaling_factor * aabb.extents()
        upper_point = aabb.pos() + scaled_extents
        lower_point = aabb.pos() - scaled_extents
        return np.array([lower_point[0], lower_point[1], lower_point[2],
                         upper_point[0], upper_point[1], upper_point[2]])


class SceneSDF(object):
    """
        A scene sdf is a signed distance field for a motion planning scene that contains
        multiple movable kinbodies (including a robot) and a potentially empty set of static obstacles.
        A scene sdf creates separate sdfs for the static obstacles and the movable objects.
        When querying a scene sdf, the returned distance takes the current state of the environment,
        i.e. the current poses of all movable kinbodies into account. Kinbodies with more degrees of freedom
        are currently not supported, i.e. robots should always be excluded if they are expected to change their
        configuration.
    """

    def __init__(self, env, movable_body_names, excluded_bodies=None, sdf_paths=None, radii=None):
        """
            Constructor for SceneSDF. In order to actually use a SceneSDF you need to
            either call load_sdf or create_sdf.
            @param env - OpenRAVE environment to use
            @param movable_body_names - list of names of kinbodies that can move
            @param excluded_bodies - a list of kinbody names to exclude, i.e. completely ignore for instance a robot
            @param sdf_paths - optionally a dictionary that maps kinbody name to filename to
                               load an sdf from for that body
            @param radii - optionally a dictionary that maps kinbody name to a radius
                           of an inscribing ball
        """
        self._env = env
        self._movable_body_names = list(movable_body_names)
        if excluded_bodies is None:
            excluded_bodies = []
        self._ignored_body_names = list(excluded_bodies)
        self._static_sdf = None
        self._body_sdfs = {}
        self._sdf_paths = sdf_paths
        self._radii = radii

    def _enable_body(self, name, b_enabled):
        """
            Disable/Enable the body with the given name
        """
        body = self._env.GetKinBody(name)
        if body is None:
            raise ValueError("Could not retrieve body with name %s from OpenRAVE environment" % body)
        body.Enable(b_enabled)

    def _compute_sdf_size(self, aabb, approx_error, body_name):
        """
            Computes the required size of an sdf for a movable kinbody such
            that at the boundary of the sdf the relative error in distance estimation to the body's
            surface is bounded by approx_error.
        """
        radius = 0.0
        if self._radii is not None and body_name in self._radii:
            radius = self._radii[body_name]
        return SDFBuilder.compute_sdf_size(aabb, approx_error, radius)

    def create_sdf(self, workspace_bounds, static_resolution=0.02, moveable_resolution=0.02,
                   approx_error=0.1, b_compute_dirs=False):
        """
            Creates a new scene sdf. This process takes time!
            @param workspace_bounds - the volume of the environment this sdf should cover
            @param static_resolution - the resolution of the sdf for the static part of the world
            @param moveable_resolution - the resolution of sdfs for movable kinbodies
            @param approx_error - a relativ error between 0 and 1 that is allowed to occur
                                  at boundaries of movable kinbody sdfs
            @param b_compute_dirs, bool, If true also computes directions to closest collision-free cells.
        """
        # before we do anything, save which bodies are enabled
        body_enable_status = {}
        for body in self._env.GetBodies():
            body_enable_status[body.GetName()] = body.IsEnabled()
        # now first we build a sdf for the static obstacles
        builder = SDFBuilder(self._env, static_resolution)
        for body_name in self._ignored_body_names:
            self._enable_body(body_name, False)
        # it only makes sense to build a static sdf, if we have static obstacles
        if len(self._ignored_body_names) + len(self._movable_body_names) < len(self._env.GetBodies()):
            # for that disable all movable objects
            for body_name in self._movable_body_names:
                self._enable_body(body_name, False)
            self._static_sdf = builder.create_sdf(workspace_bounds, b_compute_dirs)
        # Now we build SDFs for all movable object
        # if we have different resolutions for static and movables, we need a new builder
        if static_resolution != moveable_resolution:
            builder.clear()
            builder = SDFBuilder(self._env, moveable_resolution)
        # first disable everything in the scene
        for body in self._env.GetBodies():
            body.Enable(False)
        # Next create for each movable body an individual sdf
        for body_name in self._movable_body_names:
            body_sdf = None
            body = self._env.GetKinBody(body_name)
            if self._sdf_paths is not None and body_name in self._sdf_paths:  # we have a path for a body sdf
                # load an sdf
                body_sdf = SDF.load(self._sdf_paths[body_name])
                if b_compute_dirs and not body_sdf.has_directions():
                    body_sdf = None  # cannot use body sdf because it does not provide directions
            # if we do not have an sdf to load for this body or failed at doing so, create a new
            if body_sdf is None:
                # Prepare sdf creation for this body
                body.Enable(True)
                old_tf = body.GetTransform()
                body.SetTransform(np.eye(4))  # set it to the origin
                aabb = body.ComputeAABB()
                body_bounds = np.zeros(6)
                body_bounds[:3] = aabb.pos() - aabb.extents()
                body_bounds[3:] = aabb.pos() + aabb.extents()
                # Compute the size of the sdf that we need to ensure the maximum relative error
                sdf_bounds = self._compute_sdf_size(aabb, approx_error, body_name)
                # create the sdf
                body_sdf = builder.create_sdf(sdf_bounds, b_compute_dirs=b_compute_dirs)
                body_sdf.set_approximation_box(body_bounds)  # set the actual body aabb as approx box
                body.SetTransform(old_tf)  # restore transform
                body.Enable(False)  # disable body again
                # finally, if we have a body path for this body, store it
                if self._sdf_paths is None:
                    self._sdf_paths = {}
                if body_name in self._sdf_paths:
                    body_sdf.save(self._sdf_paths[body_name])
            self._body_sdfs[body_name] = (body, body_sdf)
        builder.clear()
        # Finally enable all bodies
        for body in self._env.GetBodies():
            body.Enable(body_enable_status[body.GetName()])

    def get_distance(self, position):
        """
            Returns the signed distance from the specified position to the closest obstacle surface
        """
        min_distance = float('inf')
        for (body, body_sdf) in self._body_sdfs.itervalues():
            body_sdf.set_transform(body.GetTransform())
            min_distance = min(min_distance, body_sdf.get_distance(position))
        if self._static_sdf is not None:
            min_distance = min(self._static_sdf.get_distance(position), min_distance)
        return min_distance

    def get_distances(self, positions):
        """
            Returns the signed distance from the given positions to the closest obstacle surface
            @param positions - a numpy matrix of shape (n, 3) where n is the number of query positions.
        """
        min_distances = np.full(positions.shape[0], float('inf'))
        for (body, body_sdf) in self._body_sdfs.itervalues():
            body_sdf.set_transform(body.GetTransform())
            min_distances = np.minimum(min_distances, body_sdf.get_distances(positions))
        if self._static_sdf is not None:
            min_distances = np.minimum(min_distances, self._static_sdf.get_distances(positions))
        return min_distances

    def get_distances_grad(self, positions):
        """
            Returns the signed distance from the given position to the closest obstacle surface
            as well as the gradient of that distance.
            TODO: raises NotImplementedError if any position is out of bounds - this also holds for movables!
            ---------
            Arguments
            ---------
            positions, np array of shape (n, 3) - n query positions
            -------
            Returns
            -------
            distances, np array of shape (n,) - distances
            gradients, np.array of shape (n, 3) - gradients
        """
        min_distances = np.full(positions.shape[0], float('inf'))
        gradients = np.zeros((positions.shape[0], 3))
        for (body, body_sdf) in self._body_sdfs.itervalues():
            body_sdf.set_transform(body.GetTransform())
            distances, bgradients = body_sdf.get_distances_grad(positions)
            selection_mask = distances < min_distances
            gradients[selection_mask] = bgradients[selection_mask]
            min_distances = np.min((distances, min_distances), axis=0)
        if self._static_sdf is not None:
            distances, sgradients = self._static_sdf.get_distances_grad(positions)
            selection_mask = distances < min_distances
            gradients[selection_mask] = sgradients[selection_mask]
            min_distances = np.min((distances, min_distances), axis=0)
        return min_distances, gradients

    def get_direction(self, position):
        """
            Return the approximate direction to the closest obstacle free point from the give position.
            ---------
            Arguments
            ---------
            position, numpy array of shape (3,)
            -------
            Returns
            -------
            dir, numpy array of shape (3,)
        """
        min_distance = float('inf')
        v = np.array([0.0, 0.0, 0.0])
        for (body, body_sdf) in self._body_sdfs.itervalues():
            body_sdf.set_transform(body.GetTransform())
            dist, tv = body_sdf.get_distance(position, b_return_dist=True)
            if dist < min_distance:
                min_distance = dist
                v = tv
        if self._static_sdf is not None:
            dist, tv = self._static_sdf.get_distance(position, b_return_dir=True)
            if dist < min_distance:
                v = tv
        return v

    def save(self, filename, body_dir=None):
        """
            Saves this scene sdf under the specified path.
            @param filename - absolute filename to save this sdf in (must be a path to a file)
            @param body_dir - optionally a relative path w.r.t dir(filename) to save the body sdfs in
                              if not provided, the body sdfs are saved in the same directory as filename points to
        """
        base_name = os.path.basename(filename)
        if not base_name:
            raise ValueError("The provided filename %s is invalid. The filename must be a valid path to a file" % filename)
        dir_name = os.path.dirname(filename)
        if body_dir is None:
            body_dir = '.'
        if self._sdf_paths is None:
            self._sdf_paths = {}
        # now build a dictionary mapping body name to sdf (static for static sdf) and save sdfs
        sdf_paths = {}
        rel_paths = {}
        if self._static_sdf:
            static_file_name = base_name + '.static.sdf'
            static_file_path = dir_name + '/' + static_file_name
            # TODO We could have a name collision here, if the environment contains a kinbody called static
            sdf_paths['__static_sdf__'] = static_file_path
            rel_paths['__static_sdf__'] = './' + static_file_name
            if '__static_sdf__' not in self._sdf_paths:
                self._static_sdf.save(static_file_path)
        for (key, value) in self._sdf_paths:  # reuse the filenames we loaded things from
            sdf_paths[key] = value
        for (body, body_sdf) in self._body_sdfs.itervalues():
            if body.GetName() in sdf_paths:  # no need to save body sdfs we loaded
                continue
            # we need to save the body sdfs for which we don't have an sdf path
            body_sdf_filename = str(body.GetName()) + '.sdf'
            body_sdf_filepath = dir_name + '/' + body_dir + '/' + body_sdf_filename
            sdf_paths[body.GetName()] = body_sdf_filepath
            body_sdf.save(body_sdf_filepath)
            rel_paths[str(body.GetName())] = body_dir + '/' + body_sdf_filename
        with open(filename, 'w') as meta_file:
            yaml.dump(rel_paths, meta_file)
        self._sdf_paths = sdf_paths

    def load(self, filename):
        """
            Loads a scene sdf from the specified path.
        """
        dir_name = os.path.dirname(filename)
        with open(filename, 'r') as meta_file:
            self._body_sdfs = {}
            self._static_sdf = None
            # first read in realtive paths and make them absolute
            rel_paths = yaml.load(meta_file)
            self._sdf_paths = {}
            for (name, rel_path) in rel_paths.iteritems():
                self._sdf_paths[name] = dir_name + '/' + rel_path
            # next read in sdfs
            available_sdfs = {}
            for (name, path) in self._sdf_paths.iteritems():
                if name != '__static_sdf__':
                    body = self._env.GetKinBody(name)
                    if body is None:
                        rospy.logerr("Could not find kinbody %s" % name)
                        continue
                    self._body_sdfs[name] = (body, SDF.load(path))
                    available_sdfs[name] = True
                else:
                    self._static_sdf = SDF.load(path)
                    available_sdfs['__static_sdf__'] = True
            # verify we have all movable bodies
            for name in self._movable_body_names:
                if name not in available_sdfs:
                    raise IOError("Could not load sdf for kinbody %s" % name)
            if '__static_sdf__' not in available_sdfs and len(self._movable_body_names) < len(self._env.GetBodies()):
                raise IOError("Could not load sdf for static environment")

    def has_directions(self):
        """
            Return whether this sdf supports direction queries.
        """
        if not self._static_sdf.has_directions():
            return False
        for body_sdf in self._body_sdfs:
            if not body_sdf.has_direction():
                return False
        return True


if __name__ == "__main__":
    import os
    import IPython
    import mayavi.mlab
    base_path = os.path.dirname(__file__) + '/../../../'
    sdf_file = base_path + 'data/sdfs/placement_exp_2.static.sdf'
    sdf = SDF.load(sdf_file)
    occ_file = base_path + 'data/occupancy_grids/placement_exp_2'
    occ = VoxelGrid.load(occ_file)
    xx, yy, zz = sdf._grid.get_grid_positions()
    mayavi.mlab.contour3d(xx, yy, zz, sdf._grid._cells[1:-1, 1:-1, 1:-1],
                          contours=[0.0], color=(0.9, 0.0, 0.0))
    occ_indices = np.nonzero(occ._cells[1:-1, 1:-1, 1:-1])
    x = xx[occ_indices].flatten()
    y = yy[occ_indices].flatten()
    z = zz[occ_indices].flatten()
    mayavi.mlab.points3d(x, y, z, x.shape[0] * [occ.get_cell_size()],
                         mode="sphere", scale_factor=1, mask_points=1, transparent=True, opacity=0.8)
    mayavi.mlab.show()
    # IPython.embed()
