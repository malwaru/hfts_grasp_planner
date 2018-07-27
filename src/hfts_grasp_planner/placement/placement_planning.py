#!/usr/bin/env python
import logging
import itertools
import collections
import hfts_grasp_planner.placement.optimization as optimization
import hfts_grasp_planner.external.transformations as transformations
import hfts_grasp_planner.placement.so3hierarchy as so3hierarchy
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf
import hfts_grasp_planner.utils as utils
import numpy as np
import scipy.spatial
import math
import os
import openravepy as orpy


class SimplePlacementQuality(object):
    """
        Implements a simple quality function for a placement.
        This quality function is based on quasi-static analysis of contact stability.
        Assumes that gravity is pulling in direction -z = (0, 0, -1).
    """
    class OrientationFilter(object):
        """
            A filter for placement faces based on their orientation.
            ---------
            Arguments
            ---------
            normal, numpy array of shape (3,) - the allowed normal
            max_angle, float - the maximal allowed angle between placement plane's normal and the normal
                of this filter.
        """
        def __init__(self, normal, max_angle):
            self._normal = normal
            self._max_angle = max_angle

        def accept_plane(self, plane):
            """
                Return whether to accept the given plane or not.
                ---------
                Arguments
                ---------
                plane, numpy array of shape (n, 3), where the first element is the normal of the plane
                    and all others the points on it
            """
            dot_product = np.clip(np.dot(self._normal, plane[0]), -1.0, 1.0)
            return np.arccos(dot_product) < self._max_angle

    class PreferenceFilterIO(object):
        """
            Implements an IO interface to load preference filters from files.
        """
        def __init__(self, base_dir):
            """
                Create a new IO filter that looks for preference filters in the given base_dir.
                Currently only supports OrientationFilter
                ---------
                Arguments
                ---------
                base_dir, string - directory to look for preference filters in.
            """
            self._base_dir = base_dir
            self._filters = {}

        def get_filters(self, model_name):
            """
                Return preference filters for the given model.
                Return None, if no filters available.
            """
            if model_name in self._filters:
                return self._filters[model_name]
            if os.path.exists(self._base_dir + '/' + model_name):
                filename = self._base_dir + '/' + model_name + '/' + 'placement_filters.npy'
                if os.path.exists(filename):
                    new_filters = []
                    filter_descs = np.load(filename)  # assumes array (n, 4), where n is the number of filters
                    if len(filter_descs.shape) != 2 or filter_descs.shape[1] != 4:
                        raise IOError("Could not load filter for model %s. Invalid numpy array shape encountered." % model_name)
                    for filter_desc in filter_descs:  # filter_desc is assumed to be (nx, ny, nz, angle)
                        new_filters.append(SimplePlacementQuality.OrientationFilter(filter_desc[:3], filter_desc[3]))
                    self._filters[model_name] = new_filters
                    return new_filters
            return None

    def __init__(self, env, filter_io, parameters=None):
        """
            Create a new SimplePlacementQuality function object.
            ---------
            Arguments
            ---------
            env - fully initialized OpenRAVE environment. Note that the environment is cloned.
            filter_io - Object of class PreferenceFilterIO that allows reading
                in object-specific placement plane filters.
            parameters - dictionary with parameters. Available parameters are:
                min_com_distance, float - minimal distance for center of mass from convex hull boundary TODO better description
                min_normal_similarity, float - minimal value for dot product between similar faces TODO better description
                falling_height_tolerance, float - tolerance for computing virtual contact points
        """
        self._filter_io = filter_io
        self._env = env.CloneSelf(orpy.CloningOptions.Bodies)
        col_checker = orpy.RaveCreateCollisionChecker(self._env, 'ode')  # Bullet is not working properly
        self._env.SetCollisionChecker(col_checker)
        self._env.SetViewer('qtcoin')  # for debug only
        self._body = None
        self._placement_planes = None
        self._dir_gravity = np.array([0.0, 0.0, -1.0])
        self._local_com = None
        self._parameters = {"min_com_distance": 0.001, "min_normal_similarity": 0.97,
                            "falling_height_tolerance": 0.005, "slopiness_weight": 1.6 / math.pi}
        self._max_ray_length = 2.0
        if parameters:
            for (key, value) in parameters:
                self._parameters[key] = value

    def compute_quality(self, pose):
        """
            Compute the placement quality for the given pose. 
            The larger the returned value, the more suitable a pose is for placement.
            Note that this function is merely a heuristic.
            ---------
            Arguments
            ---------
            pose, numpy array of shape (4, 4) - describes the pose of the currently
                set kinbody
            ---------
            Returns
            ---------
            score, float - the larger the better the pose is
        """
        self._body.SetTransform(pose)
        best_score = float('inf')
        # TODO can we skip some placement planes? based on their normals?
        for plane in self._placement_planes:
            virtual_contacts, _, virtual_plane_axes, distances = self._compute_virtual_contact_plane(plane, pose)
            if virtual_contacts.size > 0:
                # we have some virtual contacts and thus finite distances
                # rerieve the largest distance between a placement point and its virtual contact
                finite_distances = [d for d in distances if d < float('inf')]
                d_head = np.max(finite_distances)
                # d_head = np.min(finite_distances)
                if virtual_plane_axes is not None:
                    # we have a virtual contact plane and can compute the full heuristic
                    # project the virtual contacts to the x, y plane and compute their convex hull
                    contact_footprint = scipy.spatial.ConvexHull(virtual_contacts[:, :2])
                    # retrieve the distance to the closest edge
                    chull_distance = self._compute_hull_distance(contact_footprint, self._body.GetCenterOfMass()[:2])
                    # angle between virtual placement plane and z-axis
                    dot_product = np.dot(virtual_plane_axes[:, 2], np.abs(self._dir_gravity))
                    alpha = np.arccos(np.clip(dot_product, -1.0, 1.0))
                    # angle between the placement plane and the virtual placement plane. Our stability estimate
                    # is only somehow accurate if these two align. It's completely nonsense if the plane is upside down
                    dot_product = np.dot(virtual_plane_axes[:, 2], -1.0 * np.dot(pose[:3, :3], plane[0]))
                    gamma = np.arccos(np.clip(dot_product, -1.0, 1.0))
                    # thus penalize invalid angles through the following score
                    # TODO maybe choose a different formula for this weight here
                    weight = 1.0 / (1.0 + np.exp(-8.0 / np.pi * (gamma - np.pi / 2.0)))
                    placement_value = (1.0 - weight) * chull_distance + weight * self._max_ray_length
                else:
                    # we have contacts, but insufficiently many to fit a plane, so give it bad
                    # scores for anything related to the virtual placement plane
                    placement_value = self._max_ray_length
                    alpha = 1.57
            else:
                # we have no contacts, so there isn't anything we can say
                d_head = float('inf')
                placement_value = float('inf')
                alpha = float('inf')
            # we want to minimize the following score
            # minimizing d_head drives the placement plane downwards and aligns the placement plane with the surface
            # minimizing the placement_value leads to rotating the object (fixes upside down planes) so that the
            # placement plane gets aligned with the surface, or if the plane is already aligned, maximizes the
            # stability by minizing the chull_distance (which is negative inside of the hull).
            # Minimizing alpha prefers placements on planar surfaces rather than on slopes.
            score = d_head + placement_value + self._parameters["slopiness_weight"] * alpha
            best_score = min(score, best_score)
        return -1.0 * best_score  # externally we want to maximize

    def set_target_object(self, body, model_name=None):
        """
            Set the target object. 
            Synchronizes the underlying OpenRAVE environment with the environment of this body.
            If the environment of body has more kinbodies than the underlying environment, 
            a RuntimeError is thrown. This is due to a bug in OpenRAVE, that doesn't allow
            us to load additional kinbodies into any environment after a viewer has been set.
            ---------
            Arguments
            ---------
            body, OpenRAVE Kinbody - target object
            model_name (string, optional) - name of the kinbody model if it is different from the kinbody's name
        """
        self._body = self._env.GetKinBody(body.GetName())
        if not self._body:
            raise RuntimeError("Could not find body " + body.GetName() + " in cloned environment.")
        self._synch_env(body.GetEnv())
        if not model_name:
            model_name = body.GetName()
        user_filters = self._filter_io.get_filters(model_name)
        tf = self._body.GetTransform()
        tf_inv = utils.inverse_transform(tf)
        self._local_com = np.dot(tf_inv[:3, :3], self._body.GetCenterOfMass()) + tf_inv[:3, 3]
        self._compute_placement_planes(user_filters)
        object_diameter = 2.0 * np.linalg.norm(self._body.ComputeAABB().extents())
        self._max_ray_length = self._workspace_volume[1][2] - self._workspace_volume[0][2] \
            + 2.0 * object_diameter

    def set_placement_volume(self, workspace_volume):
        """
            Set the placement volume.
            @param workspace_volume - (min_point, max_point), where both are np.arrays of length 3
        """
        self._workspace_volume = np.array(workspace_volume)

    # TODO move this function to utils or so
    @staticmethod
    def merge_faces(chull, min_normal_similarity=0.97):
        """
            Merge adjacent faces of the convex hull if they have similar normals.
            ---------
            Arguments
            ---------
            chull - convex hull computed by scipy.spatial.ConvexHull
            min_normal_similarity - minimal value that the pairwise dot product of the normals
                of two faces need to have to be merged.
            ---------
            Returns
            ---------
            A tuple (clusters, face_clusters, num_clusters):
                - clusters, a list of tuples (normal, vertices) - describes all clusters.
                    normal is a numpy array of length 3 describing the normal of a cluster (merged faces)
                    vertices is the set of vertex indices, which compose the cluster
                - face_cluster - numpy array of length num_clusters, which stores for each original face
                    of chull which cluster it belongs to
                - num_clusters, int - the total number of clusters
        """
        clusters = []
        face_clusters = np.ones(chull.simplices.shape[0], dtype=int)
        face_clusters *= -1
        face_idx = 0
        cluster_id = 0
        cluster_candidates = collections.deque()
        # run over all faces
        while face_idx < chull.simplices.shape[0]:
            if face_clusters[face_idx] == -1:  # we have not assigned this face to a cluster yet
                cluster_normal = chull.equations[face_idx, :3]
                current_cluster = (cluster_normal, set(chull.simplices[face_idx]))  # create a new cluster
                clusters.append(current_cluster)
                face_clusters[face_idx] = cluster_id  # assign this face to the new cluster
                # grow cluster to include adjacent unclustered faces that have a similar normal
                cluster_candidates.extend(chull.neighbors[face_idx])
                while cluster_candidates:
                    candidate_idx = cluster_candidates.popleft()
                    # check if this face belongs to the same cluster, it does so, if the normal is almost the same
                    if np.dot(chull.equations[candidate_idx, :3], cluster_normal) >= min_normal_similarity:
                        if face_clusters[candidate_idx] == -1:  # check if candidate is unclustered yet
                            # add the vertices of this face to the cluster
                            current_cluster[1].update(chull.simplices[candidate_idx])
                            face_clusters[candidate_idx] = cluster_id  # assign the cluster to the face
                            # add its neighbors to our extension candidates
                            cluster_candidates.extend(chull.neighbors[candidate_idx])
                cluster_id += 1
            face_idx += 1
        return clusters, face_clusters, cluster_id

    # TODO move this to some utils
    @staticmethod
    def _compute_hull_distance(convex_hull, point):
        """
            Compute the signed distance of the given point
            to the closest edge of the given 2d convex hull.
        """
        # min_distance = 0.0
        assert(convex_hull.points[0].shape == point.shape)
        assert(point.shape == (2,))
        interior_distance = float('-inf')
        exterior_distance = float('inf')
        for idx, edge in enumerate(convex_hull.simplices):
            rel_point = point - convex_hull.points[edge[0]]
            # there are 3 cases: the point is closest to edge[0], to edge[1] or to
            # its orthogonal projection onto the edge
            edge_dir = convex_hull.points[edge[1]] - convex_hull.points[edge[0]]
            edge_length = np.linalg.norm(edge_dir)
            assert(edge_length > 0.0)
            edge_dir /= edge_length
            # in any case we need the orthogonal distance to compute the sign
            orthogonal_distance = np.dot(convex_hull.equations[idx, :2], rel_point)
            # now check the different cases
            directional_distance = np.dot(edge_dir, rel_point)
            if directional_distance >= edge_length:
                # closest distance is to edge[1]
                edge_distance = np.linalg.norm(point - convex_hull.points[edge[1]])
            elif directional_distance <= 0.0:
                # closest distance is to edge[0]
                edge_distance = np.linalg.norm(point - convex_hull.points[edge[0]])
            else:
                edge_distance = np.abs(orthogonal_distance)
            if orthogonal_distance < 0.0:  # point is inside w.r.t to this edge
                interior_distance = max(-1.0 * edge_distance, interior_distance)
            else:
                exterior_distance = min(edge_distance, exterior_distance)
        if exterior_distance == float('inf'):  # the point is inside the convex hull
            return interior_distance
        return exterior_distance  # the point is outside the convex hull

    def _is_stable_placement_plane(self, plane):
        """
            Return whether the specified plane can be used to stably place the
            current kinbody.
            ---------
            Arguments
            ---------
            plane, numpy array of shape (N + 1, 3) - the placement plane to test
            -------
            Returns
            -------
            true or false depending on whether it is stable or not
        """
        handles = self._visualize_placement_plane(plane) # TODO remove
        # due to our tolerance value when clustering faces, the points may not be co-planar.
        # hence, project them
        mean_point = np.mean(plane[1:], axis=0)
        rel_points = plane[1:] - mean_point
        projected_points = rel_points - np.dot(rel_points, plane[0].transpose())[:, np.newaxis] * plane[0]
        axes = np.empty((3, 2))
        axes[:, 0] = projected_points[1] - projected_points[0]
        axes[:, 0] /= np.linalg.norm(axes[:, 0])
        axes[:, 1] = np.cross(plane[0], axes[:, 0])
        points_2d = np.dot(projected_points, axes)
        # project the center of mass also on this plane
        projected_com = self._local_com - mean_point - \
            np.dot(self._local_com - mean_point, plane[0].transpose()) * plane[0]
        com2d = np.dot(projected_com, axes)
        # compute the convex hull of the projected points
        convex_hull = scipy.spatial.ConvexHull(points_2d)
        # ##### DRAW CONVEX HULL ###### TODO remove
        boundary = points_2d[convex_hull.vertices]
        # compute 3d boundary from bases and mean point
        boundary3d = boundary[:, 0, np.newaxis] * axes[:, 0] + boundary[:, 1, np.newaxis] * axes[:, 1] + mean_point
        # transform it to world frame
        tf = self._body.GetTransform()
        boundary3d = np.dot(boundary3d, tf[:3, :3]) + tf[:3, 3]
        handles.append(self._visualize_boundary(boundary3d))
        ##### DRAW CONVEX HULL - END ######
        ##### DRAW PROJECTED COM ######
        handles.append(self._env.drawbox(projected_com + mean_point, np.array([0.005, 0.005, 0.005]),
                                         np.array([0.29, 0, 0.5]), tf))
        # ##### DRAW PROJECTED COM - END ######
        # accept the point if the projected center of mass is inside of the convex hull
        return self._compute_hull_distance(convex_hull, com2d) < -1.0 * self._parameters["min_com_distance"]

    def _compute_placement_planes(self, user_filters):
        # first compute the convex hull of the body
        links = self._body.GetLinks()
        assert(len(links) == 1)  # the object is assumed to be a rigid body
        meshes = [geom.GetCollisionMesh() for geom in links[0].GetGeometries()]
        all_points_shape = (sum([mesh.vertices.shape[0] for mesh in meshes]), 3)
        vertices = np.empty(all_points_shape)
        offset = 0
        for mesh in meshes:
            vertices[offset:mesh.vertices.shape[0] + offset] = mesh.vertices
            offset += mesh.vertices.shape[0]
        convex_hull = scipy.spatial.ConvexHull(vertices)  # TODO do we need to add any flags?
        # merge faces
        clusters, face_clusters, _ = SimplePlacementQuality.merge_faces(convex_hull,
                                                                        self._parameters["min_normal_similarity"])
        # handles = self._visualize_clusters(convex_hull, clusters, face_clusters)
        # handles = None
        self._placement_planes = []
        # retrieve clusters to store placement faces
        for (normal, vertices) in clusters:
            plane = np.empty((len(vertices) + 1, 3), dtype=float)
            plane[0] = normal
            plane[1:] = convex_hull.points[list(vertices)]
            # filter faces based on object specific filters 
            if user_filters:
                acceptances = np.array([uf.accept_plane(plane) for uf in user_filters])
                if not acceptances.any():
                    continue
            # filter based on stability
            if not self._is_stable_placement_plane(plane):
                continue
            # if this plane passed all filters, we accept it
            self._placement_planes.append(plane)
        if not self._placement_planes:
            raise ValueError("Failed to compute placement planes for body %s." % self._body.GetName())
        handles = self._visualize_placement_planes()

    def _synch_env(self, env):
        with env:
            with self._env:
                bodies = env.GetBodies()
                for body in bodies:
                    my_body = self._env.GetKinBody(body.GetName())
                    if not my_body:
                        raise RuntimeError("Could not find body with name " + body.GetName() + " in cloned environment")
                    my_body.SetTransform(body.GetTransform())

    def _compute_virtual_contact_plane(self, placement_plane, pose):
        """
            Compute the virtual contact points for the given plane if the kinbody is located
            at the current pose.
            ---------
            Arguments
            ---------
            placement_plane, numpy array of shape (n+1, 3), where n is the number of points
                on the plane and placement_plane[0, :] is its normal.
            pose, numpy array of shape (4, 4), current pose of the body
            ---------
            Returns
            ---------
            virtual_contact_points, numpy arrays of shape (k, 3), where k is the number of
                first impact points. First impact points are the three points of the placement_plane
                that have the minimal distance along the direction of gravity towards the surface.
                If there are more virtual contact points that have a similar distance (how similar is determined by
                the parameter self._parameters["first_impact_tolerance"]) as the three impact points,
                these are also included in this array (thus k >= 3). The number of first impact points
                might, however, also be smaller than 3, if there is no surface below the object (or the surface
                is more than self._max_ray_length below the object.)
            vidx, numpy array of shape (k,). Stores for each virtual_contact_point what index in placement_plane it
                originates from.
            virtual_plane_axes, numpy array of shape (3, 3). The first two columns span a plane fitted into
                the virtual contact points. The third column is the normal of this plane (only defined if k >= 3).
            distances, numpy array of shape (n,) where n is the total number of point in the placement plane.
                Note that some of these distances may be infinity, if there is no surface within
                self._max_ray_length below this body. Distances are in the order of placement_plane points.
        """
        # first transform the placement plane to global frame
        tf_plane = np.dot(placement_plane, pose[:3, :3].transpose())
        tf_plane[1:] += pose[:3, 3]
        assert(tf_plane.shape[0] >= 4)
        # perform ray tracing to compute projected contact points
        rays = np.zeros((tf_plane.shape[0] - 1, 6))
        rays[:, 3:] = self._max_ray_length * self._dir_gravity
        rays[:, :3] = tf_plane[1:]
        self._body.Enable(False)
        collisions, virtual_contacts = self._env.CheckCollisionRays(rays)
        self._body.Enable(True)
        ##### DRAW CONTACT ARROWS ###### TODO remove
        handles = []
        for idx, contact in enumerate(virtual_contacts):
            if collisions[idx]:
                handles.append(self._env.drawarrow(tf_plane[idx + 1], contact[:3], linewidth=0.002))
        ##### DRAW CONTACT ARROWS - END ######
        distances = np.linalg.norm(tf_plane[1:] - virtual_contacts[:, :3], axis=1)
        distances[np.invert(collisions)] = np.inf
        # compute virtual contact plane
        ## first, sort virtual contact points by falling distance
        contact_distance_tuples = zip(distances[collisions], np.where(collisions)[0])
        ## if we have less than three contacts, there is nothing we can do
        if len(contact_distance_tuples) < 3:
            return virtual_contacts[collisions, :3], np.where(collisions)[0], None, distances
        ## do the actual sorting
        contact_distance_tuples.sort(key=lambda x: x[0])
        sorted_indices = [cdt[1] for cdt in contact_distance_tuples]
        ### second, select the minimal number of contact points such that they span a plane in x, y
        points_in_line = True
        for idx in xrange(2, len(contact_distance_tuples)):
            top_virtual_contacts = virtual_contacts[sorted_indices[:idx + 1], :3]
            _, s, _ = np.linalg.svd(top_virtual_contacts[:, :2] - np.mean(top_virtual_contacts[:, :2], axis=0))
            std_dev = s[1] / np.sqrt(idx) 
            if std_dev > 5e-3:  # std deviation should not be larger than some value
                points_in_line = False
                break
        if points_in_line:
            # TODO should we return a special flag if all virtual contacts are in a line?
            return virtual_contacts[collisions, :3], np.where(collisions)[0], None, distances
        #### Getting here means not all points lie in a line, so we can actually fit a plane
        #### Add some additional points if they are close
        max_falling_height = contact_distance_tuples[idx][0] + self._parameters["falling_height_tolerance"]
        vidx = np.where(distances <= max_falling_height)[0]
        top_virtual_contacts = virtual_contacts[vidx]
        #### third, fit a plane into these virtual contact points
        mean_point = np.mean(top_virtual_contacts[:, :3], axis=0)
        top_virtual_contacts[:, :3] -= mean_point
        _, s, v = np.linalg.svd(top_virtual_contacts[:, :3])
        # flip the normal of the virtual placement plane such that it points upwards
        if v[2, 2] < 0.0:
            v[2, :] *= -1.0
        ##### DRAW CONTACT PLANE ###### TODO remove
        handles.append(self._env.drawarrow(mean_point, mean_point + 0.1 * v[0, :], linewidth=0.002, color=[1.0, 0, 0, 1.0]))
        handles.append(self._env.drawarrow(mean_point, mean_point + 0.1 * v[1, :], linewidth=0.002, color=[0.0, 1.0, 0.0, 1.0]))
        handles.append(self._env.drawarrow(mean_point, mean_point + 0.1 * v[2, :], linewidth=0.002, color=[0.0, 0.0, 1.0, 1.0]))
        ##### DRAW CONTACT ARROWS - END ######
        return top_virtual_contacts[:, :3] + mean_point, vidx, v.transpose(), distances

    ###################### Visualization functions #####################
    def _visualize_clusters(self, convex_hull, clusters, face_clusters, arrow_length=0.01, arrow_width=0.002):
        handles = []
        tf = self._body.GetTransform()
        world_points = np.dot(convex_hull.points, tf[:3, :3]. transpose())
        world_points += tf[:3, 3]
        # for plane in self._placement_planes:
        for cluster_idx in range(len(clusters)):
            color = np.random.random(3)
            faces = face_clusters == cluster_idx
            # render faces
            handles.append(self._env.drawtrimesh(world_points, convex_hull.simplices[faces], color))
            # for pidx in range(1, len(tf_plane)):
            #     handles.append(self._env.drawarrow(tf_plane[pidx], tf_plane[pidx] + arrow_length * tf_plane[0], arrow_width, color))
        return handles

    def _visualize_boundary(self, boundary, linewidth=0.002):
        assert(len(boundary.shape) == 2 and boundary.shape[1] == 3)
        if boundary.shape[0] < 2:
            return None
        linelist_input = np.empty((2 * boundary.shape[0], 3))
        linelist_input[0] = boundary[0]
        lidx = 0
        for point in boundary[1:]:
            linelist_input[2 * lidx + 1] = point
            linelist_input[2 * lidx + 2] = point
            lidx += 1
        linelist_input[-1] = boundary[0]
        return self._env.drawlinelist(linelist_input, linewidth=linewidth)

    def _visualize_placement_plane(self, plane, arrow_length=0.01, arrow_width=0.002):
        handles = []
        tf = self._body.GetTransform()
        tf_plane = np.dot(plane, tf[:3, :3].transpose())
        tf_plane[1:] += tf[:3, 3]
        color = np.random.random(3)
        for pidx in range(1, len(tf_plane)):
            handles.append(self._env.drawarrow(tf_plane[pidx], tf_plane[pidx] + arrow_length * tf_plane[0],
                                               arrow_width, color))
        # handles.append(self._visualize_boundary(tf_plane[1:]))
        return handles

    def _visualize_placement_planes(self, arrow_length=0.01, arrow_width=0.002):
        handles = []
        for plane in self._placement_planes:
            handles.extend(self._visualize_placement_plane(plane, arrow_length, arrow_width))
        return handles


class PlacementObjectiveFn(object):
    """
        Implements an objective function for object placement.
        #TODO: should this class be in a different module?
    """
    def __init__(self, env, scene_sdf, object_base_path, vol_approx=0.005):
        """
            Creates a new PlacementObjectiveFn
            ---------
            Arguments
            ---------
            env, OpenRAVE environment
            scene_sdf, SceneSDF of the environment
            object_base_path, string - path to object files
        """
        self._env = env
        self._obj_name = None
        self._model_name = None
        self._kinbody = None
        self._scene_sdf = scene_sdf
        self._kinbody_octree = None
        filter_io = SimplePlacementQuality.PreferenceFilterIO(object_base_path)
        self._stability_function = SimplePlacementQuality(env, filter_io)  # TODO make settable as parameter
        self._vol_approx = vol_approx

    def set_target_object(self, obj_name, model_name=None):
        """
            Sets the target object to be obj_name.
            Model name denotes the class of object. Assumed to be the same as obj_name if None
        """
        self._obj_name = obj_name
        self._model_name = model_name if model_name is not None else obj_name
        self._kinbody = self._env.GetKinBody(self._obj_name)
        # TODO we could/should disable the object within the scene_sdf if it is in it
        if not self._kinbody:
            raise ValueError("Could not set target object " + obj_name + " because it does not exist")
        self._kinbody_octree = kinbody_sdf.OccupancyOctree(self._vol_approx, self._kinbody)
        self._stability_function.set_target_object(self._kinbody, model_name)

    def set_placement_volume(self, workspace_volume):
        """
            Sets the placement volume.
            @param workspace_volume - (min_point, max_point), where both are np.arrays of length 3
        """
        self._stability_function.set_placement_volume(workspace_volume)

    def get_target_object(self):
        """
            Returns the currently set target object name.
            May be None, if none set
        """
        return self._obj_name

    def evaluate(self, node):
        """
            Evalutes the objective function for the given node (must be SE3HierarchyNode)
        """
        assert(self._kinbody_octree)
        assert(self._scene_sdf)
        # set the transform to the pose to evaluate
        representative = node.get_representative_value()
        self._kinbody.SetTransform(representative)
        # compute the collision cost for this pose
        _, _, col_val, _ = self._kinbody_octree.compute_intersection(self._scene_sdf)
        # if col_val < 0:
            # return col_val
        # preference_val = np.dot(representative[:3, 1], np.array([0, 0, 1])) - representative[2, 3]
        preference_val = self._stability_function.compute_quality(representative)
        # preference_val = 0.0
        #TODO return proper objective
        return preference_val + col_val
        # return preference_val

    @staticmethod
    def is_better(val_1, val_2):
        """
            Returns whether val_1 is better than val_2
        """
        return val_1 > val_2


class SE3Hierarchy(object):
    """
        A grid hierarchy defined on SE(3)
    """
    class SE3HierarchyNode(object):
        """
            A representative for a node in the hierarchy
        """
        def __init__(self, cartesian_box, so3_key, depth, hierarchy, global_id=None):
            """
                Creates a new SE3HierarchyNode.
                None of the parameters may be None
                @param bounding_box - cartesian bounding box (min_point, max_point)
                @param so3_key - key of SO(3) hierarchy node
                @param depth - depth of this node
                @param hierarchy - hierarchy this node belongs to
                @param global_id - global id of this node within the hierarchy, None if root
            """
            self._global_id = global_id
            if self._global_id is None:
                self._global_id = ((), (), (), (), ())
            self._relative_id = SE3Hierarchy.extract_relative_id(self._global_id)
            self._cartesian_box = cartesian_box
            self._so3_key = so3_key
            self._depth = depth
            self._hierarchy = hierarchy
            self._so3_hierarchy = hierarchy._so3_hierarchy
            self._cartesian_range = cartesian_box[1] - cartesian_box[0]
            self._child_dimensions = self._cartesian_range / self._hierarchy._cart_branching
            self._child_cache = {}

        def get_random_node(self):
            """
                Return a random child node, as required by optimizers defined in
                the optimization module. This function simply calls get_random_child().
            """
            return self.get_random_child()

        def get_random_child(self):
            """
                Returns a randomly selected child node of this node.
                This selection respects the node blacklist of the hierarchy.
                Returns None, if this node is at the bottom of the hierarchy.
            """
            if self._depth == self._hierarchy._max_depth:
                return None
            bfs = self._so3_hierarchy.get_branching_factors(self._depth)
            random_child = np.array([np.random.randint(self._hierarchy._cart_branching),
                                     np.random.randint(self._hierarchy._cart_branching),
                                     np.random.randint(self._hierarchy._cart_branching),
                                     np.random.randint(bfs[0]),
                                     np.random.randint(bfs[1])], np.int)
            # TODO respect blacklist
            return self.get_child_node(random_child)

        def get_random_neighbor(self, node):
            """
                Returns a randomly selected neighbor of the given child node.
                This selection respects the node blacklist of the hierarchy and will extend the neighborhood
                if all direct neighbors are black listed.
                @param node has to be a child of this node.
            """
            random_dir = np.array([np.random.randint(-1, 2),
                                   np.random.randint(-1, 2),
                                   np.random.randint(-1, 2)])
            child_id = node.get_id(relative_to_parent=True)
            max_ids = [self._hierarchy._cart_branching - 1, self._hierarchy._cart_branching - 1, self._hierarchy._cart_branching - 1]
            # TODO respect blacklist
            neighbor_id = np.zeros(5, np.int)
            neighbor_id[:3] = np.clip(child_id[:3] + random_dir, 0, max_ids)
            so3_key = self._so3_hierarchy.get_random_neighbor(node.get_so3_key())
            neighbor_id[3] = so3_key[0][-1]
            neighbor_id[4] = so3_key[1][-1]
            return self.get_child_node(neighbor_id)

        def get_child_node(self, child_id):
            """
                Returns a node representing the child with the specified local id.
                @param child_id - numpy array of type int and length 5 (expected to be in range)
            """
            child_id_key = tuple(child_id)
            if child_id_key in self._child_cache:  # check whether we already have this child
                return self._child_cache[child_id_key]
            # else construct the child
            # construct range for cartesian part
            offset = child_id[:3] * self._child_dimensions
            min_point = np.zeros(3)
            min_point = self._cartesian_box[0] + offset
            max_point = min_point + self._child_dimensions
            global_child_id = SE3Hierarchy.construct_id(self._global_id, child_id_key)
            child_node = SE3Hierarchy.SE3HierarchyNode((min_point, max_point),
                                                        (global_child_id[3], global_child_id[4]),
                                                        self._depth + 1, self._hierarchy,
                                                        global_id=global_child_id)
            self._child_cache[tuple(child_id_key)] = child_node
            return child_node

        def get_children(self):
            """
                Return a generator that allows to iterate over all children.
            """
            bfs = self._so3_hierarchy.get_branching_factors(self._depth)
            local_keys = itertools.product(range(self._hierarchy._cart_branching),
                                           range(self._hierarchy._cart_branching),
                                           range(self._hierarchy._cart_branching),
                                           range(bfs[0]), range(bfs[1]))
            for lkey in local_keys:
                child = self.get_child_node(lkey)
                yield child

        def get_representative_value(self, rtype=0):
            """
                Returns a point in SE(3) that represents this cell, i.e. the center of this cell
                @param rtype - Type to represent point (0 = 4x4 matrix)
            """
            position = self._cartesian_box[0] + self._cartesian_range / 2.0
            quaternion = self._so3_hierarchy.get_quaternion(self._so3_key)
            if rtype == 0:  # return a matrix
                matrix = transformations.quaternion_matrix(quaternion)
                matrix[:3, 3] = position
                return matrix
            raise RuntimeError("Return types different from matrix are not implemented yet!")

        def get_id(self, relative_to_parent=False):
            """
                Returns the id of this node. By default global id. Local id if relative_to_parent is true
                You should not modify the returned id!
            """
            if relative_to_parent:
                return SE3Hierarchy.extract_relative_id(self._global_id)
            return self._global_id

        def get_so3_key(self):
            return self._so3_key

    """
        Implements a hierarchical grid on SE3.
    """
    def __init__(self, bounding_box, cart_branching, depth):
        """
            Creates a new hierarchical grid on SE(3), i.e. R^3 x SO(3)
            @param bounding_box - (min_point, max_point), where min_point and max_point are numpy arrays of length 3
                            describing the edges of the bounding box this grid should cover
            @param cart_branching - branching factor (number of children) for cartesian coordinates, i.e. x, y, z
            @param depth - maximal depth of the hierarchy (can be an integer in {1, 2, 3, 4})
        """
        self._cart_branching = cart_branching
        self._so3_hierarchy = so3hierarchy.SO3Hierarchy()
        self._max_depth = max(0, min(depth, self._so3_hierarchy.max_depth()))
        self._root = self.SE3HierarchyNode(cartesian_box=bounding_box,
                                           so3_key=self._so3_hierarchy.get_root_key(),
                                           depth=0,
                                           hierarchy=self)
        self._blacklist = None  # TODO need some data structure to blacklist nodes

    @staticmethod
    def extract_relative_id(global_id):
        """
            Extracts the local id from the given global id.
            Note that a local is a tuple (a,b,c,d,e,f) where all elements are integers.
            This is different from a global id!
        """
        depth = len(global_id[0])
        if depth == 0:
            return ()  # local root id is empty
        return tuple((global_id[i][depth - 1] for i in xrange(5)))

    @staticmethod
    def construct_id(parent_id, child_id):
        """
            Constructs the global id for the given child.
            Note that a global id is of type tuple(tuple(int, ...), ..., tuple(int, ...)),
            a local id on the other hand of type tuple(int, ..., int)
            @param parent_id - global id of the parent
            @param child_id - local id of the child
        """
        if parent_id is None:
            return tuple(((x,) for x in child_id))  # global id is of format ((a), (b), (c), (d), (e))
        return tuple((parent_id[i] + (child_id[i],) for i in xrange(5)))  # append elements of child id to individual elements of global id

    def get_root(self):
        return self._root

    def get_depth(self):
        return self._max_depth


class PlacementGoalPlanner:
    """This class allows to search for object placements in combination with
        the FreeSpaceProximitySampler.
    """
    def __init__(self, base_path,
                 env, scene_sdf, visualize=False):
        """
            Creates a PlacementGoalPlanner
            @param base_path Path where object data can be found
            @param env OpenRAVE environment
            @param scene_sdf SceneSDF of the OpenRAVE environment
            @param visualize If true, the internal OpenRAVE environment is set to be visualized
        """
        self._hierarchy = None
        self._env = env
        self._objective_function = PlacementObjectiveFn(env, scene_sdf, base_path)
        self._optimizer = optimization.StochasticOptimizer(self._objective_function)
        # self._optimizer = optimization.StochasticGradientDescent(self._objective_function)
        self._placement_volume = None
        self._env = env
        self._parameters = {'cart_branching': 3, 'max_depth': 4, 'num_iterations': 100}  # TODO update
        self._initialized = False

    def set_placement_volume(self, workspace_volume):
        """
            Sets the workspace volume in world frame in which the planner shall search for a placement pose.
            @param workspace_volume - (min_point, max_point), where both are np.arrays of length 3
        """
        self._placement_volume = workspace_volume
        self._objective_function.set_placement_volume(workspace_volume)
        self._initialized = False

    def sample(self, depth_limit, post_opt=True):
        """ Samples a placement configuration from the root level. """
        #TODO
        if not self._initialized:
            self._initialize()
        current_node = self.get_root()
        num_iterations = self._parameters['num_iterations']
        best_val = None
        best_node = None
        for depth in xrange(min(depth_limit, self.get_max_depth())):
            best_val, best_node = self._optimizer.run(current_node, num_iterations)
            current_node = best_node
        return best_node, best_val

    def sample_warm_start(self, hierarchy_node, depth_limit, label_cache=None, post_opt=False):
        """ Samples a placement configuration from the given node on. """
        #TODO
        pass

    def is_goal(self, sampling_result):
        """ Returns whether the given node is a goal or not. """
        # TODO
        pass

    def load_hand(self, hand_path, hand_cache_file, hand_config_file, hand_ball_file):
        """ Does nothing. """
        pass

    def set_object(self, obj_id, model_id=None):
        """ Set the object.
            @param obj_id String identifying the object.
            @param model_id (optional) Name of the model data. If None, it is assumed to be identical to obj_id
        """
        self._objective_function.set_target_object(obj_id, model_id)
        self._initialized = False

    def set_max_iter(self, iterations):
        self._parameters['num_iterations'] = iterations

    def get_max_depth(self):
        return self._hierarchy.get_depth()

    def get_root(self):
        if not self._initialized:
            self._initialize()
        return self._hierarchy.get_root()

    def set_parameters(self, **kwargs):
        self._initialized = False
        for (key, value) in kwargs.iteritems():
            self._parameters[key] = value

    def _initialize(self):
        if self._placement_volume is None:
            raise ValueError("Could not intialize as there is no placement volume available")
        if self._objective_function.get_target_object() is None:
            raise ValueError("Could not intialize as there is no placement target object available")
        self._hierarchy = SE3Hierarchy(self._placement_volume,
                                       self._parameters['cart_branching'],  # TODO it makes more sense to provide a resolution instead
                                       self._parameters['max_depth'])
        self._initialized = True
