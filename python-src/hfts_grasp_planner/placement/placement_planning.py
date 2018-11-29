#!/usr/bin/env python
import os
import rospy
import random
import itertools
import collections
import hfts_grasp_planner.placement.optimization as optimization
import hfts_grasp_planner.external.transformations as transformations
import hfts_grasp_planner.placement.arpo_placement.placement_orientations as po_mod
import hfts_grasp_planner.placement.so3hierarchy as so3hierarchy
import hfts_grasp_planner.placement.leafstage as leafstage
import hfts_grasp_planner.placement.chull_utils as chull_utils
import hfts_grasp_planner.sdf.kinbody as kinbody_sdf
import hfts_grasp_planner.sdf.robot as robot_sdf
import hfts_grasp_planner.utils as utils
from hfts_grasp_planner.sampler import GoalHierarchy
import numpy as np
import scipy.spatial
import openravepy as orpy


class SimplePlacementQuality(object):
    """
        Implements a simple quality function for a placement.
        This quality function is based on quasi-static analysis of contact stability.
        Assumes that gravity is pulling in direction -z = (0, 0, -1).
    """

    def __init__(self, env, filter_io, scene_sdf, parameters=None):
        """
            Create a new SimplePlacementQuality function object.
            ---------
            Arguments
            ---------
            env - fully initialized OpenRAVE environment. Note that the environment is cloned.
            filter_io - Object of class PreferenceFilterIO that allows reading
                in object-specific placement plane filters.
            scene_sdf, SceneSDF - signed distance field of the scene
            parameters - dictionary with parameters. Available parameters are:
                min_com_distance, float - minimal distance for center of mass from convex hull boundary TODO better description
                min_normal_similarity, float - minimal value for dot product between similar faces TODO better description
                falling_height_tolerance, float - tolerance for computing virtual contact points
        """
        self._filter_io = filter_io
        self._env = env.CloneSelf(orpy.CloningOptions.Bodies)
        col_checker = orpy.RaveCreateCollisionChecker(self._env, 'ode')  # Bullet is not working properly
        self._env.SetCollisionChecker(col_checker)
        # self._env.SetViewer('qtcoin')  # for debug only
        self._body = None
        self._scene_sdf = scene_sdf
        self._placement_planes = None  # list of placement planes, where each plane is a matrix (n +1, 3) matrix
        self._com_distances = None  # list of tuples (dist2d, dist3d), i.e. one for each placement plane, where dist2d is the maximal
                                    # distance between the projected center of mass and edges of the support, and the distance between the plane
                                    # and the 3d center of mass
        self._chull_distance_extremes = None  # best and worst chull_distances
        self._dir_gravity = np.array([0.0, 0.0, -1.0])
        self._local_com = None
        self._parameters = {"min_com_distance": 0.001, "min_normal_similarity": 0.97,
                            "minimal_contact_spread": 0.005, "maximal_contact_spread": 0.01,
                            "alpha_weight": 1.0, "d_weight": 1.0, "p_weight": 1.0, 'gamma_weight': 3.0
                            }
        self._max_ray_length = 2.0
        self._interior_radius = 0.0  # maximal radius of a ball completely inside the object's convex hull
        self._d_value_normalizer = 1.0  # maximum distance the object can fall in a target volume
        if parameters:
            for (key, value) in parameters:
                self._parameters[key] = value

    def compute_quality(self, node, return_terms=False):
        """
            Compute the placement quality for the given pose.
            The larger the returned value, the more suitable a pose is for placement.
            Note that this function is merely a heuristic.
            ---------
            Arguments
            ---------
            pose, numpy array of shape (4, 4) or SE3HierarchyNode - describes for what pose/partition
                to compute the quality for
            return_terms, bool - if true, the individual terms that the placement quality
                is computed of are returned.
            ---------
            Returns
            ---------
            score, float - the larger the better the pose is

            If return_terms is true, the function additionally returns:
            falling_distance, float - distance of how far the object will approximately fall
            chull_distance, float - signed distance of the projected center of mass to the planar convex hull
                of contact points
            alpha, float - angle between virtual placement plane and z-axis
            gamma, float - angle between virtual placement plane and placement plane
        """
        if type(node) == np.ndarray:
            pose = node
            z_partition = 0.0
        else:
            pose = node.get_representative_value()
            z_partition = pose[2, 3] - node.get_bounds()[2, 0]
            assert z_partition > 0.0
        self._body.SetTransform(pose)
        # self._env.SetViewer('qtcoin')
        # best_score = float('inf')
        # score, d_head, chull_distance, alpha, gamma
        best_plane_results = (0.0, float('inf'), float('inf'), float('inf'), float('inf'))
        # run over all placement planes
        for plane, dist_pair in itertools.izip(self._placement_planes, self._com_distances):
            # subscores
            d_value, p_value, alpha_value, gamma_value = 1.0, 1.0, 1.0, 1.0
            # check in what direction the plane is facing
            plane_normal = np.dot(pose[:3, :3], plane[0])
            if plane_normal[2] > 0.0:  # this plane is facing upwards
                distances, _, _, _ = self._project_placement_plane(plane, pose)  # project the plane, but we only care about the distances
                if len(distances) > 0:  # there is some surface beneath
                    d_head = max(np.min(distances) - z_partition, 0.0)
                else:  # there is no surface beneath
                    # d_head = self._max_ray_length
                    d_head = max(self._max_falling_distance - z_partition, 0.0)
                d_value = d_head / self._max_falling_distance
                chull_distance = dist_pair[1]
                p_value = 1.0
                alpha_value = 1.0
                alpha = np.pi / 2.0
                gamma = np.arccos(-np.clip(plane_normal[2], -1.0, 1.0))
                # gamma_value = (1.0 + plane_normal[2]) / 2.0  # in range [0.5, 1.0] for downside up planes
                gamma_value = gamma / np.pi
            else:  # plane is facing downwards
                virtual_contacts, virtual_plane_axes, distances = self._compute_virtual_contact_plane(plane, pose)
                if virtual_contacts.size > 0:
                    # we have some virtual contacts and thus finite distances
                    # rerieve the smallest distance between a placement point and its virtual contact
                    d_head = max(np.min(distances) - z_partition, 0.0)
                    d_value = d_head / self._max_falling_distance  # always >= 0.0, may be > 1.0 if dropped from a bad angle onto an edge
                    if virtual_plane_axes is not None:
                        # we have a virtual contact plane and can compute the full heuristic
                        # project the virtual contacts to the x, y plane and compute their convex hull
                        contact_footprint = scipy.spatial.ConvexHull(virtual_contacts[:, :2])
                        # TODO delete visualization
                        # hull_boundary = virtual_contacts[contact_footprint.vertices, :3]
                        # hull_boundary[:, 2] = np.min(hull_boundary[:, 2])
                        # handles = []
                        # handles.append(self._visualize_boundary(hull_boundary))
                        # projected_com = self._body.GetCenterOfMass()
                        # projected_com[2] = hull_boundary[0, 2]
                        # handles.append(self._env.drawbox(projected_com, np.array([0.005, 0.005, 0.005])))
                        # TODO until here
                        # retrieve the distance to the closest edge
                        chull_distance, _ = chull_utils.compute_hull_distance(contact_footprint, self._body.GetCenterOfMass()[:2])
                        # angle between virtual placement plane and z-axis
                        dot_product = np.dot(virtual_plane_axes[:, 2], np.abs(self._dir_gravity))
                        alpha = np.arccos(np.clip(dot_product, -1.0, 1.0))
                        alpha_value = 2.0 * alpha / np.pi  # alpha can be at most pi/2.0, thus alpha_value is in [0, 1]
                        # next, compute the angle between the placement plane and the virtual placement plane. Our stability estimate
                        # is only somehow accurate if these two align. This angle lies in [0, pi]
                        dot_product = np.clip(np.dot(virtual_plane_axes[:, 2], -1.0 * plane_normal), -1.0, 1.0)
                        gamma = np.arccos(dot_product)  # in [0, pi]
                        gamma_value = gamma / np.pi
                        # thus penalize invalid angles through the following score
                        # sigmoid that is small until gamma = pi/4 and 1 from pi/2 onwards
                        # weight = 1.0 / (1.0 + np.exp(-24.0 / np.pi * (gamma - np.pi / 4.0)))
                        # stability = (1.0 - weight) * chull_distance + weight * dist_pair[1]  # interpolate between chull_distance and worst possible chull_distance
                        p_value = (chull_distance - self._chull_distance_extremes[0]) / (np.abs(self._chull_distance_extremes[0]) + self._chull_distance_extremes[1])  # lies in [0, 1]
                    else:
                        # we have contacts, but insufficiently many to fit a plane, so give it bad
                        # scores for anything related to the virtual placement plane
                        chull_distance = dist_pair[1]
                        gamma = np.arccos(-np.clip(plane_normal[2], -1.0, 1.0))  # in [0, pi]
                        gamma_value = gamma / np.pi
                        p_value = 1.0
                        alpha = np.pi / 2.0
                        alpha_value = 1.0
                else:
                    # we have no contacts, so there isn't anything we can say
                    d_head = max(self._max_falling_distance - z_partition, 0.0)
                    d_value = 1.0
                    chull_distance = dist_pair[1]
                    gamma = np.arccos(-np.clip(plane_normal[2], -1.0, 1.0))  # in [0, pi]
                    gamma_value = gamma / np.pi
                    p_value = 1.0
                    alpha = np.pi / 2.0
                    alpha_value = 1.0
            # we want to maximize the following score
            # minimizing d_value drives the placement plane downwards and aligns the placement plane with the surface
            # minimizing the p_value leads to rotating the object so that the placement plane gets aligned with the surface,
            # or if the plane is already aligned, maximizes the stability.
            # Minimizing alpha_value prefers placements on planar surfaces rather than on slopes.
            # score = d_head + placement_value + self._parameters["slopiness_weight"] * alpha
            assert(d_value >= 0.0)  # it may be larger 1.0 for really bad cases
            assert(p_value >= 0.0)
            # assert(p_value >= 0.0 and p_value <= 1.0) # it may be larger than 1.0 if only a part of a placement plane is above the surface
            assert(alpha_value >= 0.0 and alpha_value <= 1.0)
            assert(gamma_value >= 0.0 and gamma_value <= 1.0)
            score = self._parameters['d_weight'] * (1.0 - d_value) + self._parameters['p_weight'] * (1.0 - p_value) + \
                self._parameters['alpha_weight'] * (1.0 - alpha_value) + self._parameters['gamma_weight'] * (1.0 - gamma_value)
            # normalize score back to [0, 1]
            score /= (self._parameters['d_weight'] + self._parameters['p_weight'] +
                      self._parameters['alpha_weight'] + self._parameters['gamma_weight'])
            if score > best_plane_results[0]:
                best_plane_results = score, d_head, chull_distance, alpha, gamma
        rospy.logdebug("Stability results - score:%f, d_head: %f, chull: %f, alpha: %f, gamma: %f" % best_plane_results)
        if return_terms:
            return best_plane_results[0], best_plane_results[1],\
                best_plane_results[2], best_plane_results[3], best_plane_results[4]
        return best_plane_results[0]

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
        self._max_falling_distance = self._workspace_volume[1][2] - self._workspace_volume[0][2] - self._interior_radius

    def set_placement_volume(self, workspace_volume):
        """
            Set the placement volume.
            @param workspace_volume - (min_point, max_point), where both are np.arrays of length 3
        """
        self._workspace_volume = np.array(workspace_volume)

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
            dist2d, float - distance of the projected center of mass to any boundary
            min_dist2d, float minimal distance the projected center of mass can have
            dist3d, float - maximal distance the projected center of mass can have
        """
        # handles = self._visualize_placement_plane(plane)  # TODO remove
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
        dir_to_com = self._local_com - mean_point
        projected_com = self._local_com - np.dot(dir_to_com, plane[0]) * plane[0]  # this is still object frame
        dist3d = np.linalg.norm(self._local_com - projected_com)  # also still in object frame
        # we express the hull relative to the mean point, so also com needs to be expressed relative to that point
        com2d = np.dot(projected_com - mean_point, axes)  
        # compute the convex hull of the projected points
        convex_hull = scipy.spatial.ConvexHull(points_2d)
        # ##### DRAW CONVEX HULL ###### TODO remove
        # boundary = points_2d[convex_hull.vertices]
        # compute 3d boundary from bases and mean point
        # boundary3d = boundary[:, 0, np.newaxis] * axes[:, 0] + boundary[:, 1, np.newaxis] * axes[:, 1] + mean_point
        # transform it to world frame
        # tf = self._body.GetTransform()
        # boundary3d = np.dot(boundary3d, tf[:3, :3].transpose()) + tf[:3, 3]
        # handles.append(self._visualize_boundary(boundary3d))
        # ##### DRAW CONVEX HULL - END ######
        # ##### DRAW PROJECTED COM ######
        # handles.append(self._env.drawbox(projected_com, np.array([0.005, 0.005, 0.005]),
                                        #  np.array([0.29, 0, 0.5]), tf))
        # ##### DRAW PROJECTED COM - END ######
        # accept the point if the projected center of mass is inside of the convex hull
        dist2d, _ = chull_utils.compute_hull_distance(convex_hull, com2d)
        best_dist = np.min(convex_hull.equations[:, -1])
        return dist2d < -1.0 * self._parameters["min_com_distance"], dist2d, best_dist, dist3d

    def _compute_placement_planes(self, user_filters):
        # first compute the convex hull of the body
        # self._env.SetViewer('qtcoin')
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
        assert (convex_hull.equations[:, -1] < 0.0).all()
        self._interior_radius = np.min(np.abs(convex_hull.equations[:, -1]))
        # merge faces
        clusters, _, _ = chull_utils.merge_faces(convex_hull, self._parameters["min_normal_similarity"])
        # handles = self._visualize_clusters(convex_hull, clusters, face_clusters)
        # handles = None
        self._placement_planes = []
        self._com_distances = []
        self._chull_distance_extremes = [float('inf'), float('-inf')]  # best and worst chull_distances
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
            b_stable, _, dist2d, dist3d = self._is_stable_placement_plane(plane)
            if not b_stable:
                continue
            # if this plane passed all filters, we accept it
            self._placement_planes.append(plane)
            self._com_distances.append((dist2d, dist3d))
            self._chull_distance_extremes[0] = min(dist2d, self._chull_distance_extremes[0])
            self._chull_distance_extremes[1] = max(dist3d, self._chull_distance_extremes[1])
        if not self._placement_planes:
            raise ValueError("Failed to compute placement planes for body %s." % self._body.GetName())
        assert(self._chull_distance_extremes[0] < self._chull_distance_extremes[1])
        assert(self._chull_distance_extremes[0] < 0.0)
        assert(self._chull_distance_extremes[1] > 0.0)
        # handles = self._visualize_placement_planes()

    def _synch_env(self, env):
        with env:
            with self._env:
                bodies = env.GetBodies()
                for body in bodies:
                    my_body = self._env.GetKinBody(body.GetName())
                    if not my_body:
                        raise RuntimeError("Could not find body with name " + body.GetName() + " in cloned environment")
                    my_body.SetTransform(body.GetTransform())

    def _project_placement_plane(self, placement_plane, pose):
        """
            Project the given placement plane along the direction of gravity to the surface.
            ---------
            Arguments
            ---------
            placement_plane, numpy array of shape (n+1, 3), where n is the number of points
                on the plane and placement_plane[0, :] is its normal.
            pose, numpy array of shape (4, 4), current pose of the body
            ---------
            Returns
            ---------
            distance, numpy array of shape (k,) - distances between projected and non-projected points, for those where the projection is successful
            virtual_contacts, numpy array of shape (k, 6) - the projected contact points
            tf_plane - placement plane transformed by pose.
            collisions, np array of shape (n,) of type bool - True if projection hit a surface
        """
        # first transform the placement plane to global frame
        tf_plane = np.dot(placement_plane, pose[:3, :3].transpose())
        tf_plane[1:] += pose[:3, 3]
        assert(tf_plane.shape[0] >= 4)
        # next check which of placement plane points are not in collision already
        omni_dir_distances = self._scene_sdf.get_distances(tf_plane[1:])
        colliding = np.where(omni_dir_distances <= 0.0)[0]
        non_colliding = np.where(omni_dir_distances > 0.0)[0]
        collisions = np.array((tf_plane.shape[0] - 1) * [True])
        virtual_contacts = np.empty((tf_plane.shape[0] - 1, 6))
        virtual_contacts[colliding, :3] = tf_plane[1 + colliding]
        virtual_contacts[colliding, 3:] = -tf_plane[0]
        # perform ray tracing to compute projected contact points for those that are not in collision
        if non_colliding.shape[0] > 0:
            rays = np.zeros((non_colliding.shape[0], 6))
            rays[:, 3:] = self._max_ray_length * self._dir_gravity
            rays[:, :3] = tf_plane[1 + non_colliding]
            self._body.Enable(False)
            collisions[non_colliding], virtual_contacts[non_colliding] = self._env.CheckCollisionRays(rays)
            self._body.Enable(True)
        distances = np.linalg.norm(tf_plane[1:] - virtual_contacts[:, :3], axis=1)
        return distances[collisions], virtual_contacts[collisions], tf_plane, collisions

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
            virtual_contact_points, numpy arrays of shape (k, 3), where k is the maximal number of
                of projected contacts that form a plane, if any. If there are more than k points
                that were projected on a surface, these additional points would span contact volume rather than a plane.
                In normal cases k >= 3, but it may be smaller if there are insufficiently many projected contact points.
                In this case virtual_plane_axes is None.
            virtual_plane_axes, numpy array of shape (3, 3). The first two columns span a plane fitted into
                the virtual contact points. The third column is the normal of this plane (only defined if k >= 3).
            distances, numpy array of shape (k,) where n is the total number of point in the placement plane.
                Note that some of these distances may be infinity, if there is no surface within
                self._max_ray_length below this body. Distances are in the order of placement_plane points.
        """
        distances, virtual_contacts, _, collisions = self._project_placement_plane(placement_plane, pose)
        # # DRAW CONTACT ARROWS ###### TODO remove
        # handles = []
        # for idx, contact in enumerate(virtual_contacts):
        #     if collisions[idx] and distances[idx] > 0.001:  # distances shouldn't be too small, otherwise the viewer crashes
        #         handles.append(self._env.drawarrow(tf_plane[idx + 1], contact[:3], linewidth=0.002))
        ##### DRAW CONTACT ARROWS - END ######
        # if we have less than three contacts, there is nothing we can do
        if virtual_contacts.shape[0] < 3:
            return virtual_contacts, None, distances
        # else sort the contacts by distance
        contact_distance_tuples = zip(distances, range(distances.shape[0]))
        contact_distance_tuples.sort(key=lambda x: x[0])
        sorted_indices = [cdt[1] for cdt in contact_distance_tuples]
        # next, select the maximal number of contact points such that they still span a plane in x, y
        b_has_support_plane = False
        for idx in xrange(virtual_contacts.shape[0], 2, -1):
            normalized_contacts = virtual_contacts[sorted_indices[:idx], :3] - np.mean(virtual_contacts[sorted_indices[:idx], :3], axis=0)
            _, _, v = np.linalg.svd(normalized_contacts)
            # compute maximal distance between any point from the mean point along the each axis of spread
            spread_0 = np.max(np.abs(np.dot(normalized_contacts, v[0])))
            spread_1 = np.max(np.abs(np.dot(normalized_contacts, v[1])))
            spread_2 = np.max(np.abs(np.dot(normalized_contacts, v[2])))
            min_spread = self._parameters["minimal_contact_spread"]
            max_spread = self._parameters["maximal_contact_spread"]
            if spread_0 > min_spread and spread_1 > min_spread and spread_2 < max_spread:
                b_has_support_plane = True
                break
            elif spread_2 < min_spread:  # this implies that spread_0 or spread_1 is already too small
                b_has_support_plane = False
                break
            # else continue with the loop

        # do not report the plane if we don't have any or it is orthogonal to the z axis
        if not b_has_support_plane or np.abs(v[2, 2]) < 1e-5:
            return virtual_contacts, None, distances
        # if we are here, that means virtual_contacts[sorted_indices[:idx]] span a valid support plane
        virtual_contacts = virtual_contacts[sorted_indices[:idx]]
        distances = distances[sorted_indices[:idx]]
        # flip the normal of the virtual placement plane such that it points upwards
        if v[2, 2] < 0.0:
            v[2, :] *= -1.0
        # DRAW CONTACT PLANE ###### TODO remove
        # handles.append(self._env.drawarrow(mean_point, mean_point + 0.1 *
        #                                    v[0, :], linewidth=0.002, color=[1.0, 0, 0, 1.0]))
        # handles.append(self._env.drawarrow(mean_point, mean_point + 0.1 *
        #                                    v[1, :], linewidth=0.002, color=[0.0, 1.0, 0.0, 1.0]))
        # handles.append(self._env.drawarrow(mean_point, mean_point + 0.1 *
        #                                    v[2, :], linewidth=0.002, color=[0.0, 0.0, 1.0, 1.0]))
        ##### DRAW CONTACT ARROWS - END ######
        return virtual_contacts, v.transpose(), distances

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


class QuasistaticFallingQuality(object):
    """
        Implements a quality function to evaluate placement stability. The function implemented
        by this class quasistatically simulates how an object falls if released at a given pose.
        At each contact, the function computes in which direction the object will tip under the assumption
        that the kinectic energy of the object is zero.
    """

    def __init__(self, env):
        """
            Create a new instance of a QuasistaticFallingQuality function.
            ---------
            Arguments
            ---------
            env, OpenRAVE environment - fully initialized OpenRAVE environment of the scene in which
                objects should be placed. The environment is cloned.
        """
        self._env = env.CloneSelf(orpy.CloningOptions.Bodies)
        # only fcl supports contiuous collision detection
        self._fcl_col_checker = orpy.RaveCreateCollisionChecker(self._env, 'fcl_')
        # self._ode_col_checker = orpy.RaveCreateCollisionChecker(self._env, 'ode')
        self._fcl_col_checker.SetCollisionOptions(orpy.CollisionOptions.Contacts |
                                                  orpy.CollisionOptions.AllGeometryContacts |
                                                  orpy.CollisionOptions.AllLinkCollisions)
        self._env.SetCollisionChecker(self._fcl_col_checker)
        orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
        self._env.SetDebugLevel(orpy.DebugLevel.Debug)
        self._handles = []
        self._placement_volume = None
        self._body = None
        self._link = None
        self._body_radius = None
        self._parameters = {
            "max_iterations": 100,
            "collision_step_size": 0.01,
            "rot_step_size": 0.01,
            "minimal_chull_distance": 0.0,
            "max_sloping_angle": np.pi / 9.0,
            "minimal_contact_spread": 25e-4,
        }

    def compute_quality(self, pose, return_terms=False):
        """
            Compute the placement quality for the given pose.
            The larger the returned value, the more suitable a pose is for placement.
            Note that this function is merely a heuristic.
            ---------
            Arguments
            ---------
            pose, numpy array of shape (4, 4) - describes the pose of the currently
                set kinbody
            return_terms, bool - if true, the individual terms that the placement quality
                is computed of are returned.
            ---------
            Returns
            ---------
            score, float - the larger the better the pose is

            If return_terms is true, the function additionally returns:
            bcomes_to_rest, bool - True, if the object comes to rest within the placement volume
            falling_distance, float - distance of how far the object will fall until first contact
            acc_rotation, float - accumulated rotation in radians the object will rotate until it comes to rest
            chull_distance, float - signed distance of the projected center of mass to the planar convex hull
                of contact points when the object comes to rest
        """
        assert(self._body is not None)
        assert(self._link is not None)
        assert(self._placement_volume is not None)
        # iterations = 0
        b_at_rest = False
        b_slippage = False
        b_fell_out_of_volume = False
        ccd_report = orpy.ContinuousCollisionReport()
        dcd_report = orpy.CollisionReport()
        self._body.SetTransform(pose)
        acc_distance = 0.0
        acc_rotation = 0.0
        # TODO this procedure does not make sense when the body is initially in contact
        while not b_at_rest and not b_slippage:
            self._clear_visualization()
            current_tf = self._body.GetTransform()
            # TODO check if current_tf is still within placement volume
            # self._env.SetCollisionChecker(self._ode_col_checker)
            self._env.CheckCollision(self._body, dcd_report)
            # get contacts from collision report
            contacts = np.array([[ct.pos, ct.norm]
                                 for ct in dcd_report.contacts]).reshape((len(dcd_report.contacts), 6))
            # classify contacts in supporting contacts (compensating gravity), sliding contacts and others
            sup_contacts, sliding_contacts, other_contacts = self._classify_contacts(contacts)
            self._visualize_contacts(sup_contacts, color=[1,0,0])
            self._visualize_contacts(sliding_contacts, color=[0,1,0])
            self._visualize_contacts(other_contacts, color=[0,0,1])
            # if we have no supporting contacts, the body may be sliding
            if sup_contacts.shape[0] == 0 and sliding_contacts.shape[0] > 0:  # slippage
                b_slippage = True
                break
            elif sup_contacts.shape[0] == 0:  # the object is falling
                # translate along z axis downwards until first contact
                target_tf = self._body.GetTransform()
                target_tf[2, 3] = self._placement_volume[0][2] - self._body_radius
                # self._env.SetCollisionChecker(self._fcl_col_checker)
                b_contact = self._env.CheckContinuousCollision(self._link, target_tf, ccd_report)
                if not b_contact:
                    b_fell_out_of_volume = True
                    break
                contact_quat_pose = ccd_report.vCollisions[0][1]
                acc_distance += np.linalg.norm(contact_quat_pose[4:] - current_tf[:3, 3])
                new_tf = orpy.matrixFromQuat(contact_quat_pose[:4])
                new_tf[:3, 3] = contact_quat_pose[4:]
                self._body.SetTransform(new_tf)
            else:  # the object may rotate or is at rest
                rot_axis, rot_point, b_at_rest, b_slippage = self._compute_rotation(sup_contacts, sliding_contacts, other_contacts)
                if not b_at_rest and not b_slippage:
                    acc_rotation += self._parameters["rot_step_size"]
                    self._visualize_rotation_dir(rot_axis, rot_point)
                    # rotate, TODO cache rotation matrix?
                    tf_rot_frame = transformations.rotation_matrix(
                        self._parameters["rot_step_size"], rot_axis, rot_point)
                    self._body.SetTransform(np.dot(tf_rot_frame, current_tf))

        if b_fell_out_of_volume or b_slippage:
            # TODO decide what to return in these cases
            if return_terms:
                return float('-inf'), False, float('inf'), float('inf'), float('inf')
            return float('-inf')
        if return_terms:
            return 0.0, b_at_rest, acc_distance, acc_rotation, 0.0  # TODO return correct values
        return 0.0  # TODO return correct values

    def set_target_object(self, body, model_name=None):
        """
            Set the target object.
            Synchronizes the underlying OpenRAVE environment with the environment of the given body.
            If the environment of the body has more kinbodies than the underlying environment,
            a RuntimeError is thrown. This is due to a bug in OpenRAVE, that doesn't allow
            us to load additional kinbodies into any environment after a viewer has been set.
            ---------
            Arguments
            ---------
            body, OpenRAVE Kinbody - target object, which must have exactly one link
            model_name (string, optional) - name of the kinbody model if it is different from the kinbody's name
        """
        self._body = self._env.GetKinBody(body.GetName())
        if not self._body:
            raise ValueError("Could not find body " + body.GetName() + " in cloned environment.")
        self._body_radius = np.linalg.norm(self._body.ComputeAABB().extents())
        self._link = self._body.GetLinks()[0]
        if len(self._body.GetLinks()) != 1:
            raise RuntimeError("Can not operate on bodies with more than one link")
        self._synch_env(body.GetEnv())
        if not model_name:
            model_name = body.GetName()

    def set_placement_volume(self, workspace_volume):
        """
            Set the placement volume.
            @param workspace_volume - (min_point, max_point), where both are np.arrays of length 3
        """
        self._placement_volume = np.array(workspace_volume)

    def _synch_env(self, env):
        with env:
            with self._env:
                bodies = env.GetBodies()
                for body in bodies:
                    my_body = self._env.GetKinBody(body.GetName())
                    if not my_body:
                        raise RuntimeError("Could not find body with name " + body.GetName() + " in cloned environment")
                    my_body.SetTransform(body.GetTransform())

    def _classify_contacts(self, contacts):
        """
            Classify the given contacts into supporting contacts, sliding contacts and others.
            ---------
            Arguments
            ---------
            contacts, numpy array of shape (n, 6) - n > 0 contacts in shape [x, y, z, nx, ny, nz]
            -------
            Returns
            -------
            sup_contacts, numpy array of shape (n_1, 6) - n_1 >= 0 contacts that support the weight of the object,
                i.e. stop it from falling
            sliding_contacts, numpy array of shape (n_2, 6) - n_2 >= 0 contacts that are supporting, but the object will likely
                slide downwards, e.g. contacts on a steep slope
            other_contacts, numpy array of shape (n_3, 6) - n_3 >= 0 contacts that are neither supporting nor sliding. It is
                n = n_1 + n_2 + n_3
        """
        # TODO filter out contacts for which the object normal points in the same direction
        normal_dot_products = np.dot(contacts[:, 3:], np.array([0.0, 0.0, 1.0]))
        sup_contacts_idx = normal_dot_products >= np.cos(self._parameters["max_sloping_angle"])
        sup_contacts = contacts[sup_contacts_idx]
        sliding_contacts_idx = np.logical_and(normal_dot_products > 0, np.logical_not(sup_contacts_idx))
        sliding_contacts = contacts[sliding_contacts_idx]
        other_contacts = contacts[np.logical_and(np.logical_not(
            sup_contacts_idx), np.logical_not(sliding_contacts_idx))]
        return sup_contacts, sliding_contacts, other_contacts

    def _compute_rotation(self, sup_contacts, sliding_contacts, other_contacts):
        """
            Compute the rotation direction for the body at its current pose given the contacts.
            ---------
            Arguments
            ---------
            contacts, numpy array of shape (n_1, 6) - n_1 > 0 supporting contacts in shape [x, y, z, nx, ny, nz]
            sliding_contacts, numpy array of shape (n_2, 6) - n_2 >= 0 sliding contacts in shape [x, y, z, nx, ny, nz]
            other_contacts, numpy array of shape (n_3, 6) - n_3 >= 0 other contacts in shape [x, y, z, nx, ny, nz]
            -------
            Returns
            -------
            rot_axis, numpy array of shape (3,) or None - axis in world frame to rotate body around
            rot_point, numpy array of shape (3,) or None - rotation center in world frame
            b_at_rest, bool - whether the body is at rest or not. If at rest, rot_axis and rot_point are None
            b_sliding, bool - whether the body is at risk to slide
        """
        b_at_rest, b_has_rotation, b_sliding = False, False, False
        rot_axis, rot_point = None, None
        support_shape = 0  # initially we assume we have a point contact
        chull = None
        active_contacts = sup_contacts
        assert(active_contacts.shape[0] > 0)
        if sliding_contacts.shape[0] > 0 and other_contacts.shape[0] > 0:
            inactive_contacts = np.concatenate((sliding_contacts, other_contacts))
        elif sliding_contacts.shape[0] > 0:
            inactive_contacts = sliding_contacts
        else:
            inactive_contacts = other_contacts
        prev_hull_points = None  # boundary of the convex hull in the previous iteration
        new_active_contacts = sup_contacts  # contacts that became active in this iteration
        while not b_at_rest and not b_has_rotation and not b_sliding:
            # first compute what type of support shape we have, unless we already know we have planar contact
            if support_shape != 2:
                support_shape, line_start, line_end = self._characterize_support_shape(active_contacts)
            if support_shape == 2:  # if we have planar support, compute the convex hull
                if prev_hull_points is not None:  # if we have the boundary of a previous convex hull, reuse it
                    assert(new_active_contacts.shape[0] > 0)
                    active_contacts = np.concatenate((prev_hull_points, new_active_contacts))
                    chull = scipy.spatial.ConvexHull(active_contacts[:, :2])
                    prev_hull_points = active_contacts[chull.vertices]  # TODO could we only use active_cotnacts?
                else:
                    # build a convex hull from scratch
                    chull = scipy.spatial.ConvexHull(active_contacts[:, :2])
                    prev_hull_points = active_contacts[chull.vertices]
                rot_axis, rot_point = self._compute_rotation_dir(support_shape, active_contacts, chull)
            else:  # we have a line or point contact
                rot_axis, rot_point = self._compute_rotation_dir(support_shape, active_contacts, (line_start, line_end))
            if rot_axis is not None:
                # we have a rotation axis, so let's compute whether there any opposing contacts that become active now
                new_active_contacts, sliding_active_contacts, inactive_contacts = self._compute_opposing_contacts(
                    inactive_contacts, rot_axis, rot_point)
                if new_active_contacts.shape[0] > 0:  # do we have new active contacts?
                    active_contacts = np.concatenate((active_contacts, new_active_contacts))
                elif sliding_active_contacts.shape[0] > 0:
                    # there are new active contacts, but they are sliding
                    b_sliding = True
                else:
                    # if there are no opposing contacts, we can rotate
                    b_has_rotation = True
            else:
                b_at_rest = True
        return rot_axis, rot_point, b_at_rest, b_sliding

    def _characterize_support_shape(self, contacts):
        """
            Project the given contacts to the x, y plane and analyse the shape these projected points form.
            ---------
            Arguments
            ---------
            contacts, numpy array of shape (n, 6) - contact points that support the object
            -------
            Returns
            -------
            support_shape, int - 0 = contacts form a point, 1 = contacts form a line, 2 = contacts span a plane
            start_point, numpy array of shape (3,) - start point of the line, if support_shape = 1, else None
            end_point, numpy array of shape (3,) - end point of the line, if support_shape = 1, else None
        """
        assert(contacts.shape[0] > 0)
        # handles = self._visualize_contacts(contacts)
        support_shape = 0  # 0 if point, 1 if line, 2 if planar
        start_point, end_point = None, None
        if contacts.shape[0] == 1:
            support_shape = 0
        elif contacts.shape[0] == 2:
            # compute how far this points are apart
            start_point = contacts[0, :3]
            end_point = contacts[1, :3]
            if np.linalg.norm(end_point - start_point) > self._parameters["minimal_contact_spread"] / 2.0:
                support_shape = 1
            else:
                support_shape = 0
        else:  # we have at least three contacts
            normalized_contacts = contacts[:, :3] - np.mean(contacts[:, :3], axis=0)
            _, s, v = np.linalg.svd(normalized_contacts)
            # comppute maximal distance between any point from the mean point along the main axis of spread
            spread_0 = np.max(np.abs(np.dot(normalized_contacts, v[0])))
            if spread_0 > self._parameters["minimal_contact_spread"]:
                # now check the second axis
                spread_1 = np.max(np.abs(np.dot(normalized_contacts, v[1])))
                if spread_1 > self._parameters["minimal_contact_spread"]:
                    # if it large enough, the contacts span a plane
                    support_shape = 2
                else:
                    # else they only span a line
                    start_point = contacts[np.argmin(np.dot(normalized_contacts, v[0])), :3]
                    end_point = contacts[np.argmax(np.dot(normalized_contacts, v[0])), :3]
                    support_shape = 1
            else:
                # if this spread is small, consider all contacts to be a point
                support_shape = 0
        return support_shape, start_point, end_point

    def _compute_opposing_contacts(self, contacts, rot_axis, rot_point):
        """
            Compute the contacts that oppose a rotation around the given rotation axis.
            ---------
            Arguments
            ---------
            contacts, numpy array of shape (m, 6) - m number of contacts
            rot_axis, numpy array of shape (3,) - axis in world frame
            rot_point, numpy array of shape (3,) - center of rotation in world frame
            --------
            Returns
            --------
            opposing contacts, numpy array of shape (k, 6)
            sliding_opposing_contacts, numpy array of shape (j, 6)
            non-opposing contacts, numpy array of shape (m-k-j, 6)
        """
        if contacts.shape[0] == 0:
            return np.empty((0, 6)), contacts, contacts
        rel_positions = contacts[:, :3] - rot_point
        motion_dirs = np.cross(rot_axis, rel_positions)
        norms = np.linalg.norm(motion_dirs, axis=1)
        motion_dirs /= norms[:, np.newaxis]
        dps = np.sum(motion_dirs * contacts[:, 3:], axis=1)
        opposing_contact_idx = dps < -np.cos(self._parameters["max_sloping_angle"])  # -1e-4
        sliding_opposing_idx = np.logical_and(dps < -1e-4, np.logical_not(opposing_contact_idx))
        opposing_contacts = contacts[opposing_contact_idx]
        if opposing_contacts.shape[0] == 0:
            return opposing_contacts, contacts[sliding_opposing_idx], contacts[np.logical_not(sliding_opposing_idx)]
        # some contacts may be on top of the object preventing it from tipping. These we need to mirror around the
        # center of mass of the object so that we can take them into account when computing the support plane.
        # These contact points have the property that they are on the opposite side of the plane spanned
        # by the rotation axis and the z-axis than the center of mass of the object.
        # first compute the normal of this plane
        plane_normal = np.cross(rot_axis, np.array([0, 0, 1]))
        plane_normal /= np.linalg.norm(plane_normal)
        com = self._body.GetCenterOfMass()
        rel_com = com - rot_point
        # the center of mass should always be on the side where we rotate to
        assert(np.dot(plane_normal, rel_com) > 0.0)
        # so we need to flip the contacts on the other side of the plane
        contacts_to_flip_idx = np.dot(rel_positions[opposing_contact_idx], plane_normal) < -1e-4
        if contacts_to_flip_idx.any():
            # do the flipping: c_pos = com - (c_pos - com)
            opposing_contacts[contacts_to_flip_idx][:, :3] = 2.0 * com - opposing_contacts[contacts_to_flip_idx][:, :3]
            opposing_contacts[contacts_to_flip_idx][:, 3:] *= -1.0
        others_idx = np.logical_and(np.logical_not(sliding_opposing_idx), np.logical_not(opposing_contact_idx))
        return opposing_contacts, contacts[sliding_opposing_idx], contacts[others_idx]

    def _compute_rotation_dir(self, support_shape, contacts, data):
        """
            Compute the rotation direction for the body at its current pose given the contacts.
            ---------
            Arguments
            ---------
            support_shape, int - 0 = point contact, 1 = line contact, 2 = planar contact
            contacts, numpy array of shape (n, 6) - n > 0 contacts in shape [x, y, z, nx, ny, nz]
            data, varying - if support_shape == 0: None. 
                            if support_shape == 1:
                                a tuple (start_point, end_point) describing the line
                            if support_shape == 2:
                                ConvexHull of the 2d projection of the given contacts
            -------
            Returns
            -------
            rot_axis, numpy array of shape (3,) or None - axis in world frame to rotate body around
            rot_point, numpy array of shape (3,) or None - rotation center in world frame
            Both values are None, if the object is placed stably
        """
        com = self._body.GetCenterOfMass()
        grav_dir = np.array([0, 0, -1.0])
        if support_shape == 0:  # the contacts form a point
            rot_center = np.mean(contacts[:, :3], axis=0)
            rot_axis = np.cross(com - rot_center, grav_dir)
            rot_axis /= np.linalg.norm(rot_axis)
            return rot_axis, rot_center
        elif support_shape == 1:  # the contacts span a line
            rot_axis, rot_center = self._compute_rotation_dir_line(data[0], data[1], com)
            return rot_axis, rot_center
        else:  # contacts span a plane
            convex_hull = data
            dist, edge_id = chull_utils.compute_hull_distance(convex_hull, com[:2])
            if dist < self._parameters["minimal_chull_distance"]:
                return None, None
            start_point = contacts[convex_hull.simplices[edge_id][0]][:3]
            end_point = contacts[convex_hull.simplices[edge_id][1]][:3]
            rot_axis, rot_center = self._compute_rotation_dir_line(start_point, end_point, com)
            return rot_axis, rot_center

    @staticmethod
    def _compute_rotation_dir_line(start, end, com):
        """
            Compute the rotation axis given that the provided line is the line of contacts
            the object tips over.
            ---------
            Arguments
            ---------
            start, numpy array of shape (3,) - start point of the line
            end, numpy array of shape (3,) - end point of the line
            com, numpy array of shape (3,) - center of mass
            -------
            Returns
            -------
            rot_axis, numpy array of shape (3,) - rotation axis
            rot_center, numpy array of shape (3,) - center of rotation
        """
        # proj_start = start[:2]
        # proj_end = end[:2]
        _, _, t = utils.compute_closest_point_on_line(start, end, com)
        grav_dir = np.array([0, 0, -1.0])
        if t == 0.0:
            rot_center = start
            rot_axis = np.cross(com - rot_center, grav_dir)
            rot_axis /= np.linalg.norm(rot_axis)
            return rot_axis, rot_center
        if t == 1.0:
            rot_center = end
            rot_axis = np.cross(com - rot_center, grav_dir)
            rot_axis /= np.linalg.norm(rot_axis)
            return rot_axis, rot_center
        rot_center = start + t * (end - start)
        rot_axis = start - end
        rot_axis /= np.linalg.norm(rot_axis)
        # The rotation axis is start - end, but we need to determine its rotation
        # compute in which direction the center of mass will move (it has to move downwards)
        f_z = rot_axis[0] * (com[1] - rot_center[1]) - rot_axis[1] * (com[0] - rot_center[0])
        assert(abs(f_z) > 0.0)
        if f_z > 0.0:
            rot_axis = -rot_axis
        return rot_axis, start

    def _visualize_contacts(self, contacts, arrow_length=0.01, arrow_width=0.0001, color=None):
        if color is None:
            color = [1, 0, 0]
        for contact in contacts:
            if type(contact) == np.ndarray:
                p1 = contact[:3]
                p2 = contact[:3] + arrow_length * contact[3:]
            else:
                p1 = contact.pos
                p2 = p1 + arrow_length * contact.norm
            self._handles.append(self._env.drawarrow(p1, p2, arrow_width, color=color))

    def _visualize_line(self, start_point, end_point, width=0.0001, color=None):
        if color is None:
            color = [1, 0, 0]
        self._handles.append(self._env.drawarrow(start_point, end_point, width, color=color))

    def _visualize_rotation_dir(self, rot_axis, rot_point, color=None):
        self._visualize_line(rot_point - 0.3 * rot_axis, rot_point + 0.3 * rot_axis, width=0.001, color=color)

    def _clear_visualization(self):
        self._handles = []


class SimpleGraspCompatibility(object):
    """
        Implements a heuristic to determine the grasp compatibility of an object pose.
        This heuristic simply punishes poses for which the object would fall towards the palm 
        of the robot and for which the gripper is in collision.
    """

    def __init__(self, manip):
        """
            Create a new instance of SimpleGraspCompatibility.
            ---------
            Arguments
            ---------
            manip, OpenRAVE manipulator - manipulator used for placing
        """
        self._manip = manip
        self._palm_dir = None  # stores direction pointing away from the palm in object frame
        self._gravity = np.array([0, 0, -1])

    def set_grasp_info(self, grasp_tf, grasp_config):
        """
            Sets information about the grasp of the object.
            ---------
            Arguments
            ---------
            grasp_tf, numpy array of shape (4,4) - pose of object relative to end-effector (in eef frame)
            hand_config, numpy array of shape (d_h,) - grasp configuration of the hand
        """
        inv_grasp_tf = utils.inverse_transform(grasp_tf)
        self._palm_dir = np.dot(inv_grasp_tf[:3, :3], self._manip.GetDirection())

    def compute_compatibility(self, node):
        """
            Computes the grasp compatibility for the given pose.
            ---------
            Arguments
            ---------
            node, SE3Hierarchy.SE3HierarchyNode or numpy array of shape (4, 4) - partition or pose to evaluate
            -------
            Returns
            -------
            val, float - a value in [0, 1] representing compatibility. The larger the better.
        """
        if type(node) == SE3Hierarchy.SE3HierarchyNode:
            pose = node.get_representative_value()
        else:
            pose = node  # we got a pose as argument
        out_of_palm_world_dir = np.dot(pose[:3, :3], self._palm_dir)
        dot_product = np.clip(np.dot(out_of_palm_world_dir, self._gravity) + 1.0, 0.0, 1.0)
        return dot_product


class ReachableGraspCompatibility(object):
    """
        This heuristic rates a node based the compatibility of the poses of this node with
        a given grasp. It combines the SimpleGraspCompatibility heuristic with a reachability map
        to estimate how likely it is that a given pose may be reached with the set grasp.
    """
    # TODO
    def __init__(self, manip, reachability_map):
        """
            Create new ReachableGraspCompatbility.
            ---------
            Arguments
            ---------
            manip, OpenRAVE manipulator
            reachability_map, reachability.ReachabilityMap - map to query reachability of manip
        """
        self._grasp_compatibility = SimpleGraspCompatibility(manip)
        self._reachability_map = reachability_map
        self._grasp_in_of = None

    def set_grasp_info(self, grasp_tf, grasp_config):
        """
            Sets information about the grasp of the object.
            ---------
            Arguments
            ---------
            grasp_tf, numpy array of shape (4,4) - pose of object relative to end-effector (in eef frame)
            hand_config, numpy array of shape (d_h,) - grasp configuration of the hand
        """
        self._grasp_in_of = utils.inverse_transform(grasp_tf)
        self._grasp_compatibility.set_grasp_info(grasp_tf, grasp_config)

    def is_feasible_partition(self, node):
        """
            Return True if there might be a reachable pose in the SE(3) partition descibed by node, 
            else False. In other words, if this function returns False, there is no possible reachable solution
            in the hierarchy rooted at node.
            ---------
            Arguments
            ---------
            node, SE3HierarchyNode
        """
        grasp_quat_pose = self.get_grasp_quat_pose(node.get_representative_value())
        cart_radius, so3_radius = node.get_max_radii()
        cart_dist, so3_dist = self._reachability_map.query(np.array([grasp_quat_pose]))
        cart_grid_d, so3_grid_d = self._reachability_map.get_dispersion()
        # assuming that the reachability map is sampled sufficiently densely,
        # we can filter partitions based on their distance to the closest reachable sample
        if cart_dist - cart_grid_d - cart_radius > 0.0:
            return False
        if so3_dist - so3_radius - so3_grid_d > 0.0:
            return False
        return True

    def get_grasp_quat_pose(self, obj_pose):
        """
            Return grasp pose in world frame in position-quaternion form.
            ---------
            Arguments
            ---------
            obj_pose, numpy array of shape (4, 4) - object pose in world frame
            -------
            Returns
            -------
            grasp_quat_pose, numpy array of shape (7,) - (x, y, z, w, ix, jy, kz)
        """
        grasp_pose = np.dot(obj_pose, self._grasp_in_of)
        grasp_quat_pose = np.empty(7)
        grasp_quat_pose[:3] = grasp_pose[:3, 3]
        grasp_quat_pose[3:] = orpy.quatFromRotationMatrix(grasp_pose[:3, :3])
        return grasp_quat_pose

    def compute_compatibility(self, node):
        """
            Computes the grasp compatibility for the given pose.
            ---------
            Arguments
            ---------
            node, SE3Hierarchy.SE3HierarchyNode - partition to evaluate
            -------
            Returns
            -------
            val, float - a value in [0, 1] representing compatibility. The larger the better.
        """
        # quat_pose = self.get_grasp_quat_pose(node.get_representative_value())
        # cart_radius, so3_radius = node.get_max_radii()
        # cart_dist, so3_dist = self._reachability_map.query([quat_pose])
        # TODO figure out how to compute a score from this
        # TODO maybe like on the white board
        # TODO do we really need to take reachability as score into account? This is only relevant for poses on the interface
        # TODO SimpleGraspCompatibility is not the best, should we really return it here?
        return 0.0


class PlacementHeuristic(object):
    """
        Implements a heuristic for placement. It consists of three components: a collision cost, a stability cost and
        a grasp compatibility cost. The collision cost punishes nodes that are in collision. The stability cost
        evaluates whether a given node is suitable for releasing an object so that it fall towards a good placement pose.
        The grasp compatibility cost evalutes whether the robot hand would interfere with placing/dropping the object,
        and whether the poses associated with node are reachable for an arm.
        The grasp compatibility cost is optional and only active if a manipulator is passed on construction.
    """

    def __init__(self, env, scene_sdf, object_base_path, robot_name=None, manip_name=None,
                 gripper_file=None, rmap=None, vol_approx=0.005):
        """
            Creates a new PlacementHeuristic
            ---------
            Arguments
            ---------
            env, OpenRAVE environment
            scene_sdf, SceneSDF of the environment
            object_base_path, string - path to object files
            robot_name, string (optional) - name of OpenRAVE robot used to place the object.
            manip_name, string (optional) - name of OpenRAVE manipulator used to place the object.
            gripper_file, string (optional, required if robot_name provided) - filename of OpenRAVE model of the gripper
            rmap, reachability.ReachabilityMap - reachability map for the robot
            vol_approx, float - size of octree cells for intersection costs
        """
        self._env = env.CloneSelf(orpy.CloningOptions.Bodies)
        self._ode_colchecker = orpy.RaveCreateCollisionChecker(self._env, 'ode')
        # self._fcl_colchecker = orpy.RaveCreateCollisionChecker(self._env, 'fcl_')
        self._env.SetCollisionChecker(self._ode_colchecker)  # TODO we probably wanna have fcl here -> kinbody_octree gets its own env
        self._obj_name = None
        self._model_name = None
        self._kinbody = None
        self._scene_sdf = scene_sdf
        self._kinbody_octree = None
        self._gripper_octree = None
        self._gripper = None
        self._grasp_compatibility = None
        self._inv_grasp_tf = None
        self._grasp_config = None
        filter_io = po_mod.PreferenceFilterIO(object_base_path)
        self._stability_function = SimplePlacementQuality(env, filter_io, scene_sdf)  # TODO make settable as parameter
        if robot_name is not None:
            robot = self._env.GetRobot(robot_name)
            if manip_name is None:
                manip = robot.GetActiveManipulator()
            else:
                manip = robot.GetManipulator(manip_name)
            # self._grasp_compatibility = SimpleGraspCompatibility(manip, gripper_file, vol_approx)  # TODO make settable as parameters
            if rmap != None:
                self._grasp_compatibility = ReachableGraspCompatibility(manip, reachability_map=rmap)
            # We could also move the gripper collision into the GraspCompatilibity function, but this way we have a clear separation
            # between hard constraint (collision-free) and soft constraint (preferred orientation)
            tenv = orpy.Environment()  # TODO do we really need a separate environment here?
            tenv.Load(gripper_file)
            self._gripper = tenv.GetRobots()[0]
            self._gripper_octree = robot_sdf.RobotOccupancyOctree(vol_approx, self._gripper)
        self._vol_approx = vol_approx
        self._parameters = {
            'plcmt_weight': 1.0,
            'col_weight': 1.0,
            'grasp_weight': 0.4,
         }
        # self._env.SetViewer('qtcoin')

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
        if len(self._kinbody.GetLinks()) != 1:
            raise ValueError("Could not set target object " + obj_name +
                             " because it has an invalid number of links (must be exactly 1)")
        self._kinbody_octree = kinbody_sdf.OccupancyOctree(self._vol_approx, self._kinbody.GetLinks()[0])
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

    def evaluate_stability(self, node, return_details=False):
        """
            Evalute the stability cost function for the given pose or node.
            ---------
            Arguments
            ---------
            node, SE3Hierarchy.SE3HierarchyNode or numpy array of shape (4, 4) - partition or pose to evaluate
            return_details, bool - if True, return additionaly information computed
                by stability function. What values are returned, depends on the stability
                function.
        """
        if type(node) == SE3Hierarchy.SE3HierarchyNode:
            pose = node.get_representative_value()
        else:
            pose = node  # we got a pose as argument
        self._kinbody.SetTransform(pose)
        return self._stability_function.compute_quality(node, return_details)

    def evaluate_collision(self, node):
        """
            Evalute the collision cost function for the given pose.
        """
        if type(node) == SE3Hierarchy.SE3HierarchyNode:
            pose = node.get_representative_value()
            self._kinbody.SetTransform(pose)
            dist, vdir, _ = self._kinbody_octree.compute_max_penetration(self._scene_sdf, b_compute_dir=True)
            if self._gripper_octree is not None:
                robot_pose = np.dot(pose, self._inv_grasp_tf)
                rdist, rdir, _, _ = self._gripper_octree.compute_max_penetration(robot_pose, self._grasp_config,
                                                                                 self._scene_sdf, b_compute_dir=True)
                if rdist < dist:
                    dist = rdist
                    vdir = rdir
            if dist < 0.0:
                node_bounds = node.get_bounds()
                box_extents = (node_bounds[:3, 1] - node_bounds[:3, 0]) / 2.0
                dir_norm = np.linalg.norm(vdir)
                if dir_norm > 0.0:
                    vdir = vdir / np.linalg.norm(vdir)
                else:
                    # in some rare cases on edges we might not get a direction, so just be conservative
                    shortest_dim = np.argmin(box_extents)
                    vdir = np.zeros((3,))
                    vdir[shortest_dim] = 1
                non_zero = np.abs(vdir) > 1e-9
                ts = np.abs(box_extents[non_zero] / vdir[non_zero])
                t = np.min(ts)
                assert(t > 0.0)
                score = 1.0 - np.abs(dist) / t
                rospy.logdebug("Collision heuristic: signed distance: " + str(dist) + ", t: " + str(t) + " vdir: " + str(vdir))
            else:
                score = 1.0
        else:
            pose = node  # we got a pose as argument
            self._kinbody.SetTransform(pose)
            # We get a pose as argument when we are evaluating a single pose, which is currently only
            # done for computing a constraint in the leaf stage.
            _, _, _, score, _ = self._kinbody_octree.compute_intersection(self._scene_sdf)
            rob_col = 0.0
            if self._gripper_octree is not None:
                robot_pose = np.dot(pose, self._inv_grasp_tf)
                _, _, _, rob_col = self._gripper_octree.compute_intersection(robot_pose, self._grasp_config, self._scene_sdf)
            score += rob_col
        # self._env.SetCollisionChecker(self._fcl_colchecker)
        return score

    def evaluate_grasp_compatibility(self, node):
        """
            Evalute the grasp compatibility cost for the given node or pose.
        """
        if type(node) == SE3Hierarchy.SE3HierarchyNode:
            pose = node.get_representative_value()
        else:
            pose = node  # we got a pose as argument
        if self._grasp_compatibility:
            self._kinbody.SetTransform(pose)
            return self._grasp_compatibility.compute_compatibility(pose)
        return 0.0

    def evaluate(self, node):
        """
            Evalutes the objective function for the given node (must be SE3HierarchyNode).
            The evaluation of the node is performed based on its representative.
        """
        assert(self._kinbody_octree)
        assert(self._scene_sdf)
        if node.cached_value is not None:
            rospy.logdebug("Node has cached value. Returning cached value.")
            return node.cached_value
        if self._grasp_compatibility is not None:
            if not self._grasp_compatibility.is_feasible_partition(node):
                # TODO need to update number of leaf nodes etc
                return -100000.0 # TODO decide what to return in this case
        # self._env.SetViewer('qtcoin')
        # self._handle = self.visualize_node(node)  # TODO delete
        # compute the collision cost for this pose
        col_val = self.evaluate_collision(node)
        stability_val = self.evaluate_stability(node)
        grasp_val = self.evaluate_grasp_compatibility(node)
        total_value = self._parameters['plcmt_weight'] * stability_val + self._parameters['col_weight'] * col_val + \
            self._parameters['grasp_weight'] * grasp_val
        # TODO shouldn't normalize by grasp weight if there is no grasp
        total_value /= (self._parameters['plcmt_weight'] + self._parameters['col_weight'] +
                        self._parameters['grasp_weight'] * (self._grasp_compatibility is not None))
        rospy.logdebug("Evaluated node %s. Collision: %f, Stability: %f, Grasp: %f, Total value: %f" %
                       (str(node.get_id()), col_val, stability_val, grasp_val, total_value))
        node.cached_value = total_value
        return total_value

    @staticmethod
    def is_better(val_1, val_2):
        """
            Returns whether val_1 is better than val_2
        """
        return val_1 > val_2

    def set_grasp_info(self, grasp_tf, grasp_config):
        """
            Sets information about the grasp of the object.
            ---------
            Arguments
            ---------
            grasp_tf, numpy array of shape (4,4) - pose of object relative to end-effector (in eef frame)
            hand_config, numpy array of shape (d_h,) - grasp configuration of the hand
        """
        if self._grasp_compatibility:
            self._grasp_compatibility.set_grasp_info(grasp_tf, grasp_config)
        self._grasp_config = grasp_config
        self._inv_grasp_tf = utils.inverse_transform(grasp_tf)

    def visualize_node(self, node):
        pose = node.get_representative_value()
        self._kinbody.SetTransform(pose)
        node_bounds = node.get_bounds()
        center = 0.5 * (node_bounds[:3, 0] + node_bounds[:3, 1])
        extents = 0.5 * (node_bounds[:3, 1] - node_bounds[:3, 0])
        return self._env.drawbox(center, extents, [0.0, 0.0, 1.0, 0.3])


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
            self._cartesian_range = cartesian_box[1] - cartesian_box[0]
            bfs = hierarchy.get_branching_factors(depth)
            self._child_dimensions = self._cartesian_range / bfs[:3]
            self._child_cache = {}
            self._white_list = None
            self.cached_value = None

        def get_random_node(self):
            """
                Return a random child node, as required by optimizers defined in
                the optimization module. This function simply calls get_random_child().
            """
            return self.get_random_child()

        def get_random_child(self):
            """
                Returns a randomly selected child node from the white list of this node.
            """
            if self._white_list is None:
                self._white_list = self._hierarchy.generate_white_list(self._depth)
            if self._depth == self._hierarchy._max_depth:
                return None
            if len(self._white_list) > 0:
                random_child = random.sample(self._white_list, 1)[0]
                return self.get_child_node(random_child)
            return None

        def get_random_neighbor(self, node):
            """
                Returns a randomly selected neighbor of the given child node.
                ---------
                Arguments
                ---------
                node has to be a child of this node.
                -------
                Returns
                TODO support white list
            """
            random_dir = np.array([np.random.randint(-1, 2),
                                   np.random.randint(-1, 2),
                                   np.random.randint(-1, 2)])
            child_id = node.get_id(relative_to_parent=True)
            bfs = self._hierarchy.get_branching_factors(self._depth)
            max_ids = bfs[:3] - 1
            # TODO support white list
            neighbor_id = np.zeros(5, np.int)
            neighbor_id[:3] = np.clip(child_id[:3] + random_dir, 0, max_ids)
            so3_key = so3hierarchy.get_random_neighbor(node.get_so3_key())
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

        def get_depth(self):
            """
                Return the depth of this node.
            """
            return self._depth

        def get_children(self):
            """
                Return a generator that allows to iterate over all children.
            """
            local_keys = self._hierarchy.get_white_list()
            for lkey in local_keys:
                child = self.get_child_node(lkey)
                yield child

        def get_num_children(self):
            """
                Return the number of children this node has.
            """
            if not self.is_leaf():
                bfs = self._hierarchy.get_branching_factors(self._depth)
                return np.multiply.reduce(bfs)
            return 0

        def get_num_leaves_in_branch(self):
            """
                Return the number of leaves in the branch of the hierarchy rooted at this node.
            """
            # TODO if this is turns out to be a resource hog, we could cache the result
            num_leaves = 1
            for d in xrange(self._hierarchy.get_depth(), self._depth, -1):
                # calculate the number of children a node on level d - 1 has
                bfs = self._hierarchy.get_branching_factors(d - 1)
                num_leaves *= np.multiply.reduce(bfs)
            return num_leaves

        def get_representative_value(self, rtype=0):
            """
                Returns a point in SE(3) that represents this cell, i.e. the center of this cell
                Note that for the root, None is returned.
                @param rtype - Type to represent point (0 = 4x4 matrix,
                                                        1 = [x, y, z, theta, phi, psi] (Hopf coordinates),)
                                                        2 = [x, y, z, w, ix, jy, kz] (pos, quaternion)
            """
            if self._depth == 0:  # the root does not have a representative
                return None
            position = self._cartesian_box[0] + self._cartesian_range / 2.0
            if rtype == 0:  # return a matrix
                quaternion = so3hierarchy.get_quaternion(self._so3_key)
                matrix = transformations.quaternion_matrix(quaternion)
                matrix[:3, 3] = position
                return matrix
            elif rtype == 1:  # return array with position and hopf coordinates
                result = np.empty(6)
                result[:3] = position
                result[3:] = so3hierarchy.get_hopf_coordinates(self._so3_key)
                return result
            elif rtype == 2:  # return np.array position and quaternion
                quat = so3hierarchy.get_quaternion(self._so3_key)
                return np.array([position[0], position[1], position[2], quat[0], quat[1], quat[2], quat[3]])
            raise RuntimeError("Return types different from matrix are not implemented yet!")

        def get_white_list(self):
            """
                Return white list.
            """
            return self._white_list

        def get_white_list_length(self):
            """
                Return length of white list. This is the number of children that still
                can be sampled from this node.
            """
            if self._white_list is not None:
                return len(self._white_list)
            return self.get_num_children()

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

        def get_bounds(self):
            """
                Return workspace range represented by this node.
                -------
                Returns
                -------
                bounds, numpy array of shape (6, 2) - each row represents min and max value for [x, y, z, theta, phi, psi]
            """
            bounds = np.empty((6, 2))
            bounds[:3, 0] = self._cartesian_box[0]
            bounds[:3, 1] = self._cartesian_box[1]
            bounds[3:, :] = so3hierarchy.get_hopf_coordinate_range(self._so3_key)
            return bounds

        def is_leaf(self):
            """
                Return whether this node is a leaf or not.
            """
            return self._depth == self._hierarchy.get_depth()

        def get_max_radii(self):
            """
                Return the maximum cartesian and rotational distance any pose within
                this partition can be from the center.
                -------
                Returns
                -------
                cartesian distance, float
                SO(3) distance, float
            """
            return np.linalg.norm(self._cartesian_box[1] - self._cartesian_box[0]) / 2.0, so3hierarchy.get_dispersion_bound(key=self._so3_key)

    def __init__(self, bounding_box, root_edge_length, cart_branching, depth):
        """
            Creates a new hierarchical grid on SE(3), i.e. R^3 x SO(3)
            ---------
            Arguments
            ---------
            bounding_box, (min_point, max_point), where min_point and max_point are numpy arrays of length 3
                            describing the edges of the bounding box this grid should cover
            root_edge_length, float - desired edge length of first level cartesian partitions
            cart_branching - branching factor for cartesian child partitions from depth 2 onwards
            depth - maximal depth of the hierarchy
        """
        self._bounding_box = bounding_box
        self._cart_branching = cart_branching
        self._root_edge_length = root_edge_length
        assert(self._root_edge_length > 0.0)
        assert(self._cart_branching > 0)
        self._max_depth = depth
        self._root = self.SE3HierarchyNode(cartesian_box=bounding_box,
                                           so3_key=so3hierarchy.get_root_key(),
                                           depth=0,
                                           hierarchy=self)

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
        # append elements of child id to individual elements of global id
        return tuple((parent_id[i] + (child_id[i],) for i in xrange(5)))

    def generate_white_list(self, depth):
        """
            Generates a new white list for node at the given depth.
            ---------
            Arguments
            ---------
            depth, int - depth of the node
            ---------
            Returns
            ---------
            white_list, set - a set of all possible child ids (relative to parent)
        """
        bfs = self.get_branching_factors(depth)
        return set(itertools.product(*map(xrange, bfs)))

    def get_branching_factors(self, depth):
        """
            Return branching factors at the given depth.
        """
        bfs = np.empty((5,), dtype=int)
        if depth == 0:
            # for the root compute special cartesian branching factor
            edge_lengths = self._bounding_box[1] - self._bounding_box[0]
            # determine the shortest edge of the workspace
            shortest_dim = np.argmin(edge_lengths)
            # we want along this dimension at least 1 node and ideally as many as there fit with edge length self._root_edge_length
            n_0 = edge_lengths[shortest_dim] / self._root_edge_length
            bfs[shortest_dim] = 1 if n_0 < 1.0 else int(np.round(n_0))
            e_0 = edge_lengths[shortest_dim] / bfs[shortest_dim]
            # Now select the number of cells for the other dimensions such that the cells' edge lengths are most similar
            other_dims = np.nonzero(np.array(range(3)) != shortest_dim)[0]
            bfs[other_dims] = map(int, np.round((edge_lengths[other_dims] / e_0)))
        else:
            bfs[:3] = self._cart_branching
        bfs[3:] = so3hierarchy.get_branching_factors(depth)
        return bfs

    def get_root(self):
        return self._root

    def get_depth(self):
        return self._max_depth

    def get_bounds(self):
        """
            Return workspace represented by hierarchy.
            -------
            Returns
            -------
            bounds, numpy array of shape (6, 2) - each row represents min and max value for [x, y, z, theta, phi, psi]
        """
        return self._root.get_bounds()


class PlacementGoalPlanner(GoalHierarchy):
    """This class allows to search for object placements in combination with
        the FreeSpaceProximitySampler.
    """
    class PlacementResult(GoalHierarchy.GoalHierarchyNode):
        """
            Represents the result of a placement call.
            See GoalHierarhcy.GoalHierarchyNode in sampler module for docs.
        """

        def __init__(self, hierarchy_node, quality_value):
            super(PlacementGoalPlanner.PlacementResult, self).__init__(None)
            self._hierarchy_node = hierarchy_node
            self._quality_value = quality_value
            self._valid = False
            self._bgoal = False
            self.obj_pose = hierarchy_node.get_representative_value()
            self._was_evaluated = False

        def is_valid(self):
            assert(self._was_evaluated)
            return self._valid

        def is_goal(self):
            assert(self._was_evaluated)
            return self._bgoal

        def hierarchy_info_str(self):
            return str(self._hierarchy_node.get_id())

        def get_num_possible_children(self):
            return self._hierarchy_node.get_num_children()

        def get_num_possible_leaves(self):
            return self._hierarchy_node.get_num_leaves_in_branch()

        def get_hashable_label(self):
            return str(self._hierarchy_node.get_id())

        def get_local_hashable_label(self):
            return self._hierarchy_node.get_id(relative_to_parent=True)

        def get_label(self):
            return self._hierarchy_node.get_id()

        def get_depth(self):
            return self._hierarchy_node.get_depth()

        def get_additional_data(self):
            # TODO here we could return sth like an open-hand policy
            return self.obj_pose

        def is_extendible(self):
            return not self._hierarchy_node.is_leaf()

        def get_quality(self):
            return self._quality_value

        def get_white_list(self):
            return self._hierarchy_node.get_white_list()

    ############################ PlacementGoalPlanner methods ############################
    def __init__(self, base_path,
                 env, scene_sdf, robot_name=None, manip_name=None, urdf_file_name=None, reachability_file=None,
                 urdf_path=None, gripper_file=None, visualize=False):
        """
            Creates a PlacementGoalPlanner
            ---------
            Arguments
            ---------
            base_path, string - Path where object data can be found
            env, OpenRAVE environment
            scene_sdf, SceneSDF - SceneSDF of the OpenRAVE environment
            robot_name, string (optional) - name of the robot to use for placing
            manip_name, string (optional) - in addition to robot name, name of the manipulator to use
            urdf_file_name, string (optional) - Filename of a URDF description of the robot
            reachability_file, string (optional, required if robot_name is not None) - Filename of reachability map for the robot
            urdf_path, string (optional) - path to urdf descriptions of objects (needed for physics)
            gripper_file, string (optional) - Filename of an OpenRAVE model of the robot's gripper
            @param visualize If true, the internal OpenRAVE environment is set to be visualized
        """
        self._hierarchy = None
        self._root = None
        self._env = env
        if robot_name and manip_name is None:
            manip_name = robot_name.GetActiveManipulator().GetName()
        robot_interface = None
        rmap = None
        if robot_name:
            robot_interface = leafstage.RobotInterface(env, robot_name=robot_name,
                                                       rmap_file=reachability_file,
                                                       manip_name=manip_name, 
                                                       urdf_file_name=urdf_file_name)
            rmap = robot_interface.get_reachability_map()
        self._placement_heuristic = PlacementHeuristic(
            env, scene_sdf, base_path, robot_name=robot_name, manip_name=manip_name, gripper_file=gripper_file,
            rmap=rmap)
        self._optimizer = optimization.StochasticOptimizer(self._placement_heuristic)
        self._bf_optimizer = optimization.BruteForceOptimizer(self._placement_heuristic)
        # self._optimizer = optimization.StochasticGradientDescent(self._objective_function)
        # TODO replace the leaf_stage with BayesOpt on Physics?
        # TODO integrate new physics model
        self._leaf_stage = leafstage.DefaultLeafStage(self._env,
                                                      self._placement_heuristic.evaluate_stability,
                                                      self._placement_heuristic.evaluate_collision,
                                                      robot_interface)
        self._placement_volume = None
        self._parameters = {'root_edge_length': 0.2, 'cart_branching': 4, 'max_depth': 4, 'num_iterations': 100}
        self._initialized = False

    def set_placement_volume(self, workspace_volume):
        """
            Sets the workspace volume in world frame in which the planner shall search for a placement pose.
            @param workspace_volume - (min_point, max_point), where both are np.arrays of length 3
        """
        self._placement_volume = workspace_volume
        self._placement_heuristic.set_placement_volume(workspace_volume)
        self._initialized = False

    def sample(self, depth_limit, post_opt=True):
        """ Samples a placement configuration from the root level. """
        if not self._initialized:
            self._initialize()
        return self.sample_warm_start(self.get_root(), depth_limit, post_opt=post_opt)

    def sample_warm_start(self, hierarchy_node, depth_limit, post_opt=False):
        """ Samples a placement configuration from the given node on. """
        if not self._initialized:
            self._initialize()
        num_iterations = self._parameters['num_iterations']
        best_val = None
        best_node = None
        current_node = hierarchy_node._hierarchy_node
        start_depth = current_node.get_depth()
        if start_depth + depth_limit > self.get_max_depth():
            depth_limit = self.get_max_depth() - start_depth
            rospy.logwarn("Specified depth limit exceeds maximal search depth. Capping search depth.")
        for depth in xrange(start_depth, start_depth + depth_limit):
            rospy.logdebug("Searching for placement pose on depth %i" % depth)
            if num_iterations < current_node.get_white_list_length():
                best_val, best_node = self._optimizer.run(current_node, num_iterations)
            else:
                rospy.loginfo("Few nodes left, switching to brute force.")
                best_val, best_node = self._bf_optimizer.run(current_node)
            current_node = best_node
        # we are done with searching for a good node in the hierarchy
        result = PlacementGoalPlanner.PlacementResult(best_node, best_val)
        # we are at a leaf, perform the leaf stage
        if result.is_leaf() and post_opt:
            self._leaf_stage.post_optimize(result)
        # next ask the leaf stage (also if the result is not a leaf), to evaluate the node
        self._leaf_stage.evaluate_result(result)  # this sets goal, valid flags and optionally arm configuration
        return result

    def load_hand(self, hand_path, hand_cache_file, hand_config_file, hand_ball_file):
        """ Does nothing. """
        # TODO The placement model (physics engine) will take the hand into account, so it will have
        # TODO to be set here.
        pass

    def set_object(self, obj_id, model_id=None):
        """ Set the object.
            @param obj_id String identifying the object.
            @param model_id (optional) Name of the model data. If None, it is assumed to be identical to obj_id
        """
        self._placement_heuristic.set_target_object(obj_id, model_id)
        self._leaf_stage.set_object(obj_id)
        self._initialized = False

    def setup(self, obj_name, grasp_tf, grasp_config, model_id=None):
        """
            Setup the planner for integrated planning, i.e. with arm configurations.
            ---------
            Arguments
            ---------
            obj_name, string - the name of the object
            grasp_tf, numpy array of shape (4, 4) - transformation matrix from object frame to end-effector frame,
                describing the pose of the object relative to the eef
            grasp_config, numpy array of shape (d_h,) - grasp configuration of the gripper/hand
            model_id, string (optional) - Name of the model data. If None, it is assumed to be identical to obj_id
        """
        self.set_object(obj_name, model_id=model_id)
        self._leaf_stage.set_grasp_info(grasp_tf, grasp_config)
        self._placement_heuristic.set_grasp_info(grasp_tf, grasp_config)

    def set_max_iter(self, iterations):
        self._parameters['num_iterations'] = iterations

    def get_max_depth(self):
        return self._parameters['max_depth']

    def get_root(self):
        if not self._initialized:
            self._initialize()
        return self._root

    def set_parameters(self, **kwargs):
        self._initialized = False
        for (key, value) in kwargs.iteritems():
            self._parameters[key] = value

    def _initialize(self):
        if self._placement_volume is None:
            raise ValueError("Could not intialize as there is no placement volume available")
        if self._placement_heuristic.get_target_object() is None:
            raise ValueError("Could not intialize as there is no placement target object available")
        self._hierarchy = SE3Hierarchy(self._placement_volume,
                                       self._parameters['root_edge_length'],
                                       self._parameters['cart_branching'],
                                       self._parameters['max_depth'])
        self._root = PlacementGoalPlanner.PlacementResult(self._hierarchy.get_root(), -1.0*float('inf'))
        self._root._was_evaluated = True  # root is always invalid and not a goal
        self._initialized = True
