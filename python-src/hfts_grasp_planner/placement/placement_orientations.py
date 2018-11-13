import os
import numpy as np
import scipy.spatial
import hfts_grasp_planner.placement.chull_utils as chull_utils
import hfts_grasp_planner.utils as utils


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
                    raise IOError(
                        "Could not load filter for model %s. Invalid numpy array shape encountered." % model_name)
                for filter_desc in filter_descs:  # filter_desc is assumed to be (nx, ny, nz, angle)
                    new_filters.append(OrientationFilter(filter_desc[:3], filter_desc[3]))
                self._filters[model_name] = new_filters
                return new_filters
        return None


class PlacementOrientation(object):
    """
        A placement orientation describes a class of poses in which a rigid object
        can be placed. The placement orientation is defined by the face of the object's
        surface (technically by a face of its convex hull) on which the object is placed
        on a horizontal plane. The placement orientation describes a class of poses, since it
        may be translated on this horizontal plane or rotated around the z axis in anyway.

        Placement orientations can be computed using the compute_placement_orientations(..)
        function in this module.
    """

    def __init__(self, placement_face, body, com_distance3d, com_distance2d):
        """
            Create a new placement orientation.
            ---------
            Arguments
            ---------
            placement_face, numpy array of shape (n+1, 3), where the first entry is the normal of the placement face
                and the remaining n rows the vertices of the face - all in object frame.
            body, OpenRAVE kinbody - the object. Must have only one link!
            com_distance3d, float - distance of the center of mass to the placement plane
            com_distance2d, float - distance of the projected center of mass to the closest edge of the placement face
        """
        self.placement_face = placement_face
        self.body = body
        self.com_distance_3d = com_distance3d
        self.com_distance_2d = com_distance2d
        self.reference_tf = np.eye(4)  # tf from reference frame to object frame
        self.reference_tf[:3, 2] = -placement_face[0]  # z-axis
        self.reference_tf[:3, 0] = placement_face[2] - placement_face[1]  # x_axis
        assert(np.linalg.norm(self.reference_tf[:, 0]) > 0.0)
        self.reference_tf[:3, 0] /= np.linalg.norm(self.reference_tf[:3, 0])  # normalize x_axis
        self.reference_tf[:3, 1] = np.cross(self.reference_tf[:3, 2], self.reference_tf[:3, 0])  # y_axis
        self.reference_tf[:3, 3] = placement_face[1]  # position of first vertex
        self.inv_reference_tf = utils.inverse_transform(self.reference_tf)


def is_stable_placement_plane(plane, com, min_com_distance=0.0):
    """
        Return whether the specified plane can be used to stably place an object given its center of mass
        ---------
        Arguments
        ---------
        plane, numpy array of shape (N + 1, 3) - the placement plane to test
        com, numpy array of shape (3,) - center of mass defined in the same frame as plane
        min_com_distance, float - minimal distance that the projected com should have to the closest edge
            of the support polygon
        -------
        Returns
        -------
        true or false depending on whether it is stable or not
        dist2d, float - distance of the projected center of mass to any boundary
        min_dist2d, float minimal distance the projected center of mass can have
        dist3d, float - maximal distance the projected center of mass can have
    """
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
    dir_to_com = com - mean_point
    projected_com = com - np.dot(dir_to_com, plane[0]) * plane[0]
    dist3d = np.linalg.norm(com - projected_com)
    com2d = np.dot(projected_com, axes)
    # compute the convex hull of the projected points
    convex_hull = scipy.spatial.ConvexHull(points_2d)
    # accept the point if the projected center of mass is inside of the convex hull
    dist2d, _ = chull_utils.compute_hull_distance(convex_hull, com2d)
    best_dist = np.min(convex_hull.equations[:, -1])
    return dist2d < -1.0 * min_com_distance, dist2d, best_dist, dist3d


def compute_placement_orientations(body, user_filters=None, min_normal_similarity=0.01, min_com_distance=0.01):
    """
        Compute all placement orientations for the given body.
        ---------
        Arguments
        ---------
        body, OpenRAVE Kinbody - the body to compute placement orientations for. Must have only one link.
        user_filter(optional), list of OrientationFilter - orientations to filter out based on placement plane
        min_normal_similarity(optional), float - if the angle between the normals of two adjacent face is smaller than this
            threshold, the faces are merged
        min_com_distance(optional), float - minimal distance that the com projection needs to have from its closest
            boundary of the support polygon (placement face).
    """
    # first compute the convex hull of the body
    links = body.GetLinks()
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
    # merge faces
    clusters, _, _ = chull_utils.merge_faces(convex_hull, min_normal_similarity)
    placement_orientations = []
    # compute local center of mass
    tf = body.GetTransform()
    tf_inv = utils.inverse_transform(tf)
    local_com = np.dot(tf_inv[:3, :3], body.GetCenterOfMass()) + tf_inv[:3, 3]
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
        b_stable, dist2d, _, dist3d = is_stable_placement_plane(plane, local_com, min_com_distance)
        if not b_stable:
            continue
        # if this plane passed all filters, we accept it
        placement_orientations.append(PlacementOrientation(plane, body, dist3d, np.abs(dist2d)))
    return placement_orientations
