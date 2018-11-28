#!/usr/bin/env python
import collections
import scipy.spatial
import numpy as np


def merge_faces(chull, min_normal_similarity=0.24):
    """
        Merge adjacent faces of the convex hull if they have similar normals.
        ---------
        Arguments
        ---------
        chull - convex hull computed by scipy.spatial.ConvexHull
        min_normal_similarity(optional), float - if the angle between the normals of two adjacent face is smaller than
            this threshold, the faces are merged
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
                if np.dot(chull.equations[candidate_idx, :3], cluster_normal) >= np.cos(min_normal_similarity):
                    if face_clusters[candidate_idx] == -1:  # check if candidate is unclustered yet
                        # add the vertices of this face to the cluster
                        current_cluster[1].update(chull.simplices[candidate_idx])
                        face_clusters[candidate_idx] = cluster_id  # assign the cluster to the face
                        # add its neighbors to our extension candidates
                        cluster_candidates.extend(chull.neighbors[candidate_idx])
            cluster_id += 1
        face_idx += 1
    return clusters, face_clusters, cluster_id


def compute_hull_distance(convex_hull, point):
    """
        Compute the signed distance of the given point
        to the closest edge of the given 2d convex hull.
        A negative distance means the point is inside, a positive outside.
        ---------
        Arguments
        ---------
        convex_hull, ConvexHull - a convex hull of 2d points computed using scipy's wrapper
            for QHull.
        point, numpy array of shape (2,) - a 2d point to compute the distance for
        ---------
        Returns
        ---------
        distance, float - the signed distance to the closest edge of the convex hull
        edge_id, int - the index of the closest edge
    """
    # min_distance = 0.0
    assert(convex_hull.points[0].shape == point.shape)
    assert(point.shape == (2,))
    interior_distance = float('-inf')
    exterior_distance = float('inf')
    closest_int_edge = 0
    closest_ext_edge = 0
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
            if interior_distance < -1.0 * edge_distance:
                interior_distance = -1.0 * edge_distance
                closest_int_edge = idx
        elif edge_distance < exterior_distance:
            exterior_distance = edge_distance
            closest_ext_edge = idx
    if exterior_distance == float('inf'):  # the point is inside the convex hull
        return interior_distance, closest_int_edge
    return exterior_distance, closest_ext_edge  # the point is outside the convex hull


def construct_plane_mesh(points3d, normal):
    """
        Constructs a mesh from the given planar 3d points.
        ---------
        Arguments
        ---------
        normal, numpy array of shape (3,) - normal of the plane
        points3d, numpy array of shape (n, 3) - n 3d points that lie in the plane
        -------
        Returns
        -------
        vertices, numpy array of shape (n, 3) - the same as points3d
        indices, numpy array of shape (m, 3) - each row represents a triangle
    """
    # project points to a plane
    mean_point = np.mean(points3d, axis=0)
    rel_points = points3d - mean_point
    projected_points = rel_points - np.dot(rel_points, normal.transpose())[:, np.newaxis] * normal
    axes = np.empty((3, 2))
    axes[:, 0] = projected_points[1] - projected_points[0]
    axes[:, 0] /= np.linalg.norm(axes[:, 0])
    axes[:, 1] = np.cross(normal, axes[:, 0])
    points_2d = np.dot(projected_points, axes)
    tri = scipy.spatial.Delaunay(points_2d)
    return points3d, tri.simplices
