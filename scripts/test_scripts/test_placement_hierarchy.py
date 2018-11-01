#! /usr/bin/python

import os
import time
import IPython
import itertools
import colormap
import openravepy as orpy
import numpy as np
import hfts_grasp_planner.placement.placement_planning as pp_module
import hfts_grasp_planner.placement.so3hierarchy as so3hierarchy
import hfts_grasp_planner.external.transformations as tfs


def quat_distance(prev_quat, quat):
    if prev_quat is None or quat is None:
        return 0.0
    return np.arccos(np.abs(np.dot(prev_quat, quat)))


def visualize_pose(env, pose, color=None):
    pos = pose[:, 3]
    z_axis = pose[:, 2]
    z_axis /= np.linalg.norm(z_axis)
    x_axis = pose[:, 0]
    x_axis /= np.linalg.norm(x_axis)
    z_length = 0.1
    x_length = 0.02
    arrow_tip = pos + z_length * z_axis
    arrow_color = [0.0, 0.0, 1.0, 1.0]
    if color is not None:
        arrow_color = color
    z_axis_handle = env.drawarrow(pos, arrow_tip, linewidth=0.001, color=arrow_color)
    arrow_color = [1.0, 0.0, 0.0, 1.0]
    if color is not None:
        arrow_color = color
    x_axis_handle = env.drawarrow(arrow_tip, arrow_tip + x_length * x_axis, linewidth=0.001, color=arrow_color)
    return [z_axis_handle, x_axis_handle]


def show_cluster(env, cluster, handles, colormap):
    s2_keys = itertools.product(range(4), range(4))
    s1_keys = itertools.product(range(2), range(2))
    # s2_keys = range(4)
    # s1_keys = range(2)
    # for i in range(8):
    all_keys = itertools.product(s2_keys, s1_keys)
    for rel_key in all_keys:
        child_key_0 = rel_key[0]
        child_key_1 = rel_key[1]
        if type(child_key_0) != tuple:
            child_key_0 = (child_key_0,)
        if type(child_key_1) != tuple:
            child_key_1 = (child_key_1,)
        key = ((cluster[0],) + child_key_0, (cluster[1],) + child_key_1)
        quat = so3hierarchy.get_quaternion(key)
        pose = tfs.quaternion_matrix(quat)
        color = colormap[cluster]
        lhandles = visualize_pose(env, pose, color=color)
        if cluster not in handles:
            handles[cluster] = []
        handles[cluster].extend(lhandles)


def run_through_so3hierarchy(env):
    # first create a colormap for the first level of the hierarhcy
    col_map = colormap.Colormap()
    cmap = col_map.get_cmap_rainbow()
    color_step_size = 1.0 / 72.0
    my_color_map = {}
    first_level_keys = itertools.product(range(12), range(6))
    for i, first_level_key in enumerate(first_level_keys):
        my_color_map[first_level_key] = cmap(i * color_step_size)
    # now create a drawing for each element of the hierarhcy
    handles = {}
    # s2_keys = itertools.product(range(12), range(4))
    # s1_keys = itertools.product(range(6), range(2))
    # all_keys = itertools.product(s2_keys, s1_keys)
    clusters = itertools.product(range(12), range(6))
    # clusters = [(0, 0), (0, 1), (1, 1), (1, 0),
    #             (5, 0), (5, 1), (4, 1), (4, 0),
    #             (3, 0), (3, 1)]
    # prev_quat = None
    # distances = []
    for cluster in clusters:
        cluster_key = ((cluster[0],), (cluster[1],))
        # quat = so3hierarchy.get_quaternion(cluster_key)
        show_cluster(env, cluster, handles, my_color_map)
        # distances.append(quat_distance(prev_quat, quat))
        # print cluster_key, distances[-1]
        # IPython.embed()
        raw_input("Press key for next cluster")
        # pose = orpy.matrixFromQuat(quat)
        # pose = tfs.quaternion_matrix(quat)
        # color = my_color_map[cluster]
        # lhandles = visualize_pose(env, pose, color=color)
        # if cluster not in handles:
        # handles[cluster] = []
        # handles[cluster].extend(lhandles)
        # prev_quat = quat
    # print "Mean distance is %f and std is %f" % (np.mean(distances[1:]), np.std(distances[1:]))
    # print "Max distance is %f and min distance is %f" % (np.max(distances[1:]), np.min(distances[1:]))
    # IPython.embed()
    return handles


def run_through_hierarchy(hierarchy, env, depth):
    root = hierarchy.get_root()
    handles = []
    for child in root.get_children():
        print child.get_id()
        transform = child.get_representative_value()
        handles.extend(visualize_pose(env, transform))
        print tfs.quaternion_from_matrix(transform)
    return handles


if __name__ == "__main__":
    env = orpy.Environment()
    # env.Load(os.path.dirname(__file__) + '/../data/crayola/objectModel.ply')
    env.SetViewer('qtcoin')
    placement_volume = (np.array([-1.0, -2.0, 0.5]), np.array([0.2, 0.8, 0.9]))
    depth = 1
    # hierarchy = pp_module.SE3Hierarchy(placement_volume, 4, 4)
    # handles = run_through_hierarchy(hierarchy, env, depth)
    handles = run_through_so3hierarchy(env)
    IPython.embed()
