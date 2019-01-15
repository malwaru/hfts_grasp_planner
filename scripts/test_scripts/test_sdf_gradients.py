#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf_mod
import hfts_grasp_planner.sdf.grid as grid_mod
import hfts_grasp_planner.utils as utils
import numpy as np
import mayavi.mlab
import rospy
import time

GDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/gradients/placement_exp_1.static.gdf.npy'
SDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/placement_exp_0.sdf.static.sdf'


def show_precomputed():
    gradient_field = grid_mod.VectorGrid.load(GDF_PATH)
    print gradient_field.vectors.shape
    mayavi_vec_f = mayavi.mlab.pipeline.vector_field(gradient_field.vectors[0],
                                                     gradient_field.vectors[1],
                                                     gradient_field.vectors[2])
    mayavi.mlab.pipeline.vector_cut_plane(mayavi_vec_f, plane_orientation="z_axes")
    mayavi.mlab.show()


def show_online_computed():
    sdf = sdf_mod.SDF.load(SDF_PATH)
    workspace = sdf.get_grid().get_aabb(bWorld=True)
    num_samples = 50
    x = np.linspace(workspace[0] + 0.05, workspace[3] - 0.05, num=num_samples)
    y = np.linspace(workspace[1] + 0.05, workspace[4] - 0.05, num=num_samples)
    z = np.linspace(workspace[2] + 0.05, workspace[5] - 0.05, num=num_samples)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    query_pos = np.array([xx, yy, zz]).T.reshape(-1, 3)
    print "Querying positions"
    start = time.time()
    values, gradients = sdf.get_distances_grad(query_pos)
    total_time = time.time() - start
    print "Query took %fs, that is %fs per position" % (total_time, total_time / pow(num_samples, 3))
    norms = np.linalg.norm(gradients, axis=1)
    print 'Min gradient norm: %f, max gradient norm: %f' % (np.min(norms), np.max(norms))
    xx_g, yy_g, zz_g = gradients.reshape(num_samples, num_samples, num_samples, 3).T
    mayavi_vec_f = mayavi.mlab.pipeline.vector_field(xx, yy, zz, xx_g, yy_g, zz_g)
    mayavi.mlab.pipeline.vector_cut_plane(mayavi_vec_f, plane_orientation="z_axes")
    # mayavi.mlab.volume_slice(xx, yy, zz, values.reshape(num_samples, num_samples, num_samples).T,
    #  slice_index=2, plane_orientation="z_axes")
    mayavi.mlab.show()


if __name__ == "__main__":
    rospy.init_node("TestSDFGradients")
    show_online_computed()
