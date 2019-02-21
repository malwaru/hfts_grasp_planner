#! /usr/bin/python

import IPython
import hfts_grasp_planner.sdf.core as sdf_mod
import hfts_grasp_planner.sdf.grid as grid_mod
import hfts_grasp_planner.utils as utils
import numpy as np
import openravepy as orpy
# import mayavi.mlab
import rospy
import time

ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/cabinet_high_clutter.xml'
GDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/gradients/placement_exp_1.static.gdf.npy'
SDF_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/sdfs/cabinet_high_clutter.static.sdf'


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
    print "Min value is %f and max value is %f" % (np.min(values), np.max(values))
    xx_g, yy_g, zz_g = gradients.reshape(num_samples, num_samples, num_samples, 3).T
    mayavi_vec_f = mayavi.mlab.pipeline.vector_field(xx, yy, zz, xx_g, yy_g, zz_g)
    mayavi.mlab.pipeline.vector_cut_plane(mayavi_vec_f, plane_orientation="z_axes")
    # mayavi.mlab.volume_slice(xx, yy, zz, values.reshape(num_samples, num_samples, num_samples).T,
    #  slice_index=2, plane_orientation="z_axes")
    mayavi.mlab.show()


def show_distance_field():
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
    print "Min value is %f and max value is %f" % (np.min(values), np.max(values))
    # xx_g, yy_g, zz_g = gradients.reshape(num_samples, num_samples, num_samples, 3).T
    # mayavi_vec_f = mayavi.mlab.pipeline.vector_field(xx, yy, zz, xx_g, yy_g, zz_g)
    # mayavi.mlab.pipeline.vector_cut_plane(mayavi_vec_f, plane_orientation="z_axes")
    # values = values.reshape((num_samples, num_samples, num_samples)).transpose()
    # mayavi.mlab.volume_slice(values, slice_index=1, plane_orientation="z_axes")
    mayavi.mlab.volume_slice(xx, yy, zz, values.reshape(num_samples, num_samples, num_samples).T,
                             slice_index=2, plane_orientation="z_axes")
    mayavi.mlab.show()


def show_distance_field_fixed_pos():
    sdf = sdf_mod.SDF.load(SDF_PATH)
    workspace = sdf.get_grid().get_aabb(bWorld=True)
    num_samples = 50
    x = np.linspace(workspace[0] + 0.05, workspace[3] - 0.05, num=num_samples)
    y = np.linspace(workspace[1] + 0.05, workspace[4] - 0.05, num=num_samples)
    z = np.linspace(workspace[2] + 0.05, workspace[5] - 0.05, num=num_samples)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    query_pos = np.array([xx, yy, zz]).T.reshape(-1, 3)
    print "Querying positions"
    interp = sdf.get_cuda_interpolator(query_pos)
    start = time.time()
    values, gradients = interp.chomps_smooth_distance(np.eye(4), 0.04)
    # values, gradients = interp.gradient(np.eye(4))
    total_time = time.time() - start
    print "Query took %fs, that is %fs per position" % (total_time, total_time / pow(num_samples, 3))
    norms = np.linalg.norm(gradients, axis=1)
    print 'Min gradient norm: %f, max gradient norm: %f' % (np.min(norms), np.max(norms))
    print "Min value is %f and max value is %f" % (np.min(values), np.max(values))
    xx_g, yy_g, zz_g = gradients.reshape(num_samples, num_samples, num_samples, 3).T
    mayavi_vec_f = mayavi.mlab.pipeline.vector_field(xx, yy, zz, xx_g, yy_g, zz_g)
    mayavi.mlab.pipeline.vector_cut_plane(mayavi_vec_f, plane_orientation="z_axes")
    # values = values.reshape((num_samples, num_samples, num_samples)).transpose()
    # mayavi.mlab.volume_slice(values, slice_index=1, plane_orientation="z_axes")
    # mayavi.mlab.volume_slice(xx, yy, zz, values.reshape(num_samples, num_samples, num_samples).T,
    #  slice_index=2, plane_orientation="z_axes")
    mayavi.mlab.show()


def compare_gradient_vals():
    sdf = sdf_mod.SDF.load(SDF_PATH)
    workspace = sdf.get_grid().get_aabb(bWorld=True)
    num_samples = 50
    x = np.linspace(workspace[0] + 0.05, workspace[3] - 0.05, num=num_samples)
    y = np.linspace(workspace[1] + 0.05, workspace[4] - 0.05, num=num_samples)
    z = np.linspace(workspace[2] + 0.05, workspace[5] - 0.05, num=num_samples)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    query_pos = np.array([xx, yy, zz]).T.reshape(-1, 3)
    print "Querying positions"
    interp = sdf.get_cuda_interpolator(query_pos)
    start = time.time()
    # values, gradients = interp.chomps_smooth_distance(np.eye(4), 0.04)
    cuda_f_values, cuda_f_gradients = interp.gradient(np.eye(4))
    total_time = time.time() - start
    print "GPU query took %fs, that is %fs per position" % (total_time, total_time / pow(num_samples, 3))
    start = time.time()
    cpu_values, cpu_gradients = sdf.get_distances_grad(query_pos, b_force_cpu=True)
    cpu_values = cpu_values.astype(np.float32)
    cpu_gradients = cpu_gradients.astype(np.float32)
    total_time = time.time() - start
    print "CPU query took %fs, that is %fs per position" % (total_time, total_time / pow(num_samples, 3))
    value_errors = np.abs(cpu_values - cuda_f_values)
    gradient_diffs = np.abs(cpu_gradients - cuda_f_gradients)
    gradient_errors = np.linalg.norm(cpu_gradients - cuda_f_gradients, axis=1)
    print "For fixed position cuda:"
    print "Mean value error is %f, min/max errors are %f, %f" % (
        np.mean(value_errors), np.min(value_errors), np.max(value_errors))
    print "Mean gradient error is %f, min/max errors are %f, %f" % (
        np.mean(gradient_errors), np.min(gradient_errors), np.max(gradient_errors))
    print "Diffs in gradients. Mean: %s, min: %s, max: %s" % (str(np.mean(gradient_diffs, axis=0)), str(
        np.min(gradient_diffs, axis=0)), str(np.max(gradient_diffs, axis=0)))
    print "Largest value error occurs at idx %i and largest gradient error at %i" % (
        np.argmax(value_errors), np.argmax(gradient_errors))
    IPython.embed()


def show_probe():
    sdf = sdf_mod.SDF.load(SDF_PATH)
    grid = sdf.get_grid()
    env = orpy.Environment()
    env.Load(ENV_PATH)
    body = orpy.RaveCreateKinBody(env, '')
    body.InitFromSpheres(np.array([[0, 0, 0, 0.01]]))
    body.SetName("PROBE")
    env.AddKinBody(body)
    env.SetViewer('qtcoin')
    tf = np.eye(4)
    tf[:3, 3] = [0.7, 1.0, 1.0]
    body.SetTransform(tf)

    def read_grad_val():
        pos = body.GetTransform()[:3, 3]
        _, val, grad = grid.get_cell_gradients_pos_cuda(pos)
        print "Value is %f, gradient is %s" % (val[0], str(grad[0]))
        end = pos + grad[0]
        return env.drawarrow(pos, end, linewidth=0.01)
    read_grad_val()
    IPython.embed()


if __name__ == "__main__":
    rospy.init_node("TestSDFGradients")
    # show_online_computed()
    # show_distance_field()
    # show_distance_field_fixed_pos()
    # compare_gradient_vals()
    show_probe()
