import hfts_grasp_planner.external.transformations as transformations
import hfts_grasp_planner.sdf.cuda_grid as cuda_grid
import scipy.ndimage
import numpy as np
import os
import math
import IPython


def gen_data():
    # data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0],
    #                   [2.0, 3.0, 4.0, 5.0, 6.0],
    #                   [3.0, 4.0, 5.0, 6.0, 7.0]],
    #                  [[4.0, 5.0, 6.0, 7.0, 8.0],
    #                   [5.0, 6.0, 7.0, 8.0, 9.0],
    #                   [6.0, 7.0, 8.0, 9.0, 10.0]]])
    data = np.zeros((2, 3, 5))
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                data[x, y, z] = x * 10 + (1 + y) * (1 + z)
    data = 1.0 / 100.0 * data
    return data.astype(np.float32)


def test_normal():
    data = gen_data()
    positions = np.array([[0.1, 0.5, 0.3], [0.5, 1.0, 1.0],
                          [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5], [0.0, 1.0, 1.0]], dtype=np.float32)
    tf_matrix = transformations.random_rotation_matrix().astype(np.float32)
    # tf_matrix = np.eye(4)
    tf_matrix[:3, 3] = [0.2, 0.1, 0.2]
    interp = cuda_grid.CudaInterpolator(data)
    out_values = interp.interpolate(positions, tf_matrix=tf_matrix)
    mapped_pos = np.dot(positions, tf_matrix[:3, :3].transpose()) + tf_matrix[:3, 3]
    scipy_out = scipy.ndimage.map_coordinates(data, mapped_pos.transpose() + 1.0, order=1, mode='nearest')
    x, y, z = np.where(data > 0)
    all_indices = np.array(zip(x, y, z))
    print out_values
    print scipy_out
    # print out_values - mapped_pos
    print out_values - scipy_out
    IPython.embed()


def compute_gradients(pos, data_reader, delta):
    query_pos = pos + np.array([delta, 0.0, 0.0], dtype=np.float32)
    px = data_reader(query_pos)
    # px = scipy.ndimage.map_coordinates(data, query_pos.transpose() + 1.0, order=1, mode='nearest')
    query_pos = pos - np.array([delta, 0.0, 0.0], dtype=np.float32)
    # mx = scipy.ndimage.map_coordinates(data, query_pos.transpose() + 1.0, order=1, mode='nearest')
    mx = data_reader(query_pos)
    query_pos = pos + np.array([0.0, delta, 0.0], dtype=np.float32)
    # py = scipy.ndimage.map_coordinates(data, query_pos.transpose() + 1.0, order=1, mode='nearest')
    py = data_reader(query_pos)
    query_pos = pos - np.array([0.0, delta, 0.0], dtype=np.float32)
    # my = scipy.ndimage.map_coordinates(data, query_pos.transpose() + 1.0, order=1, mode='nearest')
    my = data_reader(query_pos)
    query_pos = pos + np.array([0.0, 0.0, delta], dtype=np.float32)
    # pz = scipy.ndimage.map_coordinates(data, query_pos.transpose() + 1.0, order=1, mode='nearest')
    pz = data_reader(query_pos)
    query_pos = pos - np.array([0.0, 0.0, delta], dtype=np.float32)
    mz = data_reader(query_pos)
    # mz = scipy.ndimage.map_coordinates(data, query_pos.transpose() + 1.0, order=1, mode='nearest')
    xgrad = (px - mx) / (2.0 * delta)
    ygrad = (py - my) / (2.0 * delta)
    zgrad = (pz - mz) / (2.0 * delta)
    return np.array([xgrad, ygrad, zgrad]).transpose()


def test_gradients():
    data = gen_data()
    positions = np.array([[0.1, 0.5, 0.3], [0.5, 1.0, 1.0],
                          [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5], [0.0, 1.0, 1.0]], dtype=np.float32)
    # tf_matrix = transformations.random_rotation_matrix()
    tf_matrix = np.eye(4, dtype=np.float32)
    # tf_matrix[:3, 3] = [0.2, 0.1, 0.2]
    delta = 0.01
    interp = cuda_grid.CudaInterpolator(data)

    def read_scipy(pos):
        return scipy.ndimage.map_coordinates(data, pos.transpose() + 1.0, order=1, mode='nearest')

    def read_cuda(pos):
        return np.array(interp.interpolate(pos))

    scipy_values = read_scipy(positions)
    cuda_values, cuda_gradients = interp.gradient(positions, delta=delta)
    copied_val = np.array(cuda_values)
    cuda_values, cuda_gradients = interp.gradient(positions, delta=delta)
    print cuda_values - copied_val
    mapped_pos = np.dot(positions, tf_matrix[:3, :3].transpose()) + tf_matrix[:3, 3]
    scipy_gradients = compute_gradients(mapped_pos, read_scipy, delta)
    mixed_gradients = compute_gradients(mapped_pos, read_cuda, delta)
    # x, y, z = np.where(data > 0)
    # all_indices = np.array(zip(x, y, z))
    print cuda_values
    print scipy_values
    print cuda_values - scipy_values
    # gradients
    print "Cuda gradients"
    print cuda_gradients
    print "Scipy gradients"
    print scipy_gradients
    print cuda_gradients - scipy_gradients
    # mixed gradients
    print "Mixed gradients"
    print mixed_gradients
    print "Mixed - scipy gradients"
    print mixed_gradients - scipy_gradients
    print "Mixed - cuda gradients"
    print mixed_gradients - cuda_gradients
    IPython.embed()


def test_sum():
    data = gen_data()
    positions = np.array([[0.1, 0.5, 0.3], [0.5, 1.0, 1.0],
                          [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5], [0.0, 1.0, 1.0]])
    tf_matrix = np.eye(4)
    interp = cuda_grid.CudaStaticPositionsInterpolator(data, positions, np.eye(4), 1.0, 0.01)
    val = interp.sum(np.eye(4))
    print val
    scipy_out = scipy.ndimage.map_coordinates(data, positions.transpose() + 1.0, order=1, mode='nearest')
    print np.sum(scipy_out)
    print val - np.sum(scipy_out)


def sliced_interpolation(indices, data):
    # bilinear interpolation
    # perpare output values
    values = np.empty((indices.shape[0],))
    # 1. get integer z coordinates
    z_integer = indices[:, 2].astype(int)
    # for each z layer
    for z in z_integer:
        print z + 1
        indices_with_z = indices[:, 2].astype(int) == z
        # do bilinear interpolation
        values[indices_with_z] = scipy.ndimage.map_coordinates(
            data[:, :, z + 1], indices[indices_with_z, :2].transpose() + 1.0, order=1, mode="nearest")
    return values


def test_slices():
    data = gen_data()
    positions = np.array([[0.1, 0.5, 0.3], [0.5, 1.0, 1.0],
                          [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5], [0.0, 1.0, 1.0]])
    tf_matrix = transformations.random_rotation_matrix()
    # tf_matrix = np.eye(4)
    tf_matrix[:3, 3] = [0.2, 0.1, 0.0]
    interp = cuda_grid.CudaInterpolator(data, True)
    out_values = interp.interpolate(positions, tf_matrix=tf_matrix)
    mapped_pos = np.dot(positions, tf_matrix[:3, :3].transpose()) + tf_matrix[:3, 3]
    print mapped_pos
    scipy_out = sliced_interpolation(mapped_pos, data)
    # x, y, z = np.where(data > 0)
    # all_indices = np.array(zip(x, y, z))
    print out_values
    print scipy_out
    # print out_values - mapped_pos
    print out_values - scipy_out
    IPython.embed()


if __name__ == "__main__":
    test_gradients()
    # test_normal()
    # test_slices()
    # test_sum()
