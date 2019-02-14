import hfts_grasp_planner.external.transformations as transformations
import hfts_grasp_planner.sdf.cuda_grid as cuda_grid
import scipy.ndimage
import numpy as np
import os
import math
import IPython

if __name__ == "__main__":
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
    data = 1.0 / 10.0 * data
    # data_t = data.transpose().copy()
    positions = np.array([[0.1, 0.5, 0.3], [0.5, 1.0, 1.0],
                          [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5], [0.0, 1.0, 1.0]])
    tf_matrix = transformations.random_rotation_matrix()
    # tf_matrix = np.eye(4)
    tf_matrix[:3, 3] = [0.2, 0.1, 0.2]
    interp = cuda_grid.CudaInterpolator(data)
    out_values = interp.interpolate(positions, tf_matrix=tf_matrix)
    mapped_pos = np.dot(positions, tf_matrix[:3, :3].transpose()) + tf_matrix[:3, 3]
    scipy_out = scipy.ndimage.map_coordinates(data, mapped_pos.transpose(), order=1, mode='nearest')
    x, y, z = np.where(data > 0)
    all_indices = np.array(zip(x, y, z))
    print out_values
    print scipy_out
    # print out_values - mapped_pos
    print out_values - scipy_out
    IPython.embed()
