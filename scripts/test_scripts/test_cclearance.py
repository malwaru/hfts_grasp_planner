#! /usr/bin/python
# import openravepy as orpy
import sys
import time
import numpy as np
sys.path.append('/home/joshua/projects/placement_catkin/devel/lib')
import clearance_utils
from scipy.ndimage import distance_transform_edt
# import hfts_grasp_planner.sdf.kinbody as kinbody_core
# import hfts_grasp_planner.placement2.clearance as clearance_module
# import hfts_grasp_planner.sdf.grid as grid_module
# import hfts_grasp_planner.sdf.core as sdf_core
import IPython

def test_speed():
    in_arr = np.zeros((200, 200, 200), dtype=bool)
    in_arr[3:7, 3:8, 4:8]  = True
    in_arr = np.invert(in_arr)
    out_arr = np.empty(in_arr.shape)
    adj_arr = np.ones((3, 3, 3), dtype=bool)
    in_arr[1:2, 1:2, 1:2]  = True
    # adj_arr[:, :, 0]  = False
    # adj_arr[:, :, 2]  = False
    start = time.time()
    clearance_utils.compute_df(in_arr, adj_arr, out_arr)
    print "Computation for an array of shape ", str(in_arr.shape), " took %fs" % (time.time() - start)


if __name__ == "__main__":
    in_arr = np.zeros((10, 10, 10), dtype=bool)
    in_arr[3:7, 3:8, 4:8]  = True
    in_arr = np.invert(in_arr)
    out_arr = np.empty(in_arr.shape)
    adj_arr = np.ones((3, 3, 3), dtype=bool)
    # test base case (single block)
    clearance_utils.compute_df(in_arr, adj_arr, out_arr)
    scipy_out = distance_transform_edt(in_arr)
    assert(np.isclose(scipy_out, out_arr).all())
    # test second case (two blocks)
    in_arr[1:2, 1:2, 1:2]  = True
    clearance_utils.compute_df(in_arr, adj_arr, out_arr)
    scipy_out = distance_transform_edt(in_arr)
    assert(np.isclose(scipy_out, out_arr).all())
    # test within z-slice adjacency only
    adj_arr[:, :, 0]  = False
    adj_arr[:, :, 2]  = False
    clearance_utils.compute_df(in_arr, adj_arr, out_arr)
    for z in range(in_arr.shape[2]):
        if in_arr[:, :, z].all():
            scipy_out[:, :, z] = np.inf
        else:
            scipy_out[:, :, z] = distance_transform_edt(in_arr[:, :, z])
    assert(np.isclose(scipy_out, out_arr).all())
    print "All tests successful"
    test_speed()
    # IPython.embed()