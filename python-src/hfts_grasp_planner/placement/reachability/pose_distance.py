import os
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double


def __reduce__():
    return (get_distance_fn, ())


def get_distance_fn():
    # load distance function for ball tree
    # declare pose type
    pose_array_type = npct.ndpointer(dtype=np.double, shape=(7,), flags='CONTIGUOUS')
    # load c library - it should be in the linker path if catkin_make was run
    ld_paths = os.environ["LD_LIBRARY_PATH"].split(':')
    for ld_path in ld_paths:
        lib_pose_distance = None
        if os.path.exists(ld_path + '/libyumi_pose_distance.so'):
            lib_pose_distance = npct.load_library('libyumi_pose_distance', ld_path)
        if lib_pose_distance is not None:
            break
    if lib_pose_distance is None:
        raise RuntimeError(
            "Could not locate libyumi_pose_distance.so. Please ensure its directory is in the LD_LIBRARY_PATH")
    lib_pose_distance.pose_distance.argtypes = [pose_array_type, pose_array_type]
    lib_pose_distance.pose_distance.restype = c_double
    lib_pose_distance.pose_distance.__reduce__ = __reduce__
    return lib_pose_distance.pose_distance
