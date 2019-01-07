#! /usr/bin/python
import hfts_grasp_planner.urdf_loader as urdf_utils
import numpy as np
import os
import IPython

if __name__ == "__main__":
    urdf_filename = os.path.abspath(os.path.dirname(__file__)) + '/../../models/yumi/yumi.urdf'
    with open(urdf_filename, 'r') as urdf_file:
        urdf_content = urdf_file.read()
    new_urdf = urdf_utils.set_base_pose(urdf_content, np.eye(4))
    new_urdf2 = urdf_utils.add_grasped_obj(new_urdf, 'gripper_l_base', 'test', np.eye(4))
    IPython.embed()
