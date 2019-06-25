#! /usr/bin/python

import hfts_grasp_planner.dmg.dmg_class as dmg_module
import hfts_grasp_planner.placement.afr_placement.multi_grasp as mg_mod
from hfts_grasp_planner.utils import inverse_transform
import openravepy as orpy
import numpy as np
import yaml
import IPython


def load_dmg(basepath, object_name, voxel_res=12, ang_res=20):
    dmg = dmg_module.DexterousManipulationGraph()
    # read the dmg from files
    dmg.set_object_shape_file(basepath + '/' + object_name + '.stl')
    dmg.read_graph("%s/graph_%s_%i_%i.txt" % (basepath, object_name, voxel_res, ang_res))
    dmg.read_nodes("%s/node_position_%s_%i_%i.txt" % (basepath, object_name, voxel_res, ang_res))
    dmg.read_node_to_component("%s/node_component_%s_%i_%i.txt" % (basepath, object_name, voxel_res, ang_res))
    dmg.read_component_to_normal("%s/component_normal_%s_%i_%i.txt" % (basepath, object_name, voxel_res, ang_res))
    dmg.read_node_to_angles("%s/node_angle_%s_%i_%i.txt" % (basepath, object_name, voxel_res, ang_res), ang_res)
    dmg.read_supervoxel_angle_to_angular_component(
        "%s/node_angle_angle_component_%s_%i_%i.txt" % (basepath, object_name, voxel_res, ang_res))
    return dmg


def load_finger_tf(filename):
    with open(filename, 'r') as info_file:
        finger_info = yaml.load(info_file)
    finger_tfs = {}
    for key, info_val in finger_info.iteritems():
        link_name = info_val['reference_link']
        rot = np.array(info_val['rotation'])
        trans = np.array(info_val['translation'])
        tf = orpy.matrixFromQuat(rot)
        tf[:3, 3] = trans
        finger_tfs[key] = (link_name, tf)
    return finger_tfs


if __name__ == "__main__":
    env = orpy.Environment()
    env.Load('models/environments/table_low_clutter.xml')
    env.Load('models/objects/elmers_glue/elmers_glue.kinbody.xml')
    robot = env.GetRobots()[0]
    manip_name = robot.GetActiveManipulator().GetName()
    manip = robot.GetManipulator(manip_name)
    target_obj = env.GetKinBody('elmers_washable_no_run_school_glue')
    obj_tf = np.array([[0.,  0., -1.,  0.57738799],
                       [-1.,  0.,  0.,  0.42658821],
                       [0.,  1.,  0.,  0.74622357],
                       [0.,  0.,  0.,  1.]])
    target_obj.SetTransform(obj_tf)
    config = np.array([6.90000018e-01, -1.94999998e+00,  2.37391844e-08, -7.15904048e-08,
                       5.18053657e-09,  7.37689967e-01,  2.25940696e-09,  1.44191191e-02,
                       -9.29999992e-01, -1.54999998e+00, -2.25243151e-09, -8.38054596e-08,
                       -2.76844143e-08, -1.73566970e-09, -4.53333343e-09,  1.99999820e-02])
    robot.SetDOFValues(config)
    # finger_info = load_finger_tf('models/robots/yumi/finger_information.yaml')
    # wTf_r = np.dot(robot.GetLink(finger_info[manip_name][0]).GetTransform(), finger_info[manip_name][1])
    # wTf_l = np.dot(robot.GetLink('gripper_r_finger_l').GetTransform(), finger_info[manip_name][1])
    # dmg = load_dmg('models/objects/elmers_glue/', 'glue')
    dmg = dmg_module.DexterousManipulationGraph.initFromPath('models/objects/elmers_glue/', 'glue', 12, 20)
    env.SetViewer('qtcoin')
    grasp_set = mg_mod.DMGGraspSet(manip, target_obj, 'models/robots/yumi/yumi_gripper_r.robot.xml',
                                   'models/objects/elmers_glue/elmers_glue.kinbody.xml',
                                   'models/robots/yumi/finger_information.yaml',
                                   dmg)
    IPython.embed()
