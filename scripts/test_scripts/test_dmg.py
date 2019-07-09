#! /usr/bin/python
import hfts_grasp_planner.dmg.dmg_class as dmg_module
import hfts_grasp_planner.dmg.dmg_path as dmg_path_module
import hfts_grasp_planner.dmg.dual_arm_pushing as dmg_pushing_module
import hfts_grasp_planner.placement.afr_placement.multi_grasp as mg_mod
from hfts_grasp_planner.utils import inverse_transform
import openravepy as orpy
import numpy as np
import yaml
import IPython


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


def playback_inhand_traj(grasping_manip, obj, push_traj):
    eeftf = grasping_manip.GetEndEffectorTransform()
    robot = grasping_manip.GetRobot()
    for (trajs, start_grasp, followup_grasp) in push_traj:
        obj.SetTransform(np.dot(eeftf, start_grasp.eTo))
        robot.SetDOFValues(start_grasp.config, grasping_manip.GetGripperIndices())
        for traj in trajs:
            robot.GetController().SetPath(traj)
            robot.WaitForController(0)
        obj.SetTransform(np.dot(eeftf, followup_grasp.eTo))


if __name__ == "__main__":
    env = orpy.Environment()
    urdf_file = 'models/robots/yumi/yumi.urdf'
    env.Load('models/environments/table_low_clutter.xml')
    env.Load('models/objects/expo/expo.kinbody.xml')
    robot = env.GetRobots()[0]
    manips = robot.GetManipulators()
    grasping_manip = manips[0]
    pushing_manip = manips[1]
    # load pushing tf
    with open('models/robots/yumi/gripper_information.yaml', 'r') as info_file:
        gripper_info = yaml.load(info_file)
        pushing_tf_dict = gripper_info[pushing_manip.GetName()]['pushing_tf']
        wTr = robot.GetLink(pushing_tf_dict['reference_link']).GetTransform()
        pose = np.empty(7)
        pose[:4] = pushing_tf_dict['rotation']
        pose[4:] = pushing_tf_dict['translation']
        rTp = orpy.matrixFromPose(pose)
        wTp = np.dot(wTr, rTp)
        wTe = pushing_manip.GetEndEffector().GetTransform()
        eTp = np.dot(inverse_transform(wTe), wTp)
    target_obj = env.GetKinBody('expo')
    with open('yamls/placement_problems/grasps/new/expo.yaml', 'r') as grasp_file:
        grasp_info = yaml.load(grasp_file)[0]
        config = grasp_info['grasp_config']
        grasp_pose = grasp_info['grasp_pose']
        pos = grasp_pose[:3]
        quat = grasp_pose[3:]
        oTe = orpy.matrixFromQuat(quat)
        oTe[:3, 3] = pos
        eTo = inverse_transform(oTe)
        target_obj.SetTransform(np.dot(grasping_manip.GetEndEffectorTransform(), eTo))
        robot.SetDOFValues(config, grasping_manip.GetGripperIndices())
    # obj_tf = np.array([[0.,  0., -1.,  0.57738787],
    #                    [-0.96246591, -0.27140261,  0.,  0.43454561],
    #                    [-0.27140261,  0.96246591,  0.,  0.74675405],
    #                    [0.,  0.,  0.,  1.]])
    # target_obj.SetTransform(obj_tf)
    # config = np.array([6.90000018e-01, -1.94999998e+00,  2.37391844e-08, -7.15904048e-08,
    #                    5.18053657e-09,  7.37689967e-01,  2.25940696e-09,  1.44191191e-02,
    #                    -9.29999992e-01, -1.54999998e+00, -2.25243151e-09, -8.38054596e-08,
    #                    -2.76844143e-08, -1.73566970e-09, -4.53333343e-09,  1.99999820e-02])
    # robot.SetDOFValues(config)
    # finger_info = load_finger_tf('models/robots/yumi/finger_information.yaml')
    # wTf_r = np.dot(robot.GetLink(finger_info[manip_name][0]).GetTransform(), finger_info[manip_name][1])
    # wTf_l = np.dot(robot.GetLink('gripper_r_finger_l').GetTransform(), finger_info[manip_name][1])
    env.SetViewer('qtcoin')
    grasp_set = mg_mod.DMGGraspSet(grasping_manip, target_obj,
                                   'models/objects/expo/expo.kinbody.xml',
                                   'models/robots/yumi/gripper_information.yaml',
                                   'models/objects/expo/dmg_info.yaml')
    pushing_computer = dmg_pushing_module.DualArmPushingComputer(robot, grasping_manip, pushing_manip, urdf_file, eTp)
    grasp_path, push_path = grasp_set.return_pusher_path(10)
    inhand_config = np.array([1.21007779e+00, -1.57321833e+00, -1.30163680e+00,  9.27653204e-01,
                              -1.00904288e+00,  7.37689903e-01,  2.51809821e-09])
    robot.SetDOFValues(inhand_config, grasping_manip.GetArmIndices())
    IPython.embed()
