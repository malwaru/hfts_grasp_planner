#! /usr/bin/python
import openravepy as orpy
import numpy as np
import hfts_grasp_planner.utils as utils
import yaml
import argparse


def extract_grasp(manip, body):
    obj_tf = body.GetTransform()
    eef_tf = manip.GetEndEffectorTransform()
    inv_obj_tf = utils.inverse_transform(obj_tf)
    grasp_tf = np.dot(inv_obj_tf, eef_tf)
    grasp_pose = np.empty(7)
    grasp_pose[:3] = grasp_tf[:3, 3]
    grasp_pose[3:] = orpy.quatFromRotationMatrix(grasp_tf)
    grasp_config = manip.GetRobot().GetDOFValues(manip.GetGripperIndices())
    return grasp_pose, grasp_config


def save_grasp(grasp_pose, grasp_config, filename):
    gp = [float(x) for x in grasp_pose]
    gc = [float(x) for x in grasp_config]
    grasp_dict = {'grasp_pose': gp, 'grasp_config': gc}
    with open(filename, 'w') as f:
        yaml.dump([grasp_dict], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a grasp for a kinbody.")
    parser.add_argument('robot', type=str, help='Path to xml file storing robot description')
    parser.add_argument('body', type=str, help='Path to xml file storing kinbody description')
    parser.add_argument('output_file', type=str, help='Path to where to store grasp')
    args = parser.parse_args()

    env = orpy.Environment()
    env.Load(args.robot)
    env.Load(args.body)
    env.SetViewer('qtcoin')

    robot = env.GetRobots()[0]
    body = env.GetBodies()[1]
    manip = robot.GetActiveManipulator()

    print "Move the body into the gripper of %s and press any key." % manip.GetName()
    raw_input()
    grasp_pose, grasp_config = extract_grasp(manip, body)

    print "Saving grasp pose %s and config %s to %s" % (str(grasp_pose), str(grasp_config), args.output_file)
    save_grasp(grasp_pose, grasp_config, args.output_file)
    orpy.RaveDestroy()
