#! /usr/bin/python
import os
import IPython
import numpy as np
import openravepy as orpy
import trac_ik_python.trac_ik as trac_ik_module
import hfts_grasp_planner.urdf_loader as urdf_utils_mod
import hfts_grasp_planner.external.transformations as tf_mod


def create_ik_solver(manip, urdf_file_name, tool_pose):
    with open(urdf_file_name, 'r') as urdf_file:
        urdf_content = urdf_file.read()
    updated_urdf = urdf_utils_mod.set_base_pose(urdf_content, manip.GetRobot().GetTransform())
    # next add tf to new tip point
    updated_urdf = urdf_utils_mod.add_grasped_obj(
        updated_urdf, manip.GetEndEffector().GetName(), 'new_tooltip', tool_pose)
    IPython.embed()
    ik_solver = trac_ik_module.IK('yumi_base_link',
                                  'new_tooltip', urdf_string=updated_urdf)
    lower_limits, upper_limits = manip.GetRobot().GetDOFLimits(manip.GetArmIndices())
    ik_solver.set_joint_limits(lower_limits, upper_limits)
    return ik_solver


def to_world_frame(manip, tool_pose):
    eef_pose = manip.GetEndEffector().GetTransform()
    return np.dot(eef_pose, tool_pose)


def draw_tooltip_frame(manip, tool_pose):
    return orpy.misc.DrawAxes(manip.GetRobot().GetEnv(), to_world_frame(manip, tool_pose))


def query_ik(ik_solver, pose, **kwargs):
    quat_pose = np.empty((7,))
    quat_pose[:3] = pose[:3, 3]
    quat_pose[3:] = orpy.quatFromRotationMatrix(pose)
    lower, upper = ik_solver.get_joint_limits()
    rnd = np.random.rand(len(lower))
    seed = np.array(lower) + rnd * (np.array(upper) - np.array(lower))
    seed = ik_solver.get_ik(qinit=seed, x=quat_pose[0], y=quat_pose[1], z=quat_pose[2],
                            rx=quat_pose[4], ry=quat_pose[5], rz=quat_pose[6], rw=quat_pose[3],
                            **kwargs)
    return seed


if __name__ == "__main__":
    env = orpy.Environment()
    env_file = os.path.abspath(os.path.dirname(__file__)) + '/../../data/environments/cluttered_env_yumi.xml'
    # env_file = os.path.abspath(os.path.dirname(__file__)) + '/../../models/yumi/yumi.xml'
    env.Load(env_file)
    urdf_file = os.path.abspath(os.path.dirname(__file__)) + '/../../models/yumi/yumi.urdf'
    robot = env.GetRobots()[0]
    tool_pose = tf_mod.rotation_matrix(0.3, np.array([0, 0, 1.0]))
    tool_pose[:3, 3] = np.array([0.1, 0.03, 0.2])
    ik_solver = create_ik_solver(robot.GetActiveManipulator(), urdf_file, tool_pose)
    env.SetViewer('qtcoin')
    IPython.embed()
