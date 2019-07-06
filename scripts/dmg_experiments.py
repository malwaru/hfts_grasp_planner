#!/usr/bin/env python

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import std_msgs.msg as std_msgs
import hfts_grasp_planner.perception.ros_bridge as ros_bridge
from hfts_grasp_planner.utils import set_body_alpha, set_grasp
from hfts_grasp_planner.ik_solver import IKSolver
from hfts_grasp_planner.placement.afr_placement.multi_grasp import DMGGraspSet
import openravepy as orpy
import IPython
import argparse
import rospy
import os
import numpy as np

# Constants
MAX_TRAJ_GOAL_ERROR = 0.02
SQUEEZE_EFFORT = 20.0
INHAND_GRASP_EFFORT = 10.0
FLOPPY_GRASP_EFFORT = 0.0
RELEASE_EFFORT = -5.0

# Arm configurations
LEFT_ARM_DETECT_GRASP_CONFIG = np.array([-1.16256309, -1.48865187,  1.54684734,  0.35805732,  0.39554465,
        0.95125932,  0.74813634])
LEFT_ARM_HOME_CONFIG = np.array([-0.65002179, -2.46599984,  0.98793864,  0.39266104,  0.12282396,
        0.87903565,  0.78381282])
RIGHT_ARM_HOME_CONFIG = np.array([ 0.3597441 , -2.12499809, -1.69211626,  0.3060284 ,  0.38631114,
        0.6000796 , -1.48077667])
DETECT_GRASP_CONFIGS = {
    'left_arm_with_gripper': LEFT_ARM_DETECT_GRASP_CONFIG,
}
HOME_CONFIGS = {
    'left_arm_with_gripper': LEFT_ARM_HOME_CONFIG,
    'right_arm_with_gripper': RIGHT_ARM_HOME_CONFIG,
}

# general model folder
MODEL_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/../models/'

# Target objects information
TARGET_OBJ_NAME = 'expo'
TARGET_OBJ_FILE = MODEL_FOLDER + '/objects/' + TARGET_OBJ_NAME + '/expo.kinbody.xml'
DMG_INFO_FILE = MODEL_FOLDER + '/objects/' + TARGET_OBJ_NAME + '/dmg_info.yaml'

# Perception information
PERCEIVED_OBJ_NAMES = ['expo', 'cracker_box', 'mustard', 'elmers_glue', 'sugar_box']

# Robot information
ROBOT_URDF = MODEL_FOLDER + '/robots/yumi/yumi.urdf'
GRIPPER_INFO_FILE = MODEL_FOLDER + '/robots/yumi/gripper_information.yaml'
ROBOT_BASE_LINK = 'yumi_base_link'

# Planning scene information
ENV_FILE = MODEL_FOLDER + '/environments/rpl_lab_experiment.xml'

# ROS Topics and service names
YUMI_TRAJ_ACTION_LEFT = '/yumi/joint_traj_vel_controller_l/follow_joint_trajectory'
YUMI_TRAJ_ACTION_RIGHT = '/yumi/joint_traj_vel_controller_r/follow_joint_trajectory'
TRAJ_ACTION_NAMES = {'left_arm_with_gripper': YUMI_TRAJ_ACTION_LEFT,
                     'right_arm_with_gripper': YUMI_TRAJ_ACTION_RIGHT}
GRIPPER_EFFORT_TOPICS = {'left_arm_with_gripper': '/yumi/gripper_l_effort_cmd',
                         'right_arm_with_gripper': '/yumi_gripper_r_effort_cmd'}

def create_robot_clone(env, robot_name):
    robot = env.GetRobot(robot_name)
    robot_clone = orpy.RaveClone(robot, orpy.CloningOptions.Bodies)
    robot_clone.SetName(robot_name + "_clone")
    env.AddRobot(robot_clone)
    set_body_alpha(robot_clone, 0.1)
    return robot_clone 

class MotionPlannerWrapper(object):
    class GripperBridge(object):
        def __init__(self, robot):
            self._robot = robot
            self._gripper_pubs = {}
            for manip in self._robot.GetManipulators():
                self._gripper_pubs[manip.GetName()] = rospy.Publisher(GRIPPER_EFFORT_TOPICS[manip.GetName()],
                                                                      std_msgs.Float64, queue_size=1)

        def command_effort(self, manip_name, effort):
            self._gripper_pubs[manip_name].publish(effort)

    def __init__(self, robot, urdf_file, action_names, vel_factor=0.05):
        self._real_robot = robot 
        self._env = robot.GetEnv()
        self._robot_clone = create_robot_clone(robot.GetEnv(), robot.GetName())
        self._real_robot.Enable(False)
        # self._robot_clone.Enable(False)
        self._planners = {}
        self._ik_solvers = {}
        self._action_interfaces = {}
        for manip in self._robot_clone.GetManipulators():
            manip_name = manip.GetName()
            self._robot_clone.SetActiveManipulator(manip_name)
            self._robot_clone.SetActiveDOFs(manip.GetArmIndices())
            self._planners[manip_name] = orpy.interfaces.BaseManipulation(self._robot_clone)
            self._ik_solvers[manip_name] = IKSolver(manip, urdf_file_name=urdf_file)
            self._action_interfaces[manip_name] = ros_bridge.TrajectoryActionBridge(action_names[manip_name])
        self.set_vel_factor(vel_factor)
        self._gripper_bridge = MotionPlannerWrapper.GripperBridge(self._real_robot)

    def set_vel_factor(self, vel_factor):
        vel_limits = self._robot_clone.GetDOFVelocityLimits()
        self._robot_clone.SetDOFVelocityLimits(vel_factor * vel_limits)

    def plan(self, manip_name, target_config=None, target_pose=None):
        """
            Plan to the current configuration from the ghost robot.
            ---------
            Arguments
            ---------
            manip_name, string - which arm
            target_config (optional), np.array of shape (q,) - target arm configuration
                If not None, plan to this configuration, else to configuration of ghost robot.
            target_pose (optional), np.array of shape(4, 4) - target end-effector pose
                If not None, plan to this configuration, else configuration of ghost robot
                If target_config is specified, this argument is ignored.
        """
        traj = orpy.RaveCreateTrajectory(self._env, '')
        manip = self._robot_clone.GetManipulator(manip_name)
        manip_dofs = manip.GetArmIndices()
        self._robot_clone.SetActiveDOFs(manip_dofs)
        self._robot_clone.SetActiveManipulator(manip_name)
        start_config = self._real_robot.GetDOFValues(manip_dofs)
        with self._real_robot, self._robot_clone:
            self._real_robot.Enable(True)
            self._robot_clone.Enable(False)
            if self._env.CheckCollision(self._real_robot) or self._real_robot.CheckSelfCollision():
                rospy.logerr("Can not plan anywhere. Start configuration in collision")
                return None
        if target_config is None:
            if target_pose is None:
                target_config = self._robot_clone.GetActiveDOFValues()
            else:
                target_config = self._ik_solvers[manip_name].compute_collision_free_ik(target_pose)
                if target_config is None:
                    rospy.logerr("Could not compute a collision-free ik solution for manip %s" % manip_name)
                    return
        self._robot_clone.SetActiveDOFValues(start_config)
        traj = self._planners[manip_name].MoveActiveJoints(target_config, execute=False, outputtraj=True,
                                              outputtrajobj=True)
        return traj

    def execute_traj(self, traj, manip_name, bblock=False):
        """
            Execute the given OpenRAVE trajectory on the real robot.
            ---------
            Arguments
            ---------
            traj, OpenRAVE trajectory to execute
            manip_name, string - name of the manipulator this traj is for
            bblock, bool - if True, wait until traj should be finished
            -------
            Returns
            -------
            success - true if the goal of the trajectory has been reached (only if bblock is true)
        """
        ros_traj = self._convert_trajectory(traj, manip_name)
        self._action_interfaces[manip_name].execute_ros_traj(ros_traj)
        if bblock:
            duration = traj.GetDuration()
            rospy.loginfo("Sleeping for %fs for trajectory to finish" % duration)
            rospy.sleep(duration + 1.0)
            # check now whether we reached our goal
            target_config = np.array(ros_traj.points[-1].positions)
            arm_indices = self._real_robot.GetManipulator(manip_name).GetArmIndices()
            true_config = self._real_robot.GetDOFValues(arm_indices)
            assert(target_config.shape == true_config.shape)
            err = np.linalg.norm(target_config - true_config, ord=1)
            rospy.loginfo("Trajectory execution should be finished. Absolute error: %f" % err)
            return err < MAX_TRAJ_GOAL_ERROR

    def show_traj(self, traj):
        """
            Visualize the given OpenRAVE trajectory in OpenRAVE
        """
        self._robot_clone.GetController().SetPath(traj)
        self._robot_clone.WaitForController(0)

    def set_gripper_effort(self, manip_name, effort):
        self._gripper_bridge.command_effort(manip_name, effort)

    def _convert_trajectory(self, traj, manip_name):
            # The configuration specification allows us to interpret the trajectory data
            specs = traj.GetConfigurationSpecification()
            ros_trajectory = JointTrajectory()
            manip = self._robot_clone.GetManipulator(manip_name)
            self._robot_clone.SetActiveManipulator(manip_name)
            dof_indices = manip.GetArmIndices()
            self._robot_clone.SetActiveDOFValues(dof_indices)
            joint_names = {j.GetDOFIndex(): str(j.GetName()) for j in self._robot_clone.GetJoints()}
            ros_trajectory.joint_names = [joint_names[i] for i in dof_indices]
            time_from_start = 0.0
            # Iterate over all waypoints
            for i in range(traj.GetNumWaypoints()):
                wp = traj.GetWaypoint(i)
                ros_traj_point = JointTrajectoryPoint()
                ros_traj_point.positions = specs.ExtractJointValues(wp, self._robot_clone, dof_indices)
                ros_traj_point.velocities = specs.ExtractJointValues(wp, self._robot_clone, dof_indices, 1)
                delta_t = specs.ExtractDeltaTime(wp)
                # TODO why does this happen?
                if delta_t <= 10e-8 and i > 0:
                    rospy.logwarn('We have redundant waypoints in this trajectory, skipping...')
                    continue
                time_from_start += delta_t
                rospy.loginfo('Delta t is : %f' % delta_t)
                ros_traj_point.time_from_start = rospy.Duration().from_sec(time_from_start)
                ros_trajectory.points.append(ros_traj_point)
            return ros_trajectory



def create_grasp_set(robot, mplanner, manip_name):
    # first place the object in robot hand
    rospy.loginfo("Please place the object in the gripper of %s and press enter" % manip_name)
    raw_input()
    # grasp
    mplanner.set_gripper_effort(manip_name, SQUEEZE_EFFORT)
    # disable the target object for collision
    target_obj = robot.GetEnv().GetKinBody(TARGET_OBJ_NAME)
    target_obj.Enable(False)
    target_config = DETECT_GRASP_CONFIGS[manip_name]
    traj = mplanner.plan(manip_name, target_config=target_config)
    if traj is None:
        rospy.logerr("Could not compute a plan to grasp detection configuration")
        return
    # execute trajectory
    if not mplanner.execute_traj(traj, manip_name, bblock=True):
        rospy.logerr("Could not move to to grasp detection configuration")
        return
    # we should be seeing the object now. wait a little moment
    rospy.sleep(0.1)
    # now create the grasp set from the observed transformation
    manip = robot.GetManipulator(manip_name)
    grasp_set = DMGGraspSet(manip, target_obj, TARGET_OBJ_FILE, GRIPPER_INFO_FILE, DMG_INFO_FILE)
    initial_grasp = grasp_set.get_grasp(0)
    set_grasp(manip, target_obj, initial_grasp.eTo, initial_grasp.config)
    return grasp_set

if __name__ == "__main__":
    try:
        rospy.init_node("TestROSOrBridge")
        env = orpy.Environment()
        env.Load(ENV_FILE)
        robot = env.GetRobots()[0]
        state_synch = ros_bridge.RobotStateSynchronizer(robot, '/joint_states')
        state_synch.set_active(True)
        motion_planner = MotionPlannerWrapper(robot, ROBOT_URDF, TRAJ_ACTION_NAMES)
        tf_synch = ros_bridge.TFSynchronizer(env, PERCEIVED_OBJ_NAMES,
                                             MODEL_FOLDER + 'objects',
                                             ROBOT_BASE_LINK, robot.GetTransform())
        tf_synch.start()
        env.SetViewer('qtcoin')
        IPython.embed()
    except Exception as e:
        rospy.logerr(str(e))
    finally:
        tf_synch.end()
        orpy.RaveDestroy()
