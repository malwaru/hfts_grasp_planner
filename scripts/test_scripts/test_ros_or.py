#!/usr/bin/env python

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import hfts_grasp_planner.perception.ros_bridge as ros_bridge
from hfts_grasp_planner.utils import set_body_alpha
import openravepy as orpy
import IPython
import argparse
import rospy

def create_robot_clone(env, robot_name):
    robot = env.GetRobot(robot_name)
    robot_clone = orpy.RaveClone(robot, orpy.CloningOptions.Bodies)
    robot_clone.SetName(robot_name + "_clone")
    env.AddRobot(robot_clone)
    set_body_alpha(robot_clone, 0.1)
    return robot_clone 

class MotionPlannerWrapper(object):
    def __init__(self, robot, manip_name, vel_factor=0.05):
        self._real_robot = robot 
        # self._real_robot.Enable(False)
        self._env = robot.GetEnv()
        self._robot_clone = create_robot_clone(robot.GetEnv(), robot.GetName())
        self._robot_clone.Enable(False)
        self._real_robot.SetActiveManipulator(manip_name)
        self.set_active_dofs(self._real_robot.GetActiveManipulator().GetArmIndices())
        self._planner = orpy.interfaces.BaseManipulation(self._robot_clone)
        self.set_vel_factor(vel_factor)

    def set_active_dofs(self, active_dofs):
        self._robot_clone.SetActiveDOFs(active_dofs)

    def set_vel_factor(self, vel_factor):
        vel_limits = self._robot_clone.GetDOFVelocityLimits()
        self._robot_clone.SetDOFVelocityLimits(vel_factor * vel_limits)

    def plan(self, target_config=None):
        """
            Plan to the current configuration from the ghost robot.
        """
        traj = orpy.RaveCreateTrajectory(self._env, '')
        start_config = self._real_robot.GetDOFValues(self._robot_clone.GetActiveDOFIndices())
        if target_config is None:
            target_config = self._robot_clone.GetActiveDOFValues()
        self._robot_clone.SetActiveDOFValues(start_config)
        traj = self._planner.MoveActiveJoints(target_config, execute=False, outputtraj=True,
                                              outputtrajobj=True)
        return traj

    def show_traj(self, traj):
        self._robot_clone.GetController().SetPath(traj)
        self._robot_clone.WaitForController(0)

    def convert_trajectory(self, traj):
            # The configuration specification allows us to interpret the trajectory data
            specs = traj.GetConfigurationSpecification()
            ros_trajectory = JointTrajectory()
            manip = self._robot_clone.GetActiveManipulator()
            dof_indices = manip.GetArmIndices()
            joint_names = {j.GetDOFIndex(): str(j.GetName()) for j in self._robot_clone.GetJoints()}
            ros_trajectory.joint_names = [joint_names[i] for i in dof_indices]
            time_from_start = 0.0
            # Iterate over all waypoints
            for i in range(traj.GetNumWaypoints()):
                wp = traj.GetWaypoint(i)
                ros_traj_point = JointTrajectoryPoint()
                ros_traj_point.positions = specs.ExtractJointValues(wp, self._robot_clone, range(len(dof_indices)))
                ros_traj_point.velocities = specs.ExtractJointValues(wp, self._robot_clone, range(len(dof_indices)), 1)
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


if __name__ == "__main__":
    rospy.init_node("TestROSOrBridge")
    env = orpy.Environment()
    env.SetViewer('qtcoin')
    env.Load("/home/joshua/projects/placement_ws/src/hfts_grasp_planner/models/robots/yumi/yumi.xml")
    robot = env.GetRobots()[0]
    state_synch = ros_bridge.RobotStateSynchronizer(robot, '/joint_states')
    state_synch.set_active(True)
    planner = MotionPlannerWrapper(robot, 'right_arm_with_gripper')
    traj_action = ros_bridge.TrajectoryActionBridge('/yumi/joint_traj_vel_controller_r/follow_joint_trajectory')
    IPython.embed()