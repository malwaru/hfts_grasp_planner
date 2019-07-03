#!/usr/bin/env python

# Import required Python code.
import roslib, rospy, tf, tf_conversions, threading, numpy
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance
import os
import actionlib

class TrajectoryActionBridge(object):
    def __init__(self, action_name, pos_tol=0.02, vel_tol=0.02):
        self._action_client = actionlib.SimpleActionClient(action_name, FollowJointTrajectoryAction)
        self.pos_tol = pos_tol
        self.vel_tol = vel_tol

    def execute_ros_traj(self, ros_traj):
        action_goal = FollowJointTrajectoryGoal()
        action_goal.trajectory = ros_traj
        for n in ros_traj.joint_names:
            jt = JointTolerance()
            jt.name = n
            jt.position = self.pos_tol
            jt.velocity = self.vel_tol
            action_goal.path_tolerance.append(jt)
        goal_stat = self._action_client.send_goal_and_wait(action_goal)
        if goal_stat == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Trajectory execution succeeded")        
        elif goal_stat == actionlib.GoalStatus.ABORTED:
            rospy.logwarn("Trajectory execution aborted")
        else:
            rospy.logwarn("Unknown result %i" % goal_stat)


class RobotStateSynchronizer(object):
    def __init__(self, or_robot, topic_name, joint_name_map=None):
        """
            Create a new RobotStateSynchronizer for the given OpenRAVE robot.
            ---------
            Arguments
            ---------
            or_robot, orpy.Robot
            topic_name, string - name of the joint states topic
            joint_name_map(optional), dict - dictionary mapping joint names from ROS side
                to joint names in OpenRAVE model
        """
        self._listener = rospy.Subscriber(topic_name, JointState, self._receive_joint_state)
        self._robot = or_robot
        self._joint_name_map = joint_name_map
        self._joint_name_to_dof = {}
        self._joint_name_to_dof = {j.GetName(): j.GetDOFIndex() for j in or_robot.GetJoints()}
        self._active = True

    def _receive_joint_state(self, msg):
        if self._active:
            indices = []
            values = []
            for i in range(len(msg.name)):
                name = msg.name[i]
                if self._joint_name_map is not None and name in self._joint_name_map:
                    name = self._joint_name_map[name]
                if name in self._joint_name_to_dof:
                    indices.append(self._joint_name_to_dof[name])
                    values.append(msg.position[i])
                else:
                    rospy.logdebug("Unknown joint %s" % name)
            self._robot.SetDOFValues(values, indices)

    def set_active(self, bactive):
        """
            Enable of disable robot state synchronization.
            ---------
            Arguments
            ---------
            bactive, bool - if True, enable, else disable
        """
        self._active = bactive


class TFSynchronizer(object):
    def __init__(self, or_env, object_names, data_folder, tf_base_frame, wTb=None):
        """
            Create a new TFSynchronizer for the given OpenRAVE environment.
            ---------
            Arguments
            ---------
            or_env, orpy.Environment
            object_name, list of strings - object names to query poses from tf for
            data_folder, string - path to folder containing subfolders with the object name,
                which in turn, contain a kinbody file for the object
            tf_base_frame, string - the name of the frame in which the OpenRAVE environment
                is set
            wTb(optional), np.array (4, 4) - optionally a transformation matrix to apply
                on each pose after querying from tf - default is np.eye(4)
        """
        self._env = or_env
        self._object_names = object_names
        self._data_folder = data_folder
        self._is_running = False
        self._listener = None
        self._tf_base_frame = tf_base_frame
        self._thread = None
        if wTb is None:
            wTb = numpy.eye(4)
        self._transform = wTb
        self.ignore_set = set()

    def __del__(self):
        self.end()

    def start(self):
        if self._is_running:
            rospy.logwarn('Perception is already running')
            return
        # rospy.init_node('TFSynchr', anonymous = True)
        self._listener = tf.TransformListener()
        self._is_running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        rate = rospy.Rate(80.0)
        while not rospy.is_shutdown() and self._is_running:
            for obj in self._object_names:
                if obj in self.ignore_set:
                    continue
                try:
                    (trans, rot) = self._listener.lookupTransform(self._tf_base_frame, obj, rospy.Time(0))
                    with self._env:
                        body = self._env.GetKinBody(obj)
                        if body is None:
                            # check whether we have a kinbody file that we can load
                            folder_path = self._data_folder + '/' + obj + '/'
                            kinbody_files = filter(lambda x: 'kinbody.xml' in x, os.listdir(folder_path))
                            if len(kinbody_files) > 0:
                                for body_file in kinbody_files:
                                    bloaded = self._env.Load(folder_path + body_file)
                                    if bloaded:
                                        body = self._env.GetKinBody(obj)
                                        break
                            if body is None:
                                continue
                        # Compute transform
                        matrix = tf_conversions.transformations.quaternion_matrix([rot[0], rot[1], rot[2], rot[3]])
                        matrix[:3, 3] = trans
                        matrix = numpy.dot(self._transform, matrix)
                        body.SetTransform(matrix)
                        # self.drawer.drawPose(self._orEnv, matrix)

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            rate.sleep()

    def enableObject(self, obj_name, enable = True):
        if enable:
            if obj_name in self.ignore_set:
                self.ignore_set.remove(obj_name)
        else:
            if obj_name not in self.ignore_set:
                self.ignore_set.add(obj_name)

    def end(self):
        if self._is_running:
            self._is_running = False
            self._thread.join()

def test_tf():
    import openravepy, IPython
    rospy.init_node("TestORBridge")
    env = openravepy.Environment()
    env.SetViewer('qtcoin')
    object_names = ['elmers_glue', 'sugar_box', 'mustard', 'cabinet', 'cracker_box', 'expo']
    data_path = '/home/joshua/projects/placement_ws/src/hfts_grasp_planner/models/objects/'
    perception = TFSynchronizer(env, object_names, data_path, 'kinect2_rgb_optical_frame')
    perception.start()
    IPython.embed()
    perception.end()

def test_robot_state():
    import openravepy, IPython
    rospy.init_node("TestORBridge")
    env = openravepy.Environment()
    env.SetViewer('qtcoin')
    env.Load("/home/joshua/projects/placement_ws/src/hfts_grasp_planner/models/robots/yumi/yumi.xml")
    state_synch = RobotStateSynchronizer(env.GetRobots()[0], '/joint_states')
    state_synch.set_active(True)
    IPython.embed()


# Main function.
if __name__ == '__main__':
    test_robot_state()