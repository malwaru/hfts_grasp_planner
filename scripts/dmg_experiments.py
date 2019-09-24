#!/usr/bin/env python

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import std_msgs.msg as std_msgs
import hfts_grasp_planner.perception.ros_bridge as ros_bridge
from hfts_grasp_planner.utils import set_body_alpha, set_grasp, path_to_trajectory, inverse_transform
from hfts_grasp_planner.ik_solver import IKSolver
from hfts_grasp_planner.placement.afr_placement.multi_grasp import DMGGraspSet
from hfts_grasp_planner.placement_planner import PlacementPlanner
from hfts_grasp_planner.dmg.dual_arm_pushing import DualArmPushingComputer
import openravepy as orpy
import IPython
import argparse
import rospy
import os
import yaml
import numpy as np

# Constants
MAX_TRAJ_GOAL_ERROR = 0.02
SQUEEZE_EFFORT = 20.0
INHAND_GRASP_EFFORT = 6.5
FLOPPY_GRASP_EFFORT = 0.0
RELEASE_EFFORT = -5.0

# Arm configurations
LEFT_ARM_DETECT_GRASP_CONFIG = np.array([-1.16256309, -1.48865187,  1.54684734,  0.35805732,  0.39554465,
        0.95125932,  0.74813634])
# LEFT_ARM_INHAND_CONFIG = np.array([-0.91377819, -1.48872864,  1.92046666,  0.62049645,  0.38236198,
#         0.81718266,  2.63366151])
LEFT_ARM_INHAND_CONFIG = np.array([-1.36583662, -1.48842084,  1.73510325,  0.47979113,  0.29852417,
        0.63616234,  2.48654008])
LEFT_ARM_HOME_CONFIG = np.array([-0.65002179, -2.46599984,  0.98793864,  0.39266104,  0.12282396,
        0.87903565,  0.78381282])
RIGHT_ARM_HOME_CONFIG = np.array([ 0.3597441 , -2.12499809, -1.69211626,  0.3060284 ,  0.38631114,
        0.6000796 , -1.48077667])
RIGHT_ARM_INHAND_START_CONFIG = np.array([[-1.5239929 , -0.86381763,  1.51539719,  0.76713341,  2.8649776 ,
       -0.24115354, -0.44164947]])
DETECT_GRASP_CONFIGS = {
    'left_arm_with_gripper': LEFT_ARM_DETECT_GRASP_CONFIG,
}
HOME_CONFIGS = {
    'left_arm_with_gripper': LEFT_ARM_HOME_CONFIG,
    'right_arm_with_gripper': RIGHT_ARM_HOME_CONFIG,
}

# general model folder
MODEL_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/../models/'
YAML_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/../yamls/'

# Target objects information
TARGET_OBJ_NAME = 'expo'
TARGET_OBJ_FILE = MODEL_FOLDER + '/objects/' + TARGET_OBJ_NAME + '/expo.kinbody.xml'
DMG_INFO_FILE = MODEL_FOLDER + '/objects/' + TARGET_OBJ_NAME + '/dmg_info.yaml'

# Perception information
# PERCEIVED_OBJ_NAMES = ['expo', 'cracker_box', 'mustard', 'elmers_glue', 'sugar_box']
# PERCEIVED_OBJ_NAMES = ['expo']

# Robot information
ROBOT_URDF = MODEL_FOLDER + '/robots/yumi/yumi.urdf'
GRIPPER_INFO_FILE = MODEL_FOLDER + '/robots/yumi/gripper_information.yaml'
ROBOT_BASE_LINK = 'yumi_base_link'

# Planning scene information
ENV_FILE = MODEL_FOLDER + '/environments/rpl_lab_experiment.xml'
# Placement problem definition file
PLACEMENT_PROBLEM_FILE = YAML_FOLDER + 'placement_problems/real_experiments.yaml'

# ROS Topics and service names
YUMI_TRAJ_ACTION_LEFT = '/yumi/joint_traj_vel_controller_l/follow_joint_trajectory'
YUMI_TRAJ_ACTION_RIGHT = '/yumi/joint_traj_vel_controller_r/follow_joint_trajectory'
TRAJ_ACTION_NAMES = {'left_arm_with_gripper': YUMI_TRAJ_ACTION_LEFT,
                     'right_arm_with_gripper': YUMI_TRAJ_ACTION_RIGHT}
GRIPPER_EFFORT_TOPICS = {'left_arm_with_gripper': '/yumi/gripper_l_effort_cmd',
                         'right_arm_with_gripper': '/yumi/gripper_r_effort_cmd'}

def resolve_paths(problem_desc, yaml_file):
    global_yaml = str(yaml_file)
    if not os.path.isabs(global_yaml):
        cwd = os.getcwd()
        global_yaml = cwd + '/' + global_yaml
    head, _ = os.path.split(global_yaml)
    for key in ['or_env', 'occ_file', 'sdf_file', 'urdf_file', 'target_obj_file', 'grasp_file',
                'robot_occtree', 'robot_occgrid', 'reachability_path', 'robot_ball_desc', 'dmg_file',
                'gripper_information']:
        if key in problem_desc:
            problem_desc[key] = os.path.normpath(head + '/' + problem_desc[key])

def load_grasp(problem_desc):
    if 'grasp_pose' in problem_desc and 'grasp_config' in problem_desc:
        return
    assert 'grasp_file' in problem_desc
    with open(problem_desc['grasp_file']) as grasp_file:
        grasp_yaml = yaml.load(grasp_file)
        if 'grasp_id' in problem_desc:
            grasp_id = problem_desc['grasp_id']
        else:
            grasp_id = 0
        if grasp_id >= len(grasp_yaml):
            raise IOError("Invalid grasp id: " + str(grasp_id))
        problem_desc['grasp_pose'] = grasp_yaml[grasp_id]['grasp_pose']
        # grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        # grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        # grasp_pose = utils.inverse_transform(grasp_pose)
        # problem_desc['grasp_pose'][:3] = grasp_pose[:3, 3]
        # problem_desc['grasp_pose'][3:] = orpy.quatFromRotationMatrix(grasp_pose)
        problem_desc['grasp_config'] = grasp_yaml[grasp_id]['grasp_config']

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
        self._original_vel_limits = self._robot_clone.GetDOFVelocityLimits()
        self.set_vel_factor(vel_factor)
        self._gripper_bridge = MotionPlannerWrapper.GripperBridge(self._real_robot)

    def set_vel_factor(self, vel_factor):
        self._robot_clone.SetDOFVelocityLimits(vel_factor * self._original_vel_limits)

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
        if traj.GetEnv() != self._env:
            traj = self._transfer_trajectory(traj)
        self._robot_clone.GetController().SetPath(traj)
        self._robot_clone.WaitForController(0)
        return traj

    def set_gripper_effort(self, manip_name, effort):
        self._gripper_bridge.command_effort(manip_name, effort)

    def _transfer_trajectory(self, traj):
        # transfer a trajectory made in another openrave env to the world this one
        other_cs = traj.GetConfigurationSpecification()
        dof_indices = None
        other_robot = None
        for g in other_cs.GetGroups():
            if 'joint_values' in g.name:
                substrings = g.name.split(' ')
                substrings = [s for s in substrings if len(s) > 0]
                dof_indices = map(int, substrings[2:])
                other_robot = traj.GetEnv().GetRobot(substrings[1])
        self._robot_clone.SetActiveDOFs(dof_indices)
        # copy waypoints
        configs = []
        for i in range(traj.GetNumWaypoints()):
            wp = traj.GetWaypoint(i)
            positions = other_cs.ExtractJointValues(wp, other_robot, dof_indices)
            configs.append(positions)
        my_traj = path_to_trajectory(self._robot_clone, configs, vel_factor=1.0)
        return my_traj

    def _convert_trajectory(self, traj, manip_name):
            # The configuration specification allows us to interpret the trajectory data
            specs = traj.GetConfigurationSpecification()
            ros_trajectory = JointTrajectory()
            manip = self._robot_clone.GetManipulator(manip_name)
            self._robot_clone.SetActiveManipulator(manip_name)
            dof_indices = manip.GetArmIndices()
            self._robot_clone.SetActiveDOFs(dof_indices)
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

def save_solution(goal, traj, robot, dof_indices, file_name):
    # extract configuraitions from trajectory
    configs = []
    specs = traj.GetConfigurationSpecification()
    for i in range(traj.GetNumWaypoints()):
        wp = traj.GetWaypoint(i)
        positions = specs.ExtractJointValues(wp, robot, dof_indices)
        configs.append(positions)
    data = {}
    data['grasp_id'] = goal.grasp_id
    data['dof_indices'] = dof_indices
    data['path'] = configs
    with open(file_name, 'w') as the_file:
        yaml.dump(data, the_file)

def load_solution(file_name, robot):
    with open(file_name, 'r') as the_file:
        data = yaml.load(the_file)
    configs = data['path']
    grasp_id = data['grasp_id']
    dof_indices = data['dof_indices']
    original_actives = robot.GetActiveDOFIndices()
    robot.SetActiveDOFs(dof_indices)
    traj = path_to_trajectory(robot, configs, vel_factor=1)
    robot.SetActiveDOFs(original_actives)
    return traj, grasp_id

def load_inhand_pushes(filename, pushing_manip):
    robot = pushing_manip.GetRobot()
    original_actives = robot.GetActiveDOFIndices()
    robot.SetActiveDOFs(pushing_manip.GetArmIndices())
    with open(filename, 'r') as the_file:
        data = yaml.load(the_file)
    pushes = []
    for push_paths in data:
        trajs = []
        for path in push_paths:
            traj = path_to_trajectory(robot, path, vel_factor=1)
            trajs.append(traj)
        pushes.append(trajs)
    robot.SetActiveDOFs(original_actives)
    return pushes

def observe_grasp(robot, mplanner, manip_name, open_gripper=False):
    if open_gripper:
        mplanner.set_gripper_effort(manip_name, RELEASE_EFFORT)
    # first place the object in robot hand
    rospy.loginfo("Please place the object in the gripper of %s and press enter" % manip_name)
    raw_input()
    # grasp
    mplanner.set_gripper_effort(manip_name, SQUEEZE_EFFORT)
    # disable the target object for collision
    target_obj = robot.GetEnv().GetKinBody(TARGET_OBJ_NAME)
    if target_obj is None:
        robot.GetEnv().Load(TARGET_OBJ_FILE)
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
    # manip = robot.GetManipulator(manip_name)
    # grasp_set = DMGGraspSet(manip, target_obj, TARGET_OBJ_FILE, GRIPPER_INFO_FILE, DMG_INFO_FILE)
    # initial_grasp = grasp_set.get_grasp(0)
    # set_grasp(manip, target_obj, initial_grasp.eTo, initial_grasp.config)
    # return grasp_set

def move_to_start(motion_planner):
    traj = motion_planner.plan('left_arm_with_gripper', target_config=LEFT_ARM_INHAND_CONFIG)
    if not motion_planner.execute_traj(traj, 'left_arm_with_gripper', bblock=True):
        rospy.logerr("Could not move left arm to start config.")
        return
    motion_planner.set_gripper_effort('left_arm_with_gripper', RELEASE_EFFORT)
    traj = motion_planner.plan('right_arm_with_gripper', target_config=RIGHT_ARM_HOME_CONFIG)
    if not motion_planner.execute_traj(traj, 'right_arm_with_gripper', bblock=True):
        rospy.logerr("Could not move right arm to start config")
        return

def search_good_solution(pplanner, start_config, trials):
    all_solutions = []
    for i in range(trials):
        pplanner.plan(60, robot.GetDOFValues())
        for (traj, goal) in pplanner.motion_planner.solutions:
            _, push_path = pplanner.grasp_set.return_pusher_path(goal.grasp_id)
            all_solutions.append((len(push_path), traj, goal))
    all_solutions.sort(key=lambda x: x[0])
    return all_solutions

def run_it(mplanner, push_trajs, traj, pushing_manip, grasping_manip, target_obj):
    move_to_start(mplanner)
    # open gripper
    mplanner.set_gripper_effort(grasping_manip.GetName(), RELEASE_EFFORT)
    # close
    rospy.loginfo("Please place the object in the gripper of %s and press enter" % grasping_manip.GetName())
    raw_input()
    # grasp
    mplanner.set_gripper_effort(grasping_manip.GetName(), SQUEEZE_EFFORT)
    # open pusher fingers
    mplanner.set_gripper_effort(pushing_manip.GetName(), RELEASE_EFFORT)
    # decrease effort on grasp
    mplanner.set_gripper_effort(grasping_manip.GetName(), INHAND_GRASP_EFFORT)
    for push_traj in push_trajs:
        for ptraj in push_traj:
            rospy.loginfo("confirm for next motion")
            raw_input()
            mplanner.execute_traj(ptraj, pushing_manip.GetName(), bblock=True)
    # move right gripper home
    target_obj.Enable(False)
    rtraj = motion_planner.plan(pushing_manip.GetName(), target_config=RIGHT_ARM_HOME_CONFIG)
    if not motion_planner.execute_traj(traj, pushing_manip.GetName(), bblock=True):
        rospy.logerr("Could not move right arm to start config")
        return
    mplanner.set_gripper_effort(grasping_manip.GetName(), SQUEEZE_EFFORT)
    rospy.loginfo("confirm for next motion")
    raw_input()
    mplanner.execute_traj(traj, grasping_manip.GetName(), bblock=True)
    rospy.loginfo("confirm release")
    raw_input()
    mplanner.set_gripper_effort(grasping_manip.GetName(), RELEASE_EFFORT)
    mplanner._robot_clone.ReleaseAllGrabbed()

if __name__ == "__main__":
    try:
        rospy.init_node("TestROSOrBridge")
        env = orpy.Environment()
        env.Load(ENV_FILE)
        env.Load(TARGET_OBJ_FILE)
        robot = env.GetRobots()[0]
        state_synch = ros_bridge.RobotStateSynchronizer(robot, '/joint_states')
        state_synch.set_active(True)
        motion_planner = MotionPlannerWrapper(robot, ROBOT_URDF, TRAJ_ACTION_NAMES)
        target_obj = env.GetKinBody(TARGET_OBJ_NAME)
        # target_obj.SetTransform(pplanner.target_object.GetTransform())
        with open(PLACEMENT_PROBLEM_FILE, 'r') as problem_desc_file:
            problem_desc = yaml.load(problem_desc_file)
            resolve_paths(problem_desc, PLACEMENT_PROBLEM_FILE)
            load_grasp(problem_desc)
        grasp_pose = orpy.matrixFromQuat(problem_desc["grasp_pose"][3:])
        grasp_pose[:3, 3] = problem_desc["grasp_pose"][:3]
        set_grasp(motion_planner._robot_clone.GetManipulator('left_arm_with_gripper'), target_obj, inverse_transform(grasp_pose), problem_desc['grasp_config'])
        # tf_synch = ros_bridge.TFSynchronizer(env, PERCEIVED_OBJ_NAMES,
        #                                      MODEL_FOLDER + 'objects',
        #                                      ROBOT_BASE_LINK, robot.GetTransform())
        # tf_synch.start()
        grasping_manip = motion_planner._robot_clone.GetManipulator('left_arm_with_gripper')
        pushing_manip = motion_planner._robot_clone.GetManipulator('right_arm_with_gripper')
        # load gripper information for pusher
        with open(GRIPPER_INFO_FILE, 'r') as info_file:
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

        pplanner = PlacementPlanner(problem_desc)
        push_computer = DualArmPushingComputer(motion_planner._robot_clone, grasping_manip, pushing_manip, ROBOT_URDF, eTp)
        traj, gid = load_solution('/home/joshua/snd_unknown_solution_2_pushes.yaml', motion_planner._robot_clone)
        push_trajs = load_inhand_pushes('/home/joshua/second_pushes.yaml', pushing_manip)[:1]
        grasp_path, push_path = pplanner.grasp_set.return_pusher_path(gid)
        # pushing_trajs = push_computer.compute_pushing_trajectory(grasp_path, push_path, target_obj)
        env.SetViewer('qtcoin')
        IPython.embed()
    except Exception as e:
        rospy.logerr(str(e))
    finally:
        # tf_synch.end()
        orpy.RaveDestroy()
