import rospy
import numpy as np
import pybullet as pb
import openravepy as orpy
import hfts_grasp_planner.utils as utils


class PhysicsModel(object):
    """
        A physics model allows full simulation of a robot and its interaction
        with its environment.
    """
    class RobotController(object):
        """
            Allows you to set desired positions/velocities etc as targets for
            robot joints. These are then attempted to be acquired in the physics simulation.
        """

        def __init__(self, or_robot, robot_info, physics_client):
            """
                Create a new RobotController.
                ------
                Arguments
                ---------
                or_robot, OpenRAVE robot
                robot_info, tuple (int, dict), where the int is the bullet id
                    and dict is mapping from joint names to bullet joint ids
            """
            self._or_robot = or_robot
            self._robot_id = robot_info[0]
            self._physics_client = physics_client
            self._joint_map = robot_info[1]
            mimic_joints = [joint for joint in self._or_robot.GetPassiveJoints() if joint.IsMimic()]
            self._mimic_joints = {}
            joints = self._or_robot.GetJoints()
            for mjoint in mimic_joints:
                real_joint = joints[mjoint.GetMimicDOFIndices()[0]]
                real_joint_name = real_joint.GetName()
                if real_joint_name in self._mimic_joints:
                    self._mimic_joints[real_joint_name].append(mjoint)
                else:
                    self._mimic_joints[real_joint_name] = [mjoint]

        def set_position_target(self, joint_name, value):
            joint = self._or_robot.GetJoint(joint_name)
            mjoints = self._mimic_joints[joint_name]
            joint_values = [value] * (len(mjoints) + 1)
            joint_indices = [self._joint_map[tj.GetName()] for tj in mjoints]
            joint_indices.append(self._joint_map[joint.GetName()])
            pb.setJointMotorControlArray(self._robot_id, joint_indices, pb.POSITION_CONTROL, joint_values,
                                         physicsClientId=self._physics_client)

    class NoServerError(Exception):
        """
            Raised when there is no physics server available.
        """

        def __init__(self, message):
            super(PhysicsModel.NoServerError, self).__init__(message)

    class ImportError(Exception):
        """
            Raised when importing a robot or kinbody to Bullet failed.
        """

        def __init__(self, message):
            super(PhysicsModel.ImportError, self).__init__(message)

    def __init__(self, or_env, robot_urdf_files, body_urdf_path, server_id=None):
        """
            Create a new Bullet-based physics model.
            Note for this to work you need to run an external Bullet SharedMemory
            physics server first.
            ---------
            Arguments
            ---------
            or_env, OpenRAVE environment containing planning scene
            robot_urdf_files, dict - dictionary with elements <robot_name>: <urdf_file>, that provides
                for each robot in or_env a urdf file.
            body_urdf_path, string - path to folder containing urdf files for all bodies that are not robots
            server_id, int (optional) - If multiple Bullet physics servers are running, specify which one to use.
            ------
            Throws NoServerError if it can not connect to a Bullet server.
        """
        self._or_env = or_env
        if server_id is not None:
            self._client_id = pb.connect(pb.SHARED_MEMORY, key=server_id)
            if self._client_id == -1:
                raise PhysicsModel.NoServerError("Could not connect to Bullet server %i." % str(server_id))
        else:
            self._client_id = pb.connect(pb.SHARED_MEMORY)
            if self._client_id == -1:
                raise PhysicsModel.NoServerError("There is no Bullet server running.")
        self._robot_urdf_files = robot_urdf_files
        pb.setAdditionalSearchPath(body_urdf_path)
        # _bt_robot_infos stores tuples (id, joint_info), where id is int, joint_info a dict mapping joint name to id
        self._bt_robot_infos = {}
        self._object_ids = {}
        self._start_state = None
        self._time_step = pb.getPhysicsEngineParameters(self._client_id)['fixedTimeStep']

    def init_environment(self):
        """
            Initializes the physics environment from the set OpenRAVE environment.
            You need to call this once before using this physics model, and may call it again
            to reinitialize.
        """
        pb.resetSimulation(self._client_id)
        self._robot_ids = {}
        self._object_ids = {}
        self._start_state = None
        # run over all kinbodies and load these in pybullet
        bodies = self._or_env.GetBodies()
        for body in bodies:
            tf = body.GetTransform()
            if body.IsRobot():
                urdf_file = self._robot_urdf_files[body.GetName()]
                try:
                    robot_id = pb.loadURDF(urdf_file, basePosition=tf[:3, 3],
                                           baseOrientation=PhysicsModel.orquat2bt(orpy.quatFromRotationMatrix(tf)),
                                           useFixedBase=True,
                                           physicsClientId=self._client_id)
                except:
                    raise ImportError("Failed to import robot %s to Bullet. Invalid urdf file? URDF=%s" %
                                      (body.GetName(), urdf_file))
                joint_ids = {}
                for jid in xrange(pb.getNumJoints(robot_id, physicsClientId=self._client_id)):
                    jinfo = pb.getJointInfo(robot_id, jid)
                    joint_ids[jinfo[1]] = jid
                self._bt_robot_infos[body.GetName()] = (robot_id, joint_ids)
            else:
                try:
                    obj_id = pb.loadURDF(body.GetName() + ".urdf", basePosition=tf[:3, 3],
                                         baseOrientation=PhysicsModel.orquat2bt(orpy.quatFromRotationMatrix(tf)),
                                         physicsClientId=self._client_id)
                except:
                    raise ImportError("Failed to import kinbody %s to Bullet." % body.GetName())
                self._object_ids[body.GetName()] = obj_id
                # TODO support kinbodies with joints?
        pb.setGravity(0.0, 0.0, -9.81, self._client_id)
        self.set_env_state()
        self._start_state = pb.saveState()

    def set_env_state(self):
        """
            Synchronize the state of the Bullet environment to that of the OpenRAVE environment.
            It sets all dynamic kinbodies to the same poses/velocities and all robot joints to the same positions/velocities.
        """
        # first synchronize robots
        for robot_name, bt_robot_info in self._bt_robot_infos.iteritems():
            robot = self._or_env.GetRobot(robot_name)
            robot_key = bt_robot_info[0]
            joint_infos = bt_robot_info[1]
            # set base trasnform
            tf = robot.GetTransform()
            pb.resetBasePositionAndOrientation(
                robot_key, tf[:3, 3], PhysicsModel.orquat2bt(orpy.quatFromRotationMatrix(tf)),
                physicsClientId=self._client_id)
            base_link = robot.GetLinks()[0]
            base_vel = base_link.GetVelocity()
            pb.resetBaseVelocity(robot_key, base_vel[:3], base_vel[3:])
            # next set state of joints
            all_joints = robot.GetJoints()
            all_joints.extend(robot.GetPassiveJoints())
            for joint in all_joints:
                pb.resetJointState(robot_key, joint_infos[joint.GetName()], joint.GetValue(0), joint.GetVelocities()[0],
                                   physicsClientId=self._client_id)
        # now synch other bodies
        for body_name, body_id in self._object_ids.iteritems():
            body = self._or_env.GetKinBody(body_name)
            if body.IsRobot() or not utils.is_dynamic_body(body):
                continue
            base_link = body.GetLinks()[0]
            if len(body.GetLinks()) > 0:
                rospy.logwarn(
                    "[PhysicsModel::set_env_state] Body %s has more than one link. Only synchronizing first link." % body_name)
            tf = base_link.GetTransform()
            pb.resetBasePositionAndOrientation(body_id, tf[:3, 3], PhysicsModel.orquat2bt(orpy.quatFromRotationMatrix(tf)),
                                               physicsClientId=self._client_id)
            base_vel = base_link.GetVelocity()
            pb.resetBaseVelocity(body_id, base_vel[:3], base_vel[3:], physicsClientId=self._client_id)

    def get_env_state(self):
        """
            Synchronizes the state of the OpenRAVE environment to that of the Bullet environment.
            It sets all dynamic kinbodies to the same poses/velocities and all robot joints to the same positions/velocities.
        """
        # first synchronize robots
        for robot_name, bt_robot_info in self._bt_robot_infos.iteritems():
            robot = self._or_env.GetRobot(robot_name)
            robot_key = bt_robot_info[0]
            joint_infos = bt_robot_info[1]
            # set base trasnform
            base_link = robot.GetLinks()[0]
            tpos, btquat = pb.getBasePositionAndOrientation(robot_key, physicsClientId=self._client_id)
            tf = orpy.matrixFromQuat(PhysicsModel.btquat2or(btquat))
            tf[:3, 3] = tpos
            base_link.SetTransform(tf)
            tvel, rvel = pb.getBaseVelocity(robot_key, physicsClientId=self._client_id)
            base_link.SetVelocity(tvel, rvel)
            # next set state of joints
            joint_indices = [joint_infos[joint.GetName()] for joint in robot.GetJoints()]
            joint_states = pb.getJointStates(robot_key, joint_indices, physicsClientId=self._client_id)
            joint_pos = [state[0] for state in joint_states]
            joint_vels = [state[1] for state in joint_states]
            robot.SetDOFValues(joint_pos)
            robot.SetDOFVelocities(joint_vels)
        # now synch other bodies
        for body_name, body_id in self._object_ids.iteritems():
            body = self._or_env.GetKinBody(body_name)
            if body.IsRobot() or not utils.is_dynamic_body(body):
                continue
            base_link = body.GetLinks()[0]
            if len(body.GetLinks()) > 0:
                rospy.logwarn(
                    "[PhysicsModel::set_env_state] Body %s has more than one link. Only synchronizing first link." % body_name)

            tpos, btquat = pb.getBasePositionAndOrientation(body_id, physicsClientId=self._client_id)
            tf = orpy.matrixFromQuat(PhysicsModel.btquat2or(btquat))
            tf[:3, 3] = tpos
            base_link.SetTransform(tf)
            tvel, rvel = pb.getBaseVelocity(body_id, physicsClientId=self._client_id)
            base_link.SetVelocity(tvel, rvel)

    def step_physics(self, delta_t=None):
        """
            Forward simulate the simulation by at least the given time delta_t (in s).
            Default value for delta_t is one timestep of the simulation. Accordingly, the actual
            propagated amount of time is a multiple of the timestep, and thus <= delta_t + timestep.
        """
        if delta_t == None:
            delta_t = self._time_step
        num_iters = int(np.ceil(delta_t / self._time_step))
        for _ in xrange(num_iters):
            pb.stepSimulation(self._client_id)

    def reset_velocities(self):
        """
            Sets all velocities to zero - both in OpenRAVE and in Bullet.
        """
        for body in self._or_env.GetBodies():
            if utils.is_dynamic_body(body):
                base_link = body.GetLinks()[0]
                base_link.SetVelocity([0, 0, 0], [0, 0, 0])
                num_dofs = body.GetDOF()
                if num_dofs > 0:
                    body.SetDOFVelocities(np.zeros((num_dofs,)))
        self.set_env_state()

    def get_robot_controller(self, robot):
        """
            Return a robot controller for the given robot.
            ---------
            Arguments
            ---------
            robot, OpenRAVE robot
        """
        robot_info = self._bt_robot_infos[robot.GetName()]
        return PhysicsModel.RobotController(robot, robot_info, self._client_id)

    @staticmethod
    def orquat2bt(quat, bullet_quat=None):
        """
            Transform an OpenRAVE quaternion to Bullet quaternion.
            ---------
            Arguments
            ---------
            quat, numpy array of shape (4,) - OpenRAVE quaternion
            bullet_quat, numpy array of shape (4,) (optional) - new Bullet quaternion
        """
        if bullet_quat is None:
            bullet_quat = np.empty((4,))
        bullet_quat[:3] = quat[1:]
        bullet_quat[3] = quat[0]
        return bullet_quat

    @staticmethod
    def btquat2or(quat, or_quat=None):
        """
            Transform a Bullet quaternion to OpenRAVE.
            ---------
            Arguments
            ---------
            quat, numpy array of shape (4,) - Bullet quaternion
            bullet_quat, numpy array of shape (4,) (optional) - new OpenRAVE quaternion
        """
        if or_quat is None:
            or_quat = np.empty((4,))
        or_quat[0] = quat[3]
        or_quat[1:] = quat[:3]
        return or_quat
