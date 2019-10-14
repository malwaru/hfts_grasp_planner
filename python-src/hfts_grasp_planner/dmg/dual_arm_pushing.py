import numpy as np
import openravepy as orpy
from hfts_grasp_planner.utils import set_grasp, inverse_transform, path_to_trajectory, compute_pseudo_inverse_rank, vec_angle_diff
from hfts_grasp_planner.ik_solver import IKSolver
import hfts_grasp_planner.external.transformations as tf_mod


class PlanningException(Exception):
    def __init__(self, message):
        super(Exception, self).__init__(message)


class DualArmPushingComputer(object):

    def __init__(self, robot, grasping_manip, pushing_manip, urdf_file, pushing_tooltip=None):
        """
            Create a new DualArmPushingComputer...
            Set the frame of the pushing contact relative to pushing manipulator's gripper frame.
            ---------
            Arguments
            ---------
            pushing_tooltip, np array of shape (4, 4)
        """
        self.vel_factor = 1.0
        self.robot = robot
        self.grasping_manip = grasping_manip
        self.pushing_manip = pushing_manip
        if pushing_tooltip is None:
            pushing_tooltip = np.eye(4)
        self.pushing_tooltip = pushing_tooltip
        ltf = pushing_manip.GetLocalToolTransform()
        pushing_manip.SetLocalToolTransform(self.pushing_tooltip)
        self._ik_solver = IKSolver(pushing_manip, urdf_file_name=urdf_file)
        pushing_manip.SetLocalToolTransform(ltf)
        with self.robot:
            active_manip = self.robot.GetActiveManipulator()
            self.robot.SetActiveManipulator(pushing_manip)
            self.mplanner = orpy.interfaces.BaseManipulation(self.robot)
            self.robot.SetActiveManipulator(active_manip)

    def _check_validity(self, config):
        """
            Return whether the given configuration for the active dofs is valid.
        """
        lower, upper = self.robot.GetActiveDOFLimits()
        binlimits = np.logical_and.reduce(lower <= config) and np.logical_and.reduce(config <= upper)
        return binlimits and not self.robot.GetEnv().CheckCollision(self.robot) and not self.robot.CheckSelfCollision()

    def _move_straight(self, end_pose, err_tol):
        """
            Move the pushing end-effector (with local tool transform) along a straight line to end_pose
            ----------------------------
            Assumptions and Side-effects
            TODO only goes to position not orientation
            ----------------------------
            self.robot active dofs are set to self.pushing_manip.GetArmIndices()
            The robot starts at its current configuration.
            At the end of this function the robot is at final configuration of the straigt line motion.
            ---------
            Arguments
            ---------
            end_pose, np.array (4, 4) - target pose in world frame
            err_tol, float - error tolerance (when to stop)
            -------
            Returns
            -------
            list of configs
        """
        config = np.array(self.robot.GetActiveDOFValues())
        config_path = [np.array(config)]
        cart_dir = np.zeros(6)
        step_size = 0.01
        wTp = self.pushing_manip.GetEndEffectorTransform()
        distance_to_go = np.linalg.norm(end_pose[:3, 3] - wTp[:3, 3])
        while distance_to_go >= err_tol:
            wTp = self.pushing_manip.GetEndEffectorTransform()
            position_err = end_pose[:3, 3] - wTp[:3, 3]
            distance_to_go = np.linalg.norm(position_err)
            cart_dir[:3] = position_err
            # compute Jacobian
            jacobian = np.empty((6, self.pushing_manip.GetArmDOF()))
            jacobian[:3] = self.pushing_manip.CalculateJacobian()
            jacobian[3:] = self.pushing_manip.CalculateAngularVelocityJacobian()
            # compute pseudo inverse
            inv_jac, rank = compute_pseudo_inverse_rank(jacobian)
            if rank < 6:  # if we are in a singularity, we failed
                raise PlanningException("Failed to move straight. Ran into a singularity")
            dq = np.matmul(inv_jac, cart_dir)
            delta_q_norm = np.linalg.norm(dq)
            if delta_q_norm <= 1e-4:
                break
            dq /= delta_q_norm
            config += min(step_size, delta_q_norm) * dq
            if not self._check_validity(config):
                raise PlanningException("Failed to move straight. Ran into joint limit or collision")
            config_path.append(np.array(config))
            self.robot.SetActiveDOFValues(config)
        return config_path

    def _compute_translational_push(self, start_pose, end_pose, obj_body, offset=0.01, err_tol=1e-3):
        """
            Compute a pushing path in joint space for a translational push from start_pose to end_pose.
        """
        fixed_gripper_frame = self.grasping_manip.GetEndEffector().GetTransform()
        world_start = np.dot(fixed_gripper_frame, start_pose)
        world_end = np.dot(fixed_gripper_frame, end_pose)
        pushing_dir = world_end[:3, 3] - world_start[:3, 3]
        pushing_distance = np.linalg.norm(pushing_dir)
        if pushing_distance <= 1e-3:
            return []
        pushing_dir = pushing_dir / pushing_distance
        approach_pose = np.array(world_start)
        approach_pose[:3, 3] -= offset * pushing_dir
        start_config, bfree = self._ik_solver.compute_collision_free_ik(approach_pose)
        if start_config is None or not bfree:
            raise PlanningException(
                "Could not compute collision-free ik solution for approach pose of translational push")
        # move to start configuration using motion planner to avoid colliding with the object
        approach_traj = self.mplanner.MoveActiveJoints(start_config, execute=False, outputtraj=True, outputtrajobj=True)
        if approach_traj is None:
            raise PlanningException("Failed to plan a motion to the approach pose of translational push")
        # now start the actual push
        self.robot.SetActiveDOFValues(start_config)
        with obj_body:
            obj_body.Enable(False)
            config_path = self._move_straight(world_end, err_tol)
            push_traj = path_to_trajectory(self.robot, config_path, vel_factor=self.vel_factor)
            retreat_path = list(config_path)
            retreat_path.reverse()
            retreat_traj = path_to_trajectory(self.robot, retreat_path, vel_factor=self.vel_factor)
            # move to a retreat pose
            self.robot.SetActiveDOFValues(start_config)
            obj_body.Enable(True)
            return [approach_traj, push_traj, retreat_traj]

    def _make_grasp_frame(self, start_pose, end_pose, rot_center):
        r1 = start_pose[:3, 3] - rot_center
        r1 /= np.linalg.norm(r1)
        r2 = end_pose[:3, 3] - rot_center
        r2 /= np.linalg.norm(r2)
        z = np.cross(r1, r2)
        # normalize z
        z_length = np.linalg.norm(z)
        z /= z_length
        y = np.cross(z, r1)
        # normalize y (for numerical reasons might be not 1.0)
        y_length = np.linalg.norm(y)
        y /= y_length
        frame = np.eye(4)
        frame[:3, :3] = np.c_[r1, y, z]
        frame[:3, 3] = rot_center
        return frame

    def _compute_rotational_push(self, start_pose, end_pose, rot_center, obj_body, err_tol=1e-3, angular_err_tol=1e-2, offset=0.02):
        """
            Compute a pushing path in joint space for a rotational psuh from start_pose to end_pose.
            The rotational push is centered at rot_center.
        """
        fixed_gripper_frame = self.grasping_manip.GetEndEffector().GetTransform()
        # compute approach pose first; we push along -x in start pose frame, so make it a little bit offset
        sTa = np.eye(4)
        sTa[0, 3] = offset
        approach_pose = np.dot(fixed_gripper_frame, np.dot(start_pose, sTa))
        # handle = orpy.misc.DrawAxes(self.robot.GetEnv(), approach_pose)
        start_config, bfree = self._ik_solver.compute_collision_free_ik(approach_pose, seed=self.robot.GetActiveDOFValues())
        if start_config is None or not bfree:
            raise PlanningException("Could not compute collision-free ik solution for approach pose of rotational push")
        # move to start configuration using motion planner to avoid colliding with the object
        approach_traj = self.mplanner.MoveActiveJoints(start_config, execute=False, outputtraj=True, outputtrajobj=True)
        if approach_traj is None:
            raise PlanningException("Failed to plan a motion to the approach pose of rotational push")
        # set robot to start config
        self.robot.SetActiveDOFValues(start_config)
        # now move to pushing point in a straight line
        world_start_pose = np.dot(fixed_gripper_frame, start_pose)
        # start_pose_handle = orpy.misc.DrawAxes(self.robot.GetEnv(), world_start_pose)
        world_end_pose = np.dot(fixed_gripper_frame, end_pose)
        # end_pose_handle = orpy.misc.DrawAxes(self.robot.GetEnv(), world_end_pose)
        with obj_body:
            obj_body.Enable(False)
            config_path = self._move_straight(world_start_pose, err_tol)
            approach2_traj = path_to_trajectory(self.robot, config_path, vel_factor=self.vel_factor)
            # now rotate
            # first compute grasp frame in end-effector frame (grasp frame is located at the contact point; we rotate around its z axis)
            eTg = self._make_grasp_frame(start_pose, end_pose, rot_center)
            gTe = inverse_transform(eTg)
            wTg = np.dot(fixed_gripper_frame, eTg)
            gTw = inverse_transform(wTg)
            # grasp_frame_handle = orpy.misc.DrawAxes(self.robot.GetEnv(), wTg)
            # get current pusher pose in world framce
            wTp = self.pushing_manip.GetEndEffectorTransform()
            # translate to grasp frame
            # eTw = inverse_transform(fixed_gripper_frame)
            gTp = np.dot(gTw, wTp)
            # angular error
            r1 = gTp[:3, 3]
            # transform target pose into grasp frame
            gTp_target = np.dot(gTe, end_pose)
            r2 = gTp_target[:3, 3]
            angular_error = vec_angle_diff(r1, r2)
            cart_vel = np.zeros(6)
            config = self.robot.GetActiveDOFValues()
            config_path = [np.array(config)]
            step_size = 0.01
            # self.handles = [orpy.misc.DrawCircle(self.robot.GetEnv(), wTg[:3, 3], wTg[:3, 2], np.linalg.norm(r1))]
            while angular_error > angular_err_tol:
                # get current pusher pose in world framce
                wTp = self.pushing_manip.GetEndEffectorTransform()
                # handle_2 = orpy.misc.DrawAxes(self.robot.GetEnv(), wTp)
                # self.handles.append(handle_2)
                # translate to grasp frame
                gTp = np.dot(gTw, wTp)
                # angular error
                r1 = gTp[:3, 3]
                # could also measure angular error as error between the two poses gTp and gTp_target
                angular_error = vec_angle_diff(r1, r2)
                omega = np.array([0, 0, angular_error])
                v = np.cross(omega, r1)
                cart_vel[:3] = np.dot(wTg[:3, :3], v)
                cart_vel[3:] = np.dot(wTg[:3, :3], omega)
                # handle_v = self.robot.GetEnv().drawarrow(wTp[:3, 3], wTp[:3, 3] + cart_vel[:3], 0.001)
                # compute Jacobian
                jacobian = np.empty((6, self.pushing_manip.GetArmDOF()))
                jacobian[:3] = self.pushing_manip.CalculateJacobian()
                jacobian[3:] = self.pushing_manip.CalculateAngularVelocityJacobian()
                # compute pseudo inverse
                inv_jac, rank = compute_pseudo_inverse_rank(jacobian)
                if rank < 6:  # if we are in a singularity, we failed
                    raise PlanningException("Failed to rotate. Ran into a singularity")
                dq = np.matmul(inv_jac, cart_vel)
                delta_q_norm = np.linalg.norm(dq)
                if delta_q_norm <= 1e-4:
                    break
                # dq /= delta_q_norm
                # config += min(step_size, delta_q_norm) * dq
                config += step_size * dq
                if not self._check_validity(config):
                    raise PlanningException("Failed to rotate. Ran into joint limit or collision")
                config_path.append(np.array(config))
                self.robot.SetActiveDOFValues(config)
            rotate_traj = path_to_trajectory(self.robot, config_path, vel_factor=self.vel_factor)
            # retreat motion, move away from grasp center
            wTp = self.pushing_manip.GetEndEffectorTransform()
            retreat_pose = np.array(wTp)
            retreat_dir = retreat_pose[:3, 3] - wTg[:3, 3]
            hack_retreat = world_start_pose[:3, 3] - wTg[:3, 3]
            retreat_dir += hack_retreat
            retreat_dir /= 2.0
            retreat_pose[:3, 3] += offset * retreat_dir / np.linalg.norm(retreat_dir)
            config_path = self._move_straight(retreat_pose, err_tol)
            retreat_traj = path_to_trajectory(self.robot, config_path, vel_factor=self.vel_factor)
            obj_body.Enable(True)
            return [approach_traj, approach2_traj, rotate_traj, retreat_traj]

    def compute_pushing_trajectory(self, grasp_path, pusher_path, obj_body):
        """
            Compute a pushing trajectory.
            ---------
            Arguments
            ---------
            grasp_path, list of Grasps - sequence of grasp to transition through
            pusher_pat, list of tuples describing pushes
            obj_body, Kinbody - the object to regrasp
            -------
            Returns
            -------
            push_trajs, list tuples (trajs, start_grasp, resulting_grasp),
                where trajs is a list of OpenRAVE trajectories that change the grasp from start_grasp
                to resulting_grasp
        """
        assert(len(grasp_path) == len(pusher_path) + 1)
        with self.robot:
            self.pushing_manip.SetLocalToolTransform(self.pushing_tooltip)
            manip_dofs = self.pushing_manip.GetArmIndices()
            self.robot.SetActiveDOFs(manip_dofs)
            vel_limits = self.robot.GetDOFVelocityLimits()
            self.robot.SetDOFVelocityLimits(self.vel_factor * vel_limits)
            push_trajs = []
            for pid in xrange(len(pusher_path)):
                # set current state of object
                current_grasp = grasp_path[pid]
                set_grasp(self.grasping_manip, obj_body, current_grasp.eTo, current_grasp.config)
                start_pose, end_pose, rot_center = pusher_path[pid]
                if rot_center is None:
                    # translational push
                    push = self._compute_translational_push(start_pose, end_pose, obj_body)
                else:
                    # rotational push
                    push = self._compute_rotational_push(start_pose, end_pose, rot_center, obj_body)
                push_trajs.append((push, current_grasp, grasp_path[pid + 1]))
            self.robot.SetDOFVelocityLimits(vel_limits)
            self.robot.Release(obj_body)
            return push_trajs
