import numpy as np
import openravepy as orpy
from hfts_grasp_planner.utils import set_grasp, inverse_transform, path_to_trajectory, compute_pseudo_inverse_rank
from hfts_grasp_planner.ik_solver import IKSolver
import hfts_grasp_planner.external.transformations as tf_mod


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
        self.vel_factor = 0.05
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

    def _compute_translational_push(self, start_pose, end_pose, offset=0.03, err_tol=1e-3):
        """
            Compute a pushing path in joint space for a translational push from start_pose to end_pose.
        """
        trajs = []
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
            raise RuntimeError("Could not compute collision-free ik solution for approach pose of translational push")
        # move to start configuration using motion planner to avoid colliding with the object
        approach_traj = self.mplanner.MoveActiveJoints(start_config, execute=False, outputtraj=True, outputtrajobj=True)
        if approach_traj is None:
            raise RuntimeError("Failed to plan a motion to the approach pose of translational push")
        trajs.append(approach_traj)
        # now start the actual push
        self.robot.SetActiveDOFValues(start_config)
        config = np.array(start_config)
        config_path = [start_config]
        cart_dir = np.zeros(6)
        step_size = 0.01
        while pushing_distance >= err_tol:
            wTp = self.pushing_manip.GetEndEffectorTransform()
            position_err = world_end[:3, 3] - wTp[:3, 3]
            pushing_distance = np.linalg.norm(position_err)
            cart_dir[:3] = position_err
            # compute Jacobian
            jacobian = np.empty((6, self.pushing_manip.GetArmDOF()))
            jacobian[:3] = self.pushing_manip.CalculateJacobian()
            jacobian[3:] = self.pushing_manip.CalculateAngularVelocityJacobian()
            # compute pseudo inverse
            inv_jac, rank = compute_pseudo_inverse_rank(jacobian)
            if rank < 6:  # if we are in a singularity, we failed
                raise RuntimeError("Failed to push translationally. Ran into a singularity")
            dq = np.matmul(inv_jac, cart_dir)
            delta_q_norm = np.linalg.norm(dq)
            if delta_q_norm <= 1e-4:
                break
            dq /= delta_q_norm
            config += min(step_size, delta_q_norm) * dq
            config_path.append(config)
            self.robot.SetActiveDOFValues(config)
        push_traj = path_to_trajectory(self.robot, config_path, vel_factor=0.05)
        return [approach_traj, push_traj]

    def _compute_rotational_push(self, start_pose, end_pose, rot_center):
        """
            Compute a pushing path in joint space for a rotational psuh from start_pose to end_pose.
            The rotational push is centered at rot_center.
        """
        # TODO compute approach motion
        return []

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
            trajs, list of OpenRAVE trajectories
        """
        assert(len(grasp_path) == len(pusher_path) + 1)
        # with self.robot:
        self.pushing_manip.SetLocalToolTransform(self.pushing_tooltip)
        manip_dofs = self.pushing_manip.GetArmIndices()
        self.robot.SetActiveDOFs(manip_dofs)
        vel_limits = self.robot.GetDOFVelocityLimits()
        self.robot.SetDOFVelocityLimits(self.vel_factor * vel_limits)
        arm_trajs = []
        for pid in xrange(len(pusher_path)):
            # set current state of object
            current_grasp = grasp_path[pid]
            set_grasp(self.grasping_manip, obj_body, current_grasp.eTo, current_grasp.config)
            start_pose, end_pose, rot_center = pusher_path[pid]
            if rot_center is None:
                # translational push
                push = self._compute_translational_push(start_pose, end_pose)
            else:
                # rotational push
                push = self._compute_rotational_push(start_pose, end_pose, rot_center)
            arm_trajs.extend(push)
        return arm_trajs
