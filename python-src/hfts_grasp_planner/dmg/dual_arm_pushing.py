import numpy as np
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
        self.robot = robot
        self.grasping_manip = grasping_manip
        self.pushing_manip = pushing_manip
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

    def _compute_translational_push(self, start_pose, end_pose, offset=0.03):
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
        start_config = self._ik_solver.compute_collision_free_ik(approach_pose)
        if start_config is None:
            raise RuntimeError("Could not compute collision-free ik solution for approach pose of translational push")
        # move to start configuration using motion planner to avoid colliding with the object
        manip_dofs = self.pushing_manip.GetArmIndices()
        self.robot.SetActiveDOFs(manip_dofs)
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
        while True:
            wTp = self.pushing_manip.GetEndEffectorTransform()
            pushing_dir = world_end[:3, 3] - wTp[:3, 3]
            pushing_distance = np.linalg.norm(pushing_dir)
            if pushing_distance <= 0.0:
                break
            pushing_dir /= pushing_distance
            cart_dir[:3] = pushing_dir
            # compute Jacobian
            jacobian = np.empty((6, self.pushing_manip.GetArmDOF()))
            jacobian[:3] = self.pushing_manip.CalculateJacobian()
            jacobian[3:] = self.pushing_manip.CalculateAngularVelocityJacobian()
            # compute pseudo inverse
            inv_jac, rank = compute_pseudo_inverse_rank(jacobian)
            if rank < 6:  # if we are in a singularity, we failed
                raise RuntimeError("Failed to push translationally. Ran into a singularity")
            dq = np.matmul(inv_jac, cart_dir)
            config_path.append(config)
            config += step_size * dq
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
        assert(len(grasp_path) == len(pusher_path) + 1)
        with self.robot:
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
