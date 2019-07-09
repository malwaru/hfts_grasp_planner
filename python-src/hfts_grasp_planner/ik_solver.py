#! /usr/bin/python

import rospy
import numpy as np
import openravepy as orpy
import trac_ik_python.trac_ik as trac_ik
import hfts_grasp_planner.utils as utils
import hfts_grasp_planner.urdf_loader as urdf_utils_mod


class IKSolver(object):
    """
        This class wraps OpenRAVE's IKFast and Trac_Ik under a common interface.
        If the robot given at construction is not supported by IKFast and its urdf description is available,
        trac_ik is used to solve ik queries.
    """

    def __init__(self, manipulator, urdf_file_name=None, urdf_content=None, timeout=0.005):
        """
            Create a new IKSolver.
            NOTE: This function does not attempt to create a new IKFast model, it only tries to load one.
            If you want to try to create a model, call try_gen_ik_fast().
            NOTE: You can create an IK solver for a tooltip pose relative to the manipulator's end-effector
            by setting the local tool transform. Once created, this IKSolver will not update the transform.
            Hence, if you change the local transform on the manipulator subsequent to calling this constructor,
            this will have no effect on this IKSolver. (TODO does this also hold for IKFast?)
            ---------
            Arguments
            ---------
            manipulator, OpenRAVE manipulator to solve IK for
            urdf_file_name, string (optional) - filename of urdf description file of the robot. If provided and there is no
                IKFast model available for the robot, this is used to load trac_ik.
            urdf_content, string (optional) - alternatively provide the urdf content (not the filename)
            timeout, float (optional) - timeout in seconds for trac-ik
        """
        self._manip = manipulator
        self._robot = self._manip.GetRobot()
        self._env = self._robot.GetEnv()
        self._arm_indices = self._manip.GetArmIndices()
        with self._robot:
            self._robot.SetActiveManipulator(self._manip.GetName())
            self._or_arm_ik = orpy.databases.inversekinematics.InverseKinematicsModel(self._robot,
                                                                                      iktype=orpy.IkParameterization.Type.Transform6D)
        self._trac_ik_solver = None
        self._lower_limits, self._upper_limits = self._robot.GetDOFLimits(self._arm_indices)
        if not self._or_arm_ik.load():
            self._or_arm_ik = None
        if (urdf_file_name is not None or urdf_content is not None) and self._or_arm_ik is None:
            if urdf_content is None:
                if urdf_file_name is None:
                    raise ValueError("Either urdf_file_name or urdf_content must be provided when using trac_ik.")
                with open(urdf_file_name, 'r') as file:
                    urdf_content = file.read()
            tooltip_name = self._manip.GetEndEffector().GetName()
            local_tool_tf = self._manip.GetLocalToolTransform()
            # check whether we have customized tooltip point, if yes, update urdf
            if not np.allclose(local_tool_tf, np.eye(4)):
                tooltip_name = '_custom_tool_tip_pose_'
                urdf_content = urdf_utils_mod.add_grasped_obj(
                    urdf_content, self._manip.GetEndEffector().GetName(), tooltip_name, local_tool_tf)
            # create trac_ik solver
            self._trac_ik_solver = trac_ik.IK(str(self._manip.GetBase().GetName()),
                                              str(tooltip_name),
                                              urdf_string=urdf_content,
                                              timeout=timeout)
            self._trac_ik_solver.set_joint_limits(self._lower_limits, self._upper_limits)
        self._parameters = {  # TODO make this settable?
            'num_trials': 10,
        }

    def try_gen_ik_fast(self):
        """
            Try to generate an IKFast model. Returns True on success and False otherwise
        """
        with self._robot:
            self._robot.SetActiveManipulator(self._manip.GetName())
            self._or_arm_ik = orpy.databases.inversekinematics.InverseKinematicsModel(self._robot,
                                                                                      iktype=orpy.IkParameterization.Type.Transform6D)
            if not self._or_arm_ik.load():
                try:
                    self._or_arm_ik.autogenerate()
                    return True
                except Exception as e:
                    rospy.logwarn("Could not generate IKFast model: " + str(e))
                    self._or_arm_ik = None
            return False

    def compute_ik(self, pose, seed=None, joint_limit_margin=0.0, **kwargs):
        """
            Compute an inverse kinematics solution for the given pose.
            This function does not check for collisions and the returned solution is only guaranteed to be within joint limits.
            ---------
            Arguments
            ---------
            pose, numpy array of shape (4, 4) - end-effector transformation matrix
            seed, numpy array of shape (q,) - initial arm configuration to search from (q is the #DOFs of the arm)
            joint_limit_margin, float - minimal distance from joint limits
            further key word arguments:
                tolerances for ik solver (in target pose frame): bx, by, bz, brx, bry, brz - defaults to small values (only supported with trac_ik)
            -------
            Returns
            -------
            sol, numpy array of shape (q,) - solution (q is #DOFs fo the arm), or None, if no solution found
        """
        with self._robot:
            if self._or_arm_ik:
                if kwargs is not None:
                    rospy.logwarn('IkSolver: Tolerances are only suppported when using trac_ik')
                if seed is not None:
                    self._robot.SetDOFValues(seed, dofindices=self._arm_indices)
                return self._manip.FindIKSolution(pose, orpy.IkFilterOptions.IgnoreCustomFilters)
            elif self._trac_ik_solver is not None:
                if seed is None:
                    rnd = np.random.rand(self._manip.GetArmDOF())
                    seed = self._lower_limits + rnd * (self._upper_limits - self._lower_limits)
                base_pose = self._manip.GetBase().GetTransform()
                inv_base_pose = utils.inverse_transform(base_pose)
                pose_in_base = np.dot(inv_base_pose, pose)
                quat = orpy.quatFromRotationMatrix(pose_in_base)
                sol = self._trac_ik_solver.get_ik(qinit=seed,
                                                  x=pose_in_base[0, 3],
                                                  y=pose_in_base[1, 3],
                                                  z=pose_in_base[2, 3],
                                                  rx=quat[1], ry=quat[2],
                                                  rz=quat[3], rw=quat[0],
                                                  **kwargs)
                if sol is not None:
                    sol = np.array(sol)
                    in_limits = np.logical_and.reduce(np.logical_and(
                        sol >= self._lower_limits + joint_limit_margin, sol <= self._upper_limits - joint_limit_margin))
                    if in_limits:
                        return sol
                    else:
                        return None
                return None
            else:
                raise RuntimeError("Neither IKFast nor TracIK is available. Can not solve IK queries!")

    def compute_collision_free_ik(self, pose, seed=None, joint_limit_margin=0.0, **kwargs):
        """
            Compute a collision-free inverse kinematics solution for the given pose.
            ---------
            Arguments
            ---------
            pose, numpy array of shape (4, 4) - end-effector transformation matrix
            seed, numpy array of shape (q,) - initial arm configuration to search from (q is the #DOFs of the arm)
            joint_limit_margin, float - minimal distance from joint limits
            further key word arguments:
                tolerances for ik solver (in target pose frame): bx, by, bz, brx, bry, brz - defaults to small values (only supported with trac_ik)
            -------
            Returns
            -------
            sol, numpy array of shape (q,) - solution (q is #DOFs fo the arm), or None, if no solution found
            col_free, bool - True if sol is not None and collision-free, else False
        """
        with self._robot:
            if self._or_arm_ik:
                if kwargs is not None:
                    rospy.logwarn('IkSolver: Tolerances are only suppported when using trac_ik')
                if seed is not None:
                    self._robot.SetDOFValues(seed, dofindices=self._arm_indices)
                sol = self._manip.FindIKSolution(pose, orpy.IkFilterOptions.CheckEnvCollisions)
                return sol, sol is not None
            elif self._trac_ik_solver is not None:
                base_pose = self._manip.GetBase().GetTransform()
                inv_base_pose = utils.inverse_transform(base_pose)
                pose_in_base = np.dot(inv_base_pose, pose)
                quat = orpy.quatFromRotationMatrix(pose_in_base)
                np_sol = None
                # trac_ik does not do collision checks, so try multiple times from random initializations
                for i in xrange(self._parameters['num_trials']):
                    if seed is None or i > 0:
                        rnd = np.random.rand(self._manip.GetArmDOF())
                        seed = self._lower_limits + rnd * (self._upper_limits - self._lower_limits)
                    sol = self._trac_ik_solver.get_ik(qinit=seed,
                                                      x=pose_in_base[0, 3],
                                                      y=pose_in_base[1, 3],
                                                      z=pose_in_base[2, 3],
                                                      rx=quat[1], ry=quat[2],
                                                      rz=quat[3], rw=quat[0],
                                                      **kwargs)
                    if sol is not None:
                        np_sol = np.array(sol)
                        in_limits = np.logical_and.reduce(np.logical_and(
                            np_sol >= self._lower_limits + joint_limit_margin, np_sol <= self._upper_limits - joint_limit_margin))
                        if in_limits:
                            # with self._robot:
                            self._robot.SetDOFValues(np_sol, dofindices=self._arm_indices)
                            if not self._env.CheckCollision(self._robot) and not self._robot.CheckSelfCollision():
                                return np_sol, True
                return np_sol, False
            else:
                raise RuntimeError("Neither IKFast nor TracIK is available. Can not solve IK queries!")
