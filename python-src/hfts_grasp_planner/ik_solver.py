#! /usr/bin/python

import rospy
import numpy as np
import openravepy as orpy
import trac_ik_python.trac_ik as trac_ik
import hfts_grasp_planner.utils as utils


class IKSolver(object):
    """
        This class wraps OpenRAVE's IKFast and Trac_Ik under a common interface.
        If the robot given at construction is not supported by IKFast and its urdf description is available,
        trac_ik is used to solve ik queries.
    """

    def __init__(self, manipulator, urdf_file_name=None):
        """
            Create a new IKSolver.
            NOTE: This function does not attempt to create a new IKFast model, it only tries to load one.
            If you want to try to create a model, call try_gen_ik_fast().
            ---------
            Arguments
            ---------
            manipulator, OpenRAVE manipulator to solve IK for
            urdf_file_name, string (optional) - filename of urdf description file of the robot. If provided and there is no
                IKFast model available for the robot, this is used to load trac_ik.
        """
        self._urdf_file_name = urdf_file_name
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
        if urdf_file_name is not None and self._or_arm_ik is None:
            urdf_content = None
            with open(urdf_file_name, 'r') as file:
                urdf_content = file.read()
            self._trac_ik_solver = trac_ik.IK(str(self._manip.GetBase().GetName()),
                                              str(self._manip.GetEndEffector().GetName()),
                                              urdf_string=urdf_content)
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

    def compute_ik(self, pose, seed=None):
        """
            Compute an inverse kinematics solution for the given pose.
            This function does not check for collisions and the returned solution is only guaranteed to be within joint limits.
            ---------
            Arguments
            ---------
            pose, numpy array of shape (4, 4) - end-effector transformation matrix
            seed, numpy array of shape (q,) - initial arm configuration to search from (q is the #DOFs of the arm)
            -------
            Returns
            -------
            sol, numpy array of shape (q,) - solution (q is #DOFs fo the arm), or None, if no solution found
        """
        with self._robot:
            if self._or_arm_ik:
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
                                                  rz=quat[3], rw=quat[0])
                if sol is not None:
                    sol = np.array(sol)
                    in_limits = np.logical_and.reduce(np.logical_and(
                        sol >= self._lower_limits, sol <= self._upper_limits))
                    if in_limits:
                        return sol
                    else:
                        return None
                return None
            else:
                raise RuntimeError("Neither IKFast nor TracIK is available. Can not solve IK queries!")

    def compute_collision_free_ik(self, pose, seed=None):
        """
            Compute a collision-free inverse kinematics solution for the given pose.
            ---------
            Arguments
            ---------
            pose, numpy array of shape (4, 4) - end-effector transformation matrix
            seed, numpy array of shape (q,) - initial arm configuration to search from (q is the #DOFs of the arm)
            -------
            Returns
            -------
            sol, numpy array of shape (q,) - solution (q is #DOFs fo the arm), or None, if no solution found
            col_free, bool - True if sol is not None and collision-free, else False
        """
        with self._robot:
            if self._or_arm_ik:
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
                                                      rz=quat[3], rw=quat[0])
                    if sol is not None:
                        np_sol = np.array(sol)
                        in_limits = np.logical_and.reduce(np.logical_and(
                            np_sol >= self._lower_limits, np_sol <= self._upper_limits))
                        if in_limits:
                            # with self._robot:
                            self._robot.SetDOFValues(np_sol, dofindices=self._arm_indices)
                            if not self._env.CheckCollision(self._robot) and not self._robot.CheckSelfCollision():
                                return np_sol, True
                return np_sol, False
            else:
                raise RuntimeError("Neither IKFast nor TracIK is available. Can not solve IK queries!")
