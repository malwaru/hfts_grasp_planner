import so3hierarchy
import numpy as np
from rtree import index
from utils import inverse_transform


class ReachabilityMap(object):
    """
        Represents what end-effector poses relative to the robot
        frame are reachable.
    """

    def __init__(self, manip, filename, ik_solver):
        """
            Initialize a new reachability map.
            ---------
            Arguments
            ---------
            manip, OpenRAVE manipulator
            filename, string - path to where to load/save reachability map from
            ik_solver, ik_solver.IKSolver - ik solver for the given manipulator
        """
        self._orientation_hierarchies = index.Index(filename)
        # TODO save and load bounding box information
        self._manip = manip
        self._ik_solver = ik_solver

    def create(self, res_metric, res_quat):
        """
            Create the map for the given robot manipulator.
            If the map was already created, this is a null operation.
            ---------
            Arguments
            ---------
            res_metric, float - grid distance in metric space
            res_quat, float - grid distance in rotation space
        """
        # TODO check whether index has already values inside
        robot = self._manip.GetRobot()
        with robot:
            # first compute bounding box
            base_tf = self._manip.GetBase().GetTransform()
            inv_base_tf = inverse_transform(base_tf)
            robot_tf_in_base = np.dot(inv_base_tf, robot.GetTransform())
            robot.SetTransform(robot_tf_in_base)  # set base link to global origin
            maniplinks = ReachabilityMap.get_manipulator_links(self._manip)
            for link in robot.GetLinks():
                link.Enable(link in maniplinks)
            # the axes' anchors are the best way to find the max radius
            # the best estimate of arm length is to sum up the distances of the anchors of all the points in between the chain
            arm_joints = ReachabilityMap.get_ordered_arm_joints(robot, self._manip)
            base_anchor = arm_joints[0].GetAnchor()
            eetrans = self._manip.GetEndEffectorTransform()[0:3, 3]
            arm_length = 0
            for j in arm_joints[::-1]:
                arm_length += np.sqrt(sum((eetrans-j.GetAnchor())**2))
                eetrans = j.GetAnchor()
            # if maxradius is None:
            max_radius = arm_length + res_metric * np.sqrt(3.0) * 1.05
            # TODO sample workspace positions for arm
            # TODO for each position create a kdtree that stores all samples of orientations
            # TODO we can reach for that position. store this kdtree somehwere.
            # TODO save links/pointers to these trees in another kdree

            # TODO query time: for a partition just sample the child nodes and query
            # TODO nearest neighbors
            # sample_positions_per_dim = (np.linspace(-max_radius, max_radius, (int)(2*max_radius / res_metric)))
            # sample_positions = np.array(np.meshgrid(*sample_positions_per_dim)).T.reshape((actual_num_samples, num_dofs))

    @staticmethod
    def get_manipulator_links(manip):
        """
            Copied from OpenRAVEpy.
            Return all links of the given manipulator.
        """
        links = manip.GetChildLinks()
        # add the links connecting to the base link.... although this reduces the freespace of the arm, it is better to have than not (ie waist on humanoid)
        tobasejoints = manip.GetRobot().GetChain(0, manip.GetBase().GetIndex())
        dofindices = [np.arange(joint.GetDOFIndex(), joint.GetDOFIndex()+joint.GetDOF())
                      for joint in tobasejoints if joint.GetDOFIndex() >= 0 and not joint.IsStatic()]
        tobasedofs = np.hstack(dofindices) if len(dofindices) > 0 else np.array([], int)
        robot = manip.GetRobot()
        joints = robot.GetJoints()
        for jindex in np.r_[manip.GetArmIndices(), tobasedofs]:
            joint = joints[jindex]
            if joint.GetFirstAttached() and not joint.GetFirstAttached() in links:
                links.append(joint.GetFirstAttached())
            if joint.GetSecondAttached() and not joint.GetSecondAttached() in links:
                links.append(joint.GetSecondAttached())
        # don't forget the rigidly attached links
        for link in links[:]:
            for newlink in link.GetRigidlyAttachedLinks():
                if newlink not in links:
                    links.append(newlink)
        return links

    @staticmethod
    def get_ordered_arm_joints(robot, manip):
        return [j for j in robot.GetDependencyOrderedJoints() if j.GetJointIndex() in manip.GetArmIndices()]
