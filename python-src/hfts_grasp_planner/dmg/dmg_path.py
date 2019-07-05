import numpy as np
import math
import hfts_grasp_planner.external.transformations as tf_utils
from pycaster import pycaster


def axisAngle(axis, angle):
    rrt = np.outer(axis, axis)
    Sr = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = rrt + (np.identity(3) - rrt)*math.cos(angle) + Sr*math.sin(angle)
    mat = np.identity(4)
    mat[0:3, 0:3] = R
    return mat


def matrixToVectorPose(matrix):
    vec = np.empty(7,)
    vec[3:] = tf_utils.quaternion_from_matrix(matrix)
    # q = q/np.linalg.norm(q)
    vec[:3] = matrix[:3, 3]
    return vec


class DMGPushPath():
    """class to convert from DMG path to pusher gripper path"""

    def __init__(self, object_file_name):
        self._object_shape_file = object_file_name
        self._caster = pycaster.rayCaster.fromSTL(object_file_name, scale=1.0)
        self._grasping_z_axis = None

    def rayShapeIntersections(self, start_point, goal_point):
        '''get the intersection of a ray with the object's shape'''
        intersections = self._caster.castRay(1000.0*start_point, 1000.0*goal_point)
        return intersections

    def translationContactPoint(self, translation, current_middle_point):
        '''returns the contact point for pushing the object to obtain the translation'''
        # translation (from DMG) is the motion of the finger on the object
        start_point = current_middle_point
        end_point = current_middle_point + 1000*translation  # the end point must be ouside of the object, along the translation line
        possible_push_points = self.rayShapeIntersections(start_point, end_point)
        if len(possible_push_points) < 1:
            return None
        push_point = np.array(possible_push_points[-1])/1000.0  # get the most external one
        return push_point

    def rotationContactPoint(self, rotation, current_middle_point, current_grasp_pose, side=1):
        '''returns the contact point for rotating the object to obtain the translation'''
        start_point = current_middle_point
        z_axis = current_grasp_pose[0:3, 2:3].T[0]
        z_point = start_point + 1000*z_axis
        intersections = self.rayShapeIntersections(start_point, z_point)
        if len(intersections) < 1:
            return None, None
        a_point = np.array(intersections[-1])/1000.0
        # now slide inside the object for like 2 cm?
        b_point = a_point - 0.02*z_axis
        sgn = 1
        if rotation < 0:
            sgn = -1
        y_axis = current_grasp_pose[0:3, 1:2].T[0]
        y_point = b_point + 1000*sgn*side*y_axis
        intersections = self.rayShapeIntersections(b_point, y_point)
        if len(intersections) < 1:
            return None, None
        push_point = np.array(intersections[-1])/1000.0
        return push_point, b_point

    def translationPushPoses(self, translation, object_to_fixed_gripper, current_fingertip_contacts):
        '''returns a sequence of push poses w.r.t. the initial grasp pose for executing translation'''
        current_middle_point = (current_fingertip_contacts[0] + current_fingertip_contacts[1])/2.0
        push_point = self.translationContactPoint(translation, current_middle_point)
        if push_point is None:
            return None, None
        # the x axis of the pusher is aligned with the translation
        x_axis = translation/np.linalg.norm(translation)
        # the z axis of the pusher is aligned with the axis between the two fingertip contacts
        z_axis = current_fingertip_contacts[1] - current_fingertip_contacts[0]
        z_axis = z_axis/np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)

        # transformation matrix containing the pose of the gripper pusher at the beginning
        T_start = np.array([np.append(x_axis, 0), np.append(y_axis, 0),
                            np.append(z_axis, 0), np.append(push_point, 1)]).T
        # transformation matrix containing the pose of the gripper pusher at the end (after translation)
        T_end = np.array([np.append(x_axis, 0), np.append(y_axis, 0), np.append(
            z_axis, 0), np.append(push_point - translation, 1)]).T

        # now these poses must be transformed into the fixed reference frame
        g_T_start = np.dot(object_to_fixed_gripper, T_start)
        g_T_end = np.dot(object_to_fixed_gripper, T_end)
        # interpolate_translations()
        return g_T_start, g_T_end

    def rotationPushPoses(self, rotation, object_to_fixed_gripper, current_grasp_pose, current_fingertip_contacts):
        '''returns a sequence of push poses w.r.t. the initial grasp pose for executing rotation'''
        current_middle_point = (current_fingertip_contacts[0] + current_fingertip_contacts[1])/2.0
        push_point, b_point = self.rotationContactPoint(rotation, current_middle_point, current_grasp_pose, 1)
        if push_point is None:
            return None, None

        push_point_to_object = np.eye(4)
        push_point_to_object[0, 3] = push_point[0]
        push_point_to_object[1, 3] = push_point[1]
        push_point_to_object[2, 3] = push_point[2]

        # the x axis of the pusher is aligned with the direction from b to push point
        v = push_point - b_point
        x_axis = v/np.linalg.norm(v)
        # print x_axis
        # the z axis of the pusher is aligned with the axis between the two fingertip contacts
        z_axis = current_fingertip_contacts[1] - current_fingertip_contacts[0]
        z_axis = z_axis/np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)

        push_point_to_object = np.array([np.append(x_axis, 0), np.append(
            y_axis, 0), np.append(z_axis, 0), np.array([0, 0, 0, 1])]).T
        push_point_to_object[0, 3] = push_point[0]
        push_point_to_object[1, 3] = push_point[1]
        push_point_to_object[2, 3] = push_point[2]

        push_point_to_gripper = np.dot(object_to_fixed_gripper, push_point_to_object)
        T_start = push_point_to_gripper

        current_push_to_fingertip = np.dot(self._gripper_to_fingertip, push_point_to_gripper)
        x_axis = np.array([1, 0, 0])
        next_push_to_fingertip = np.dot(axisAngle(x_axis, rotation), current_push_to_fingertip)
        next_push_to_gripper = np.dot(self._fingertip_to_gripper, next_push_to_fingertip)
        T_end = next_push_to_gripper
        return T_start, T_end

    def convert_path(self, inhand_path, initial_grasp_pose, fingertip_contacts):
        """
            Convert a given inhand path into a sequence of pushing poses for the second end-effector.
            An inhand path is a sequence of translations and rotations. A translation is a directional
            vector in the original object frame. A rotation is an angle in radians around the normal
            at the fingertip contact.
            ---------
            Arguments
            ---------
            inhand_path, tuple (translations, rotations), 
                where translations is a list of np.arrays of shape (3,)
                rotations is a list of floats  - Both lists are assumed to have the same length
            initial_grasp_pose, np.array of shape (4, 4) - the pose of the grasping gipper in the original object frame oTe 
            fingertip_contacts, np.array of shape (2, 3) - the initial positions of the figertips in the original object frame
            -------
            Returns
            -------
            pushes, a list of tuples (start, end, rot_center), 
                where start and end are np.array of shape (4, 4) representing the initial and final pose of the pusher
                rot_center is either None or a np.array of shape (3,) denoting the center of rotation for a rotational push
        """
        # '''given inhand_path (sequence of translations and rotations) returns poses for the end effector'''
        # initial grasp is a rotation matrix with the grasping gripper frame w.r.t. the object
        # convention: z-axis is along the fingers, x-axis between the fingers
        self._grasping_z_axis = initial_grasp_pose[0:3, 2:3]
        fingertip_middle_point = (fingertip_contacts[0] + fingertip_contacts[1])/2.0
        self._initial_grasp_pose = initial_grasp_pose

        translations = inhand_path[0]
        rotations = inhand_path[1]

        current_fingertip_contacts = np.array(fingertip_contacts)
        current_object_to_gripper = np.linalg.inv(initial_grasp_pose)
        current_grasp_pose = initial_grasp_pose

        pusher_poses = []

        self._gripper_to_fingertip = np.eye(4)
        self._gripper_to_fingertip[2, 3] = - \
            np.linalg.norm(fingertip_middle_point - current_object_to_gripper[0:3, 3:4].T[0])
        self._fingertip_to_gripper = np.linalg.inv(self._gripper_to_fingertip)

        # the first rotation
        # if len(rotations) > 0:
        #     r = rotations[0]
        #     if abs(r) > 0.00001:
        #         # print current_object_to_gripper
        #         pose1, pose2 = self.rotationPushPoses(r, current_object_to_gripper,
        #                                               current_grasp_pose, current_fingertip_contacts)
        #         if pose1 is not None:
        #             # TODO rotation center
        #             pusher_poses.append((pose1, pose2, None))
        #         # the fingertip contacts do not vary, but the pose does (about the x axis, around fingertip contact point)
        #         # x_axis = current_grasp_pose[0:3, 0:1].T[0]
        #         # print x_axis
        #         x_axis = np.array([1, 0, 0])
        #         current_object_to_fingertip = np.dot(self._gripper_to_fingertip, current_object_to_gripper)
        #         current_object_to_fingertip = np.dot(axisAngle(x_axis, r), current_object_to_fingertip)
        #         current_object_to_gripper = np.dot(self._fingertip_to_gripper, current_object_to_fingertip)
        #         current_grasp_pose = np.linalg.inv(current_object_to_gripper)
        #         # pusher_poses.append(matrixToVectorPose(current_object_to_gripper))

        for t, r in zip(translations, rotations):
            if np.linalg.norm(t, ord=1) > 0.001:
                # print current_object_to_gripper
                pose1, pose2 = self.translationPushPoses(t, current_object_to_gripper, current_fingertip_contacts)
                # the fingertips have translated on the object:
                current_fingertip_contacts[0] = current_fingertip_contacts[0] + t
                current_fingertip_contacts[1] = current_fingertip_contacts[1] + t
                # also, the grasping gripper on the object has moved
                current_grasp_pose[0, 3] = current_grasp_pose[0, 3] + t[0]
                current_grasp_pose[1, 3] = current_grasp_pose[1, 3] + t[1]
                current_grasp_pose[2, 3] = current_grasp_pose[2, 3] + t[2]
                current_object_to_gripper = np.linalg.inv(current_grasp_pose)
                if pose1 is not None:
                    pusher_poses.append((pose1, pose2, None))

            # after a translation comes a rotation
            if abs(r) > 0.00001:
                pose1, pose2 = self.rotationPushPoses(r, current_object_to_gripper,
                                                      current_grasp_pose, current_fingertip_contacts)
                if pose1 is not None:
                    pusher_poses.append((pose1, pose2, np.mean(current_fingertip_contacts, axis=0)))
                # the fingertip contacts do not vary, but the pose does (around the x axis)
                x_axis = np.array([1, 0, 0])
                current_object_to_fingertip = np.dot(self._gripper_to_fingertip, current_object_to_gripper)
                current_object_to_fingertip = np.dot(axisAngle(x_axis, r), current_object_to_fingertip)
                current_object_to_gripper = np.dot(self._fingertip_to_gripper, current_object_to_fingertip)
                current_grasp_pose = np.linalg.inv(current_object_to_gripper)
        return pusher_poses
