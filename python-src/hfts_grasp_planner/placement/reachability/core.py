import rospy
import time
import numpy as np
import openravepy as orpy
from functools import partial
from itertools import product
import sklearn.neighbors as skn
import pose_distance
import hfts_grasp_planner.placement.so3hierarchy as so3hierarchy
import hfts_grasp_planner.yumi_reachability as gnat_reachability
from hfts_grasp_planner.utils import inverse_transform, transform_pos_quats_by, get_manipulator_links,\
    get_ordered_arm_joints

YUMI_SE3_WEIGHT = 0.07131
YUMI_MAX_RADIUS = 0.07131
YUMI_MIN_RADIUS = 0.02
YUMI_DELTA_SQ_RADII = YUMI_MIN_RADIUS**2 - YUMI_MAX_RADIUS**2


class ReachabilityMap(object):
    """
        Represents what end-effector poses relative to the robot
        frame are reachable.
        TODO: Because sklearn data structures do not allow to save any data anotations, this
        TODO: class currently does not save arm configurations. (too much book keeping)
    """

    def __init__(self, manip, ik_solver):
        """
            Initialize a new reachability map.
            ---------
            Arguments
            ---------
            manip, OpenRAVE manipulator
            ik_solver, ik_solver.IKSolver - ik solver for the given manipulator
        """
        self._quat_balltrees = []
        self._cart_kdtree = None
        # self._configurations = None
        # self._positions = None
        self._manip = manip
        self._ik_solver = ik_solver
        self._eucl_dispersion = 0.0
        self._so3_dispersion = 0.0

    def create(self, res_metric, so3depth):
        """
            Create the map for the given robot manipulator.
            If the map was already created, this is a null operation.
            ---------
            Arguments
            ---------
            res_metric, float - grid distance in metric space
            so3depth, int - depth in so3hierarchy to sample in (0 = 72 orientations, >1 = 72 * 8^so3depth orientations)
        """
        if self._cart_kdtree is not None:
            return
        robot = self._manip.GetRobot()
        assert(not robot.CheckSelfCollision())
        with robot:
            # env = robot.GetEnv()
            # first compute bounding box
            base_tf = self._manip.GetBase().GetTransform()
            inv_base_tf = inverse_transform(base_tf)
            robot_tf_in_base = np.dot(inv_base_tf, robot.GetTransform())
            robot.SetTransform(robot_tf_in_base)  # set base link to global origin
            maniplinks = get_manipulator_links(self._manip)
            arm_indices = self._manip.GetArmIndices()
            for link in robot.GetLinks():
                link.Enable(link in maniplinks)
            # the axes' anchors are the best way to find the max radius
            # the best estimate of arm length is to sum up the distances of the anchors of all the points in between the chain
            arm_joints = get_ordered_arm_joints(robot, self._manip)
            base_anchor = arm_joints[0].GetAnchor()
            eetrans = self._manip.GetEndEffectorTransform()[0:3, 3]
            arm_length = 0
            for j in arm_joints[::-1]:
                arm_length += np.sqrt(sum((eetrans-j.GetAnchor())**2))
                eetrans = j.GetAnchor()
            # if maxradius is None:
            max_radius = arm_length + res_metric * np.sqrt(3.0) * 1.05
            # sample workspace positions for arm
            sample_positions_per_dim = (np.linspace(-max_radius, max_radius, (int)(2*max_radius / res_metric)))
            # save grid distances
            self._eucl_dispersion = np.sqrt(3) * res_metric / 2.0
            self._so3_dispersion = so3hierarchy.get_dispersion_bound(so3depth)
            # create a distance metric for quaternions
            quat_metric = skn.DistanceMetric.get_metric('pyfunc', func=so3hierarchy.quat_distance)
            configs = []
            reachable_positions = []
            num_total_poses = len(sample_positions_per_dim)**3 * 72 * 8**so3depth
            rospy.loginfo("Creating Simple reachability map. Going to evaluate %i poses." % num_total_poses)
            pose_id = 0
            for rel_pos in product(sample_positions_per_dim, repeat=3):
                # for each position create a kdtree that stores all reachable orientations
                pos = rel_pos + base_anchor
                reachable_quats = []
                quats = (so3hierarchy.get_quaternion(key) for key in so3hierarchy.get_key_generator(so3depth))
                # quats = [np.array([1, 0, 0, 0])]
                # check which of these orietnations we can reach for the given position
                for quat in quats:
                    pose = orpy.matrixFromQuat(quat)
                    pose[:3, 3] = pos
                    # handle = orpy.misc.DrawAxes(env, pose)
                    config = self._ik_solver.compute_ik(pose)
                    if config is not None:
                        robot.SetDOFValues(config, arm_indices)
                        if not robot.CheckSelfCollision():
                            configs.append(config)
                            reachable_quats.append(quat)
                    else:
                        rospy.logdebug("Could not find any ik solution for pos " +
                                       str(pos) + " and quat " + str(quat))
                    pose_id += 1
                    rospy.logdebug("Tested %i of %i total poses. Found %i reachable poses so far." %
                                   (pose_id, num_total_poses, len(configs)))
                if len(reachable_quats) > 0:
                    self._quat_balltrees.append(skn.BallTree(np.array(reachable_quats),
                                                             metric=quat_metric))  # TODO tune leaf_size?
                    reachable_positions.append(pos)

        if len(reachable_positions) > 0:
            positions = np.array(reachable_positions)
            self._cart_kdtree = skn.KDTree(positions)
            # self._configurations = np.array(configs)
            rospy.loginfo("Created reachability map: %i reachable positions, %i reachable poses" %
                          (positions.shape[0], len(configs)))
        else:
            rospy.logerr("There is not a single reachable pose for this manipulator. Something must be wrong")

    def save(self, filename):
        """
            Save this reachability map to file.
            ---------
            Arguments
            ---------
            filename, string - path to where to load/save reachability map from
        """
        data_to_dump = np.array([self._manip.GetRobot().GetName(), self._manip.GetName(),
                                 self._eucl_dispersion, self._so3_dispersion,
                                 self._cart_kdtree, self._quat_balltrees], dtype=object)
        np.save(filename, data_to_dump)

    def load(self, filename):
        """
            Load reachability map from file.
        """
        rname, mname, eucl_disp, so3_disp, cart_kdtree, quat_balltrees = np.load(filename)
        if self._manip.GetRobot().GetName() != rname:
            rospy.logerr("Could not load reachability map from file %s because it is made for a different robot" % filename)
            rospy.logerr("This map was created for robot %s. The file is for robot %s" %
                         (self._manip.GetRobot().GetName(), rname))
            return
        if self._manip.GetName() != mname:
            rospy.logerr("Could not load reachability map from file %s because it is made for a different manipluator" % filename)
            rospy.logerr("This map was created for manipulator %s. The file is for manipulator %s" %
                         (self._manip.GetName(), mname))
            return
        # self._positions = positions
        # self._configurations = configs
        self._cart_kdtree = cart_kdtree
        self._quat_balltrees = quat_balltrees
        self._eucl_dispersion = eucl_disp
        self._so3_dispersion = so3_disp

    def query(self, poses):
        """
            Query nearest reachable poses.
            ---------
            Arguments
            ---------
            poses, numpy array of shape (n, 7) - (x,y,z, w, i, k, j) i.e. position followed by quaternion in world frame
            -------
            Returns
            -------
            cart_distances, numpy array of shape (n,) - cartesian distances between respective query pose and closest reachable pose
            quat_distances, numpy array of shape (n,) - quaternion distance between respective query pose and closest reachable pose
            # nearest_poses, numpy array of shape (n, 7) TODO not supported (too much book keeping)
        """
        # create array to store nearest poses in
        # nearest_poses = np.empty(poses.shape)
        robot = self._manip.GetRobot()
        with robot:
            # transform query poses from world frame to robot frame
            robot_tf = robot.GetTransform()
            inv_robot_tf = inverse_transform(robot_tf)
            poses = transform_pos_quats_by(inv_robot_tf, poses)
            # query for closest positions
            cart_distances, ball_indices = self._cart_kdtree.query(poses[:, :3])
            ball_indices = ball_indices.reshape((ball_indices.shape[0],))
            cart_distances = cart_distances.reshape((cart_distances.shape[0],))
            # nearest_poses[:, :3] = self._positions[ball_indices]
            # create array to save quaternion distances
            quat_distances = np.empty((poses.shape[0],))
            # query for each position only once
            orientation_queries = {}
            for pose_idx, bidx in enumerate(ball_indices):
                if bidx not in orientation_queries:
                    orientation_queries[bidx] = []
                orientation_queries[bidx].append(pose_idx)
            # run over all orientation queries
            for bidx, pose_indices in orientation_queries.iteritems():
                np_indices = np.array(pose_indices)
                tdistances, _ = self._quat_balltrees[bidx].query(poses[np_indices, 3:])
                quat_distances[np_indices] = tdistances.reshape((tdistances.shape[0],))
            return cart_distances, quat_distances

    def get_dispersion(self):
        """
            Return the dispersion of this reachability map.
            The dispersion is the maximum distance any reachable pose can be away
            from a sample stored in this map.
            -------
            Returns
            -------
            eucl_dist, float - upper bound for end-effector position distance
            so3_dist, float - upper bound for end-effector orientation distance
        """
        return self._eucl_dispersion, self._so3_dispersion

    def visualize(self, env, min_reachability=0.0, sprite_size=5.0, reachable_col=[0.8, 0.8, 0.5, 0.5], unreachable_col=[1.0, 0.0, 0.0, 0.5]):
        if self._cart_kdtree is None:
            return None
        max_balls = max([len(self._quat_balltrees[bidx].data) for bidx in xrange(len(self._cart_kdtree.data))])
        robot_tf = self._manip.GetRobot().GetTransform()
        positions = np.dot(np.array(self._cart_kdtree.data), robot_tf[:3, :3].transpose()) + robot_tf[:3, 3]
        num_reachable = np.array([len(self._quat_balltrees[bidx].data)
                                  for bidx in xrange(len(self._cart_kdtree.data))], dtype=np.float_)
        rel_reachable = num_reachable / max_balls
        good_reachability = np.nonzero(rel_reachable > min_reachability)[0]
        colors = rel_reachable[good_reachability, np.newaxis] * reachable_col + \
            (1.0 - rel_reachable)[good_reachability, np.newaxis] * unreachable_col
        return env.plot3(positions[good_reachability], sprite_size, colors)


def hard_coded_distance_fn(a, b):
    """
        SE(3) pose distance for Yumi. Hardcoding this distance function (i.e. the weight) is the
        only way pickling ball tree seems to work.
    """
    return np.linalg.norm(a[:3] - b[:3]) + YUMI_SE3_WEIGHT * so3hierarchy.quat_distance(a, b)


def yumi_distance_fn(a, b):
    # extract quaternions
    q_a, q_b = a[3:], b[3:]
    # express everything in frame a
    q_a_inv = np.array(q_a)
    q_a_inv[1:] *= -1.0
    q_b_in_a = orpy.quatMult(q_b, q_a_inv)
    # rotation distance is determined by the radius of the end-effector. We approximate the end-effector by an ellipse
    angle = 2.0 * np.arccos(np.clip(np.abs(q_b_in_a[0]), 0.0, 1.0))
    radius = 0.0
    if not np.isclose(angle, 0.0):
        # compute dot product of angle between rotation axis and z axis
        cos_alpha = q_b_in_a[3] / np.sin(angle)
        radius = np.sqrt(max(YUMI_MAX_RADIUS**2 + cos_alpha**2 * YUMI_DELTA_SQ_RADII, 0.0))
    return np.linalg.norm(a[:3] - b[:3]) + radius * angle


class SimpleReachabilityMap(object):
    """
        Represents what end-effector poses relative to the robot frame are reachable.
        A SimpleReachabilityMap in contrast to a ReachabilityMap stores poses as a joint vector,
        thus it requires a weighting factor between distance in SO(3) and distance in R^3 to compute
        nearest neighbors.
    """

    def __init__(self, manip, nn_type=None):  # , weight=None):
        """
            Initialize a new simple reachability map.
            ---------
            Arguments
            ---------
            manip, OpenRAVE manipulator
            nn_type, string - Type for nearest neighbor data structure. May either be
                'gnat', 'pyball', 'cball' or None in which case it is set to 'gnat'.
            # weight (optional), float - weight factor to scale between R^3 distance and SO(3) distance
            #     If not provided (None), the factor is set to the radius of the minimal bounding sphere of
            #     the manipulator's gripper.
        """
        self._reachable_poses = None
        self._nn = None
        self._configurations = None
        self._manip = manip
        self._eucl_dispersion = 0.0
        self._so3_dispersion = 0.0
        if nn_type not in ['gnat', 'pyball', 'cball']:
            self._nn_type = 'gnat'
        else:
            self._nn_type = nn_type
        # if weight is None:
        # weight = np.linalg.norm(self._manip.GetEndEffector().ComputeLocalAABB().extents())
        self._weight = YUMI_SE3_WEIGHT

    def create(self, res_metric, so3depth):
        """
            Create the map for the given robot manipulator.
            If the map was already created, this is a no-op.
            ---------
            Arguments
            ---------
            res_metric, float - grid distance in metric space
            so3depth, int - depth in so3hierarchy to sample in (0 = 72 orientations, >1 = 72 * 8^so3depth orientations)
            ik_solver, ik_solver.IKSolver - ik solver for the given manipulator
        """
        if self._nn is not None:
            return
        robot = self._manip.GetRobot()
        assert(not robot.CheckSelfCollision())
        with robot:
            # save grid distances
            self._eucl_dispersion = np.sqrt(3) * res_metric / 2.0
            self._so3_dispersion = so3hierarchy.get_dispersion_bound(so3depth)
            # first compute bounding box
            base_tf = self._manip.GetBase().GetTransform()
            inv_base_tf = inverse_transform(base_tf)
            robot_tf_in_base = np.dot(inv_base_tf, robot.GetTransform())
            robot.SetTransform(robot_tf_in_base)  # set base link to global origin
            maniplinks = get_manipulator_links(self._manip)
            arm_indices = self._manip.GetArmIndices()
            for link in robot.GetLinks():
                link.Enable(link in maniplinks)
            # the axes' anchors are the best way to find the max radius
            # the best estimate of arm length is to sum up the distances of the anchors of all the points in between the chain
            arm_joints = get_ordered_arm_joints(robot, self._manip)
            base_anchor = arm_joints[0].GetAnchor()
            eetrans = self._manip.GetEndEffectorTransform()[0:3, 3]
            arm_length = 0
            for j in arm_joints[::-1]:
                arm_length += np.sqrt(sum((eetrans-j.GetAnchor())**2))
                eetrans = j.GetAnchor()
            self._configurations = []
            self._reachable_poses = []
            # if maxradius is None:
            max_radius = arm_length + res_metric * np.sqrt(3.0) * 1.05
            # sample workspace positions for arm
            sample_positions_per_dim = (np.linspace(-max_radius, max_radius, (int)(2*max_radius / res_metric)))
            # run over grid positions
            num_total_poses = len(sample_positions_per_dim)**3 * 72 * 8**so3depth
            rospy.loginfo("Creating Simple reachability map. Going to evaluate %i poses." % num_total_poses)
            pose_id = 0
            for rel_pos in product(sample_positions_per_dim, repeat=3):
                pos = rel_pos + base_anchor
                # run over grid orientations
                quats = (so3hierarchy.get_quaternion(key) for key in so3hierarchy.get_key_generator(so3depth))
                # check which of these orientations we can reach for the given position
                for quat in quats:
                    pose = orpy.matrixFromQuat(quat)
                    pose[:3, 3] = pos
                    config = ik_solver.compute_ik(pose)
                    if config is not None:
                        robot.SetDOFValues(config, arm_indices)
                        if not robot.CheckSelfCollision():
                            self._configurations.append(config)
                            pose_array = np.empty(7)
                            pose_array[:3] = pos
                            pose_array[3:] = quat
                            self._reachable_poses.append(pose_array)
                    else:
                        rospy.logdebug("Could not find any ik solution for pos " +
                                       str(pos) + " and quat " + str(quat))
                    pose_id += 1
                    rospy.logdebug("Tested %i of %i total poses. Found %i reachable poses so far." %
                                   (pose_id, num_total_poses, len(self._reachable_poses)))
        if len(self._reachable_poses) > 0:
            self._create_nn(self._reachable_poses)
            self._configurations = np.array(self._configurations)
            rospy.loginfo("Created simple reachability map: %i reachable poses" %
                          (len(self._configurations)))
        else:
            rospy.logerr("There is not a single reachable pose for this manipulator. Something must be wrong")

    def save(self, filename):
        """
            Save this reachability map to file.
            ---------
            Arguments
            ---------
            filename, string - path to where to load/save reachability map from
        """
        dumpable_nn = self._nn if self._nn_type in ['pyball', 'cball'] else None
        data_to_dump = np.array([self._manip.GetRobot().GetName(), self._manip.GetName(),
                                 self._eucl_dispersion, self._so3_dispersion,
                                 self._configurations, self._reachable_poses, dumpable_nn, self._nn_type], dtype=object)
        np.save(filename, data_to_dump)

    def load(self, filename):
        """
            Load reachability map from file.
        """
        start_time = time.time()
        rname, mname, eucl_disp, so3_disp, configs, poses, nn, nn_type = np.load(filename)
        if self._manip.GetRobot().GetName() != rname:
            rospy.logerr("Could not load reachability map from file %s because it is made for a different robot" % filename)
            rospy.logerr("This map was created for robot %s. The file is for robot %s" %
                         (self._manip.GetRobot().GetName(), rname))
            return
        if self._manip.GetName() != mname:
            rospy.logerr("Could not load reachability map from file %s because it is made for a different manipluator" % filename)
            rospy.logerr("This map was created for manipulator %s. The file is for manipulator %s" %
                         (self._manip.GetName(), mname))
            return
        self._configurations = configs
        self._eucl_dispersion = eucl_disp
        self._so3_dispersion = so3_disp
        self._weight = YUMI_SE3_WEIGHT
        if self._nn_type != nn_type:
            rospy.logwarn("The reachability map that was requested to be loaded had nearest neighbor type %s, but this instance has %s" % (
                nn_type, self._nn_type))
            rospy.logwarn("Creating new nearest neighbor structure from loaded poses.")
            self._create_nn(poses)
        elif self._nn_type == 'gnat':
            self._create_nn(poses)
        else:
            self._reachable_poses = poses
            self._nn = nn
        rospy.logdebug("[SimpleReachabilityMap] Loading took %fs" % (time.time() - start_time))

    # def load(self, filename):
    #     """
    #         Load reachability map from file.
    #     """
    #     start_time = time.time()
    #     rname, mname, eucl_disp, so3_disp, weight, configs, poses = np.load(filename)
    #     rospy.logdebug("[SimpleReachabilityMap] Loading took %fs" % (time.time() - start_time))
    #     if self._manip.GetRobot().GetName() != rname:
    #         rospy.logerr("Could not load reachability map from file %s because it is made for a different robot" % filename)
    #         rospy.logerr("This map was created for robot %s. The file is for robot %s" %
    #                      (self._manip.GetRobot().GetName(), rname))
    #         return
    #     if self._manip.GetName() != mname:
    #         rospy.logerr("Could not load reachability map from file %s because it is made for a different manipluator" % filename)
    #         rospy.logerr("This map was created for manipulator %s. The file is for manipulator %s" %
    #                      (self._manip.GetName(), mname))
    #         return
    #     self._configurations = configs
    #     self._eucl_dispersion = eucl_disp
    #     self._so3_dispersion = so3_disp
    #     self._weight = weight
    #     start_time = time.time()
    #     self._create_ball_tree(poses)
    #     rospy.logdebug("[SimpleReachabilityMap] Creating ball tree took %fs" % (time.time() - start_time))

    def query(self, poses):
        """
            Query nearest reachable poses.
            ---------
            Arguments
            ---------
            poses, numpy array of shape (n, 7) - (x,y,z, w, i, k, j) i.e. position followed by quaternion in world frame
            -------
            Returns
            -------
            distances, numpy array of shape (n,) - distances between respective query pose and closest reachable pose
            nearest_poses, numpy array of shape (n, 7) - the closest reachable poses
            configs, numpy array of shape (n, 7) - arm configurations for closest reachable poses
        """
        start_time = time.time()
        robot = self._manip.GetRobot()
        with robot:
            # transform query poses from world frame to robot frame
            robot_tf = robot.GetTransform()
            inv_robot_tf = inverse_transform(robot_tf)
            poses = transform_pos_quats_by(inv_robot_tf, poses)
            # query for closest poses
            distances, indices = self._nn.query(poses)
            indices = indices.reshape((indices.shape[0]))
            distances = distances.reshape((distances.shape[0]))
            nearest_poses = transform_pos_quats_by(robot_tf, self._reachable_poses[indices])
            configs = self._configurations[indices]
            rospy.logdebug('Reachability query took %fs' % (time.time() - start_time))
            return distances, nearest_poses, configs

    def get_dispersion(self):
        """
            Return the dispersion of this reachability map.
            The dispersion is the maximum distance any reachable pose can be away
            from a sample stored in this map.
            -------
            Returns
            -------
            dist, float - upper bound on distance
        """
        return self._eucl_dispersion + self._weight * self._so3_dispersion  # TODO is this correct?

    def _create_nn(self, poses):
        """
            Create ball tree from the given list of poses.
            Sets self._reachable_poses and self._nn.
        """
        self._reachable_poses = np.array(poses)
        if self._nn_type == 'cball':
            distance_metric = skn.DistanceMetric.get_metric('pyfunc', func=pose_distance.get_distance_fn())
            self._nn = skn.BallTree(self._reachable_poses, metric=distance_metric)
        elif self._nn_type == 'pyball':
            distance_metric = skn.DistanceMetric.get_metric('pyfunc', func=yumi_distance_fn)
            self._nn = skn.BallTree(self._reachable_poses, metric=distance_metric)
        else:
            self._nn = gnat_reachability.YumiReachabilityMap(self._reachable_poses)
