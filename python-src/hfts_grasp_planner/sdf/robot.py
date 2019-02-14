"""
    This module provides SDF-based information for an OpenRAVE robot. In particular,
    this module provides an SDF-based obstacle cost function.
"""
import yaml
import rospy
import numpy as np
import hfts_grasp_planner.utils as utils
from itertools import izip
from hfts_grasp_planner.sdf.core import SceneSDF
from hfts_grasp_planner.sdf.kinbody import OccupancyOctree, RigidBodyOccupancyGrid


class RobotOccupancyGrid(object):
    """
        This class provides a series of occupancy grids representing a robot.
        Each link is represented by an RigidBodyOccupancyGrid. For computing the intersection
        of the robot with an obstacle, the intersections of all links together is summed up.
        This is class is somehow similar to RobotOccupancyOctree but uses RigidBodyOccupancyGrid
        as underlying data structure. Accordingly, it offers a slightly different interface.
    """

    def __init__(self, cell_size, robot, link_names=None, occupancy_grids=None):
        """
            Construct new RobotOccupancyGrid.
            ---------
            Arguments
            ---------
            cell_size, float - minimum edge length of a cell (all cells are cubes)
            robot, OpenRAVE robot - the robot
            link_names, list of strings - if provided, limits the intersection cost to
                the specified links
            occupancy_grids (internal use), list of RigidBodyOccupancyGrid -
                NOTE: for internal use, used to load RigidBodyOccupancyGrid from file rather than creating them
        """
        self._robot = robot
        self._occupancy_grids = []
        self._total_volume = 0.0
        self._total_num_occupied_cells = 0
        all_link_names = [link.GetName() for link in self._robot.GetLinks()]
        if link_names:
            link_names = [name for name in all_link_names if name in link_names]
        else:
            link_names = all_link_names
        links = [robot.GetLink(name) for name in link_names]
        if occupancy_grids is None:
            grid_list = RigidBodyOccupancyGrid.create_occupancy_grids(cell_size, links)
        else:
            grid_list = occupancy_grids
        self._occupancy_grids = {grid.get_link().GetName(): grid for grid in grid_list}
        self._total_volume = sum([grid.get_volume() for grid in self._occupancy_grids.itervalues()])
        self._total_num_occupied_cells = sum([grid.get_num_occupied_cells()
                                              for grid in self._occupancy_grids.itervalues()])

    @staticmethod
    def load(base_file_name, robot, link_names=None):
        """
            Load a RobotOccupancyGrid from disk. There is no sanity check performed on whether
            any of the loaded RigidBodyOccupancyGrids really represent the given robot's links.
            ---------
            Arguments
            ---------
            base_file_name, string - base file name
            robot, OpenRAVE robot to load occupancy octree for
            link_name (optional), list of string - list of links to create the occupancy octree for.
                If not provided, all links are loaded.
            ------
            Throws
            ------
            IOError if files do not exist
        """
        if link_names is not None:
            links = [robot.GetLink(link_name) for link_name in link_names]
        else:
            links = robot.GetLinks()
            link_names = [link.GetName() for link in links]
        grids = [RigidBodyOccupancyGrid.load(base_file_name + '.' + link_name, link)
                 for link_name, link in zip(link_names, links)]
        return RobotOccupancyGrid(None, robot, link_names, grids)

    def save(self, base_file_name):
        """
            Save this RobotOccupancyGrid to file.
            ---------
            Arguments
            ---------
            base_file_name, string - base file name. Use this string again to load it again.
        """
        for link_name, grid in self._occupancy_grids.iteritems():
            grid.save(base_file_name + '.' + link_name)

    def compute_intersection(self, robot_pose, robot_config, scene_sdf, links=None):
        """
            Computes the intersection between the occupancy grids of the robot's links
            and the geometry in the scene described by the provided scene sdf.
            ---------
            Arguments
            ---------
            robot_pose, numpy array of shape (4, 4) - pose of the robot
            robot_config, numpy array of shape (q,) - configuration of active DOFs
            scene_sdf, signed distance field
            links (optional), list of OpenRAVE links - if provided, limits intersection computation
                to the given links
            -------
            Returns
            -------
            v, rv -
                v is the total volume that is intersecting
                rv is this volume relative to the robot's total volume, i.e. in range [0, 1]
        """
        with self._robot:
            if links is None:
                grids = self._occupancy_grids.values()
            else:
                grids = [self._occupancy_grids[link.GetName()]
                         for link in links if link.GetName() in self._occupancy_grids]
            self._robot.SetTransform(robot_pose)
            self._robot.SetActiveDOFValues(robot_config)
            v = 0.0
            for grid in grids:
                _, _, lv, = grid.compute_intersection(scene_sdf)
                v += lv
            return v, v / self._total_volume

    def compute_penetration_cost(self, scene_sdf, robot_config, b_compute_gradient=False, robot_pose=None, links=None):
        """
            Compute CHOMP's penetration cost for the robot.
            ---------
            Arguments
            ---------
            scene_sdf, SceneSDF - scene sdf to retrieve distances from
            robot_config, np array of shape (q,) - configuration of active DOFs
            b_compute_gradients(optional), bool - if True, also compute gradient of the cost w.r.t. active DOFs
            robot_pose(optional), np array of shape (4, 4) - robot pose (default current)
            links (optional), list of OpenRAVE links - if provided, limits computation to the given links
            -------
            Returns
            -------
            penetration_cost, float - penetration cost of the robot (in range [0, inf))
            gradient(optional), np array of shape (q,) - If requested, gradient w.r.t active DOFs
        """
        with self._robot:
            if robot_pose is not None:
                self._robot.SetTransform(robot_pose)
            self._robot.SetActiveDOFValues(robot_config)
            if links is None:
                grids = self._occupancy_grids.values()
            else:
                grids = [self._occupancy_grids[link.GetName()]
                         for link in links if link.GetName() in self._occupancy_grids]
            if b_compute_gradient:
                total_value = 0.0
                gradient = np.zeros(self._robot.GetActiveDOF())
                for grid in grids:
                    link_id = grid.get_link().GetIndex()
                    values, cart_gradients, lposs = grid.compute_obstacle_cost(
                        scene_sdf, bgradients=b_compute_gradient)
                    total_value += np.sum(values)
                    for cart_grad, lpos in izip(cart_gradients, lposs):
                        jacobian = self._robot.CalculateActiveJacobian(link_id, lpos)
                        gradient += np.matmul(cart_grad, jacobian)  # it's jacobian.T * cart_grad
                return total_value, gradient
            else:
                total_value = 0.0
                for grid in grids:
                    link_id = grid.get_link().GetIndex()
                    values = grid.compute_obstacle_cost(scene_sdf, bgradients=b_compute_gradient)
                    total_value += np.sum(values)
                return total_value

    def get_robot(self):
        """
            Return the robot this occupancy tree is created for.
        """
        return self._robot


class RobotOccupancyOctree(object):
    """
        This class provides an occupancy octree representing a robot.
        Each link is represented by a OccupancyOctree. For computing the intersection
        of the robot with an obstacle, the intersections of all links together is summed up.
    """

    def __init__(self, cell_size, robot, link_names=None, occupancy_trees=None):
        """
            Construct new RobotOccupancyOctree.
            ---------
            Arguments
            ---------
            cell_size, float - minimum edge length of a cell (all cells are cubes)
            robot, OpenRAVE robot - the robot
            link_names, list of strings - if provided, limits the intersection cost to
                the specified links
            occupancy_trees (internal use), list of OccupancyOctree -
                NOTE: for internal use, used to load OccupancyOctrees from file rather than creating them
        """
        self._robot = robot
        self._occupancy_trees = {}
        self._total_volume = 0.0
        self._total_num_occupied_cells = 0
        self._link_names = [link.GetName() for link in self._robot.GetLinks()]
        if link_names:
            self._link_names = [name for name in self._link_names if name in link_names]
        links = [robot.GetLink(name) for name in self._link_names]
        if occupancy_trees is None:
            trees = OccupancyOctree.create_occupancy_octrees(cell_size, links)
        else:
            trees = occupancy_trees
        self._occupancy_trees = dict(zip(self._link_names, trees))
        self._total_volume = sum([tree.get_volume() for tree in self._occupancy_trees.values()])
        self._total_num_occupied_cells = sum([tree.get_num_occupied_cells() for tree in self._occupancy_trees.values()])

    @staticmethod
    def load(base_file_name, robot, link_names=None):
        """
            Load a robot occupancy octree from disk. There is no sanity check performed on whether
            any of the loaded OccuancyOctree really represent the given robot's links.
            ---------
            Arguments
            ---------
            base_file_name, string - base file name
            robot, OpenRAVE robot to load occupancy octree for
            link_name (optional), list of string - list of links to create the occupancy octree for.
                If not provided, all links are loaded.
            ------
            Throws
            ------
            IOError if files do not exist
        """
        if link_names is not None:
            links = [robot.GetLink(link_name) for link_name in link_names]
        else:
            links = robot.GetLinks()
            link_names = [link.GetName() for link in links]
        trees = []
        for link_name, link in zip(link_names, links):
            trees.append(OccupancyOctree.load(base_file_name + '.' + link_name, link))
        return RobotOccupancyOctree(None, robot, link_names, trees)

    def save(self, base_file_name):
        """
            Save this RobotOccupancyOctree to file.
            ---------
            Arguments
            ---------
            base_file_name, string - base file name. Use this string again to load it again.
        """
        for link_name, octree in self._occupancy_trees.iteritems():
            octree.save(base_file_name + '.' + link_name)

    def compute_intersection(self, robot_pose, robot_config, scene_sdf, links=None):
        """
            Computes the intersection between the octrees of the robot's links
            and the geometry in the scene described by the provided scene sdf.
            ---------
            Arguments
            ---------
            robot_pose, numpy array of shape (4, 4) - pose of the robot
            robot_config, numpy array of shape (q,) - configuration of active DOFs
            scene_sdf, signed distance field
            links (optional), list of OpenRAVE links - if provided, limits intersection computation
                on the given links
            -------
            Returns
            -------
            (v, rv, dc, adc) -
                v is the total volume that is intersecting
                rv is this volume relative to the robot's total volume, i.e. in range [0, 1]
                dc is a cost that is computed by (approximately) summing up all signed
                    distances of intersecting cells
                adc is this cost divided by the number of intersecting cells, i.e. the average
                    signed distance of the intersecting cells
        """
        with self._robot:
            self._robot.SetTransform(robot_pose)
            self._robot.SetActiveDOFValues(robot_config)
            v, dc = 0.0, 0.0
            if links is not None:
                trees = [self._occupancy_trees[link.GetName()]
                         for link in links if link.GetName() in self._occupancy_trees]
            else:
                trees = self._occupancy_trees.values()
            for tree in trees:
                tv, _, tdc, _, _ = tree.compute_intersection(scene_sdf, bvolume_in_cells=True)
                v += tv
                dc += tdc
            return v, v / self._total_num_occupied_cells, dc, dc / self._total_num_occupied_cells

    def compute_max_penetration(self, robot_pose, robot_config, scene_sdf, b_compute_dir=False):
        """
            Computes the maximum penetration of this robot with the given sdf.
            The maximum penetration is the minimal signed distance in this robot's volume
            in the given scene_sdf.
            ---------
            Arguments
            ---------
            robot_pose, numpy array of shape (4, 4) - pose of the robot
            robot_config, numpy array of shape (q,) - configuration of active DOFs
            scene_sdf, SceneSDF
            b_compute_dir, bool - If True, also retrieve the direction from the maximally penetrating
                cell to the closest free cell.
            -------
            Returns
            -------
            penetration distance, float - minimum in scene_sdf in the volume covered by this link.
            v_to_border, numpy array of shape (3,) - translation vector to move the cell with maximum penetration
                out of collision (None if b_compute_dir is False)
            pos, numpy array of shape (3,) - world position of the cell with maximum penetration
            link_name, string - name of the link with maximal penetration
        """
        with self._robot:
            self._robot.SetTransform(robot_pose)
            self._robot.SetActiveDOFValues(robot_config)
            pdist, vdir, pos, link_name = 0.0, None, None, None
            for tree, name in self._occupancy_trees.iteritems():
                tdist, tdir, tpos = tree.compute_max_penetration(scene_sdf, b_compute_dir=b_compute_dir)
                if tdist < pdist:
                    pdist = tdist
                    vdir = tdir
                    pos = tpos
                    link_name = name
            return pdist, vdir, pos, link_name

    def get_robot(self):
        """
            Return the robot this occupancy tree is created for.
        """
        return self._robot

    def visualize(self, level, config=None):
        """
            Visualize the octrees for the given level.
            ---------
            Arguments
            ---------
            level, int - level to draw
            config, numpy array (q,) (optional) - robot configuration to draw for
        """
        if config:
            self._robot.SetActiveDOFValues(config)
        handles = []
        for tree in self._occupancy_trees.values():
            handles.extend(tree.visualize(level))
        return handles


class GrabbingRobotOccupancyTree(RobotOccupancyOctree):
    """
        Like a RobotOccupancyOctree but it also considers the object that is
        currently grasped by the robot.
    """

    def __init__(self, cell_size, robot, link_names=None):
        """
            Create a new GrabbingRobotOccupancyTree.
            See RobotOccupancyTree for information on arguments.
            NOTE: Before using this object, you need to call update_object().
        """
        super(GrabbingRobotOccupancyTree, self).__init__(cell_size, robot, link_names)
        self._object = None
        self._object_octree = None
        self._octree_cache = {}

    def update_object(self):
        """
            Update the object octree.
            If the grabbed object is still the same as last time
            this function was called, this is a no-op. Otherwise, it will
            update the internal object octree. It is assumed that the robot
            only grasps ONE object at a time. If no object is grasped, this class
            continues to function in the exact same way as an RobotOccupancyOctree.
        """
        # whatever happened, cache the old octree
        if self._object is not None:
            self._octree_cache[self._object.GetName()] = self._object_octree
        # check what the currently grabbed object is
        grabbed_objects = self._robot.GetGrabbed()
        if len(grabbed_objects) > 0:
            if len(grabbed_objects) > 1:
                rospy.logwarn("The robot is grasping more than one object. Working with first one.")
            self._object = grabbed_objects[0]
            if self._object.GetName() in self._octree_cache:
                self._object_octree = self._octree_cache[self._object.GetName()]
            else:
                link = self._object.GetLinks()[0]
                self._object_octree = OccupancyOctree(self._cell_size, link)
        else:
            # no grasped object
            self._object = None
            self._object_octree = None

    def compute_intersection(self, robot_pose, robot_config, scene_sdf):
        """
            See RobotOccupancyTree for documentation.
            Also considers the grasped object.
        """
        v, rv, dc, adc = super(GrabbingRobotOccupancyTree, self).compute_intersection(
            robot_pose, robot_config, scene_sdf)
        if self._object_octree is not None:
            ov, orv, odc, oadc, _ = self._object_octree.compute_intersection(scene_sdf)
            return v + ov, rv + orv, dc + odc, adc + oadc  # TODO figure out what to doe with the relative values
        return v, rv, dc, adc

    def compute_max_penetration(self, robot_pose, robot_config, scene_sdf, b_compute_dir=False):
        """
            See RobotOccupancyTree for documentation.
            Also considers the grasped object.
        """
        pdist, vdir, ppos, link_name = super(GrabbingRobotOccupancyTree, self).compute_max_penetration(
            robot_pose, robot_config, scene_sdf, b_compute_dir)
        if self._object_octree is not None:
            opdist, ovdir, opos = self._object_octree.compute_max_penetration(scene_sdf, b_compute_dir=b_compute_dir)
            if opdist < pdist and b_compute_dir:
                vdir = ovdir
                pdist = opdist
                ppos = opos
                link_name = self._object.GetName()
        return pdist, vdir, ppos, link_name


class RobotBallApproximation(object):
    """
        This class provides signed distance information for a robot.
        It utilizes a set of balls to approximate a robot and provides access functions
        to acquire the shortest distance of each ball to the closest obstacle.
        In order to instantiate an object of this class, you need a signed distance field
        for the OpenRAVE scene the robot is embedded in as well as a description file
        containing the definition of the approximating balls.
    """

    def __init__(self, robot, desc_file):
        """
            Creates a new RobotSDF.
            ---------
            Arguments
            ---------
            robot - OpenRAVE robot this ball approximation approximates
            desc_file - robot ball decription file
        """
        self._robot = robot
        self._handles = []  # list of openrave handles for visualization
        self._balls = {}  # dictionary that maps each link name to an array balls
        self._load_approximation(desc_file)
        self._links = [self._robot.GetLink(lname) for lname in self._balls.keys()]

    def _load_approximation(self, filename):
        """
            Loads a ball approximation for the robot from the given file.
        """
        with open(filename, 'r') as in_file:
            link_descs = yaml.load(in_file)
        for lname, ball_list in link_descs.iteritems():
            self._balls[lname] = np.array(ball_list)

        # self._ball_positions = []
        # self._ball_radii = []
        # self._link_indices = []
        # link_descs = None
        # with open(filename, 'r') as in_file:
        #     link_descs = yaml.load(in_file)
        # # first we need to know how many balls we have
        # num_balls = 0
        # for ball_descs in link_descs.itervalues():
        #     num_balls += len(ball_descs)
        # # now we can create our data structures
        # self._ball_positions = np.ones((num_balls, 4))
        # self._query_positions = np.ones((num_balls, 3))
        # self._ball_radii = np.zeros(num_balls)
        # self._ball_indices = []
        # index_offset = 0
        # # run over all links
        # links = self._robot.GetLinks()
        # for link_idx in range(len(links)):  # we need the index
        #     link_name = links[link_idx].GetName()
        #     if link_name in link_descs:  # if we have some balls for this link
        #         ball_descs = link_descs[link_name]
        #         num_balls_link = len(ball_descs)
        #         index_offset_link = index_offset
        #         # save this link
        #         self._link_indices.append(link_idx)
        #         # save the offset and number of balls
        #         self._ball_indices.append((num_balls_link, index_offset_link))
        #         # save all balls
        #         for ball_idx in range(num_balls_link):
        #             self._ball_positions[index_offset_link + ball_idx, :3] = np.array(ball_descs[ball_idx][:3])
        #             self._ball_radii[index_offset_link + ball_idx] = ball_descs[ball_idx][3]
        #         index_offset += num_balls_link
        #     else:
        #         # for links that don't have any balls we need to store None so we can index ball_indices easily
        #         self._ball_indices.append(None)

    # def get_distances(self):
    #     """
    #         Returns the distances of all balls to the respective closest obstacle.
    #     """
    #     if self._ball_positions is None or self._sdf is None:
    #         return None
    #     link_tfs = self._robot.GetLinkTransformations()
    #     for link_idx in self._link_indices:
    #         nb, off = self._ball_indices[link_idx]  # number of balls, offset
    #         ltf = link_tfs[link_idx]
    #         self._query_positions[off:off +
    #                               nb] = np.dot(self._ball_positions[off: off + nb, : 3], ltf[:3, :3].transpose()) + ltf[:3, 3]
    #     return self._sdf.get_distances(self._query_positions) - self._ball_radii

    # def get_distances2(self, links=None, breturn_gradients=False):
    #     """
    #         Returns the distances of all balls to the respective closest obstacle.
    #         NOTE: Throws a ValueError if not intialized.
    #         ---------
    #         Arguments
    #         ---------
    #         links (optional), list of OpenRAVE Links - if provided, limit distance retrieval to the specified
    #             links
    #         breturn_gradients (optional), bool - if True, also return gradients at query points
    #         --------
    #         Additional requirements:
    #             Assumes that the OpenRAVE robot passed on construction is set to the desired configuration before.
    #         -------
    #         Returns
    #         -------
    #         distances, float np.array of shape (n,) - n number of balls that approximate the given links
    #         gradients, np array of shape (n, 3) - distance gradients at the ball centers
    #     """
    #     if self._ball_positions is None:
    #         raise ValueError("Ball approximation of robot not loaded. Can not compute distances.")
    #     if self._sdf is None:
    #         raise ValueError("No sdf set. Can not retrieve distances.")
    #     link_tfs = np.array(self._robot.GetLinkTransformations())
    #     # limit computation to given links
    #     if links is not None:
    #         link_idx = np.array([link.GetIndex() for link in links], dtype=int)
    #         link_tfs = link_tfs[link_idx]
    #     else:
    #         link_idx = self._link_indices
    #     # retrieve distances
    #     nbs, offsets = self._ball_indices[link_idx]
    #     total_num_balls = np.sum(nbs)
    #     query_positions = np.empty((total_num_balls, 3))
    #     ball_radii = np.empty(total_num_balls)
    #     query_off = 0
    #     for lid, ltf in link_idx, link_tfs:
    #         local_positions = self._ball_positions[offsets[lid]:offsets[lid] + nbs[lid], : 3]
    #         query_positions[query_off:query_off + nbs[lid]] = np.dot(local_positions, ltf[:3, :3].T) + ltf[:3, 3]
    #         ball_radii[query_off: query_off + nbs[lid]] = self._ball_radii[offsets[lid]:offsets[lid] + nbs[lid]]
    #         query_off += nbs[lid]
    #     if breturn_gradients:
    #         distances, grad = self._sdf.get_distances_grad(query_positions)
    #         distances -= ball_radii
    #         return distances, grad
    #     return self._sdf.get_distances(local_positions) - ball_radii

    def compute_penetration_cost(self, scene_sdf, links=None, eps=0.005):
        """
            Compute CHOMP's penetration cost for the robot.
            Assumes active DOFs are set.
            ---------
            Arguments
            ---------
            scene_sdf, SceneSDF - signed distance field of the scene to retrieve distances from
            links (optional), list of OpenRAVE Links - if provided, limit distance retrieval to the specified
                links
            eps, float - tolerance
            -------
            Returns
            -------
            penetration cost, float
            gradient, np.array of shape (q,) - gradient of penetration cost
        """
        if links is None:
            ball_links = self._links
        else:
            ball_links = [link for link in links if link.GetName() in self._balls]
        total_value = 0.0
        gradient = np.zeros(self._robot.GetActiveDOF())
        num_colliding_balls = 0
        # first retrieve all query positions
        query_positions = []
        ball_infos = []
        radii = []
        for link in ball_links:
            ball_info = self._balls[link.GetName()]
            ltf = link.GetTransform()
            local_query_positions = ball_info[:, :3]
            offset = len(query_positions)
            query_positions.extend(np.matmul(local_query_positions, ltf[:3, :3].T) + ltf[:3, 3])
            ball_infos.append((offset, ball_info))
            radii.extend(ball_info[:, 3])
        # query distances and gradients for all balls at the same time to make use of parallelism
        query_positions = np.array(query_positions)
        distances, cart_grads = scene_sdf.get_distances_grad(query_positions)
        distances -= radii  # substract radii
        smooth_dists, smooth_gradients = utils.chomps_distance(distances, eps, cart_grads)
        # gradient_handles = []
        # compute gradients per link
        for link, (offset, ball_info) in izip(ball_links, ball_infos):
            # select subsets of positions and distances relevant to this link
            link_positions = query_positions[offset: offset + ball_info.shape[0]]
            link_distances = smooth_dists[offset: offset + ball_info.shape[0]]
            link_gradients = smooth_gradients[offset: offset + ball_info.shape[0]]
            # find non-zero gradients
            non_zero_idx = np.unique(np.nonzero(link_gradients)[0])
            total_value += np.sum(link_distances[non_zero_idx])
            num_colliding_balls += non_zero_idx.shape[0]
            for sgradient, pos in izip(link_gradients[non_zero_idx], link_positions[non_zero_idx]):
                jacobian = self._robot.CalculateActiveJacobian(link.GetIndex(), pos)
                inv_jac, rank = utils.compute_pseudo_inverse_rank(jacobian)
                if rank < 3:
                    num_colliding_balls -= 1
                    continue
                gradient += np.matmul(inv_jac, sgradient)  # it's jacobian.T * cart_grad
                # gradient += np.matmul(sgradient, jacobian)  # it's jacobian.T * cart_grad
                # ppos = ltf[:3, 3] + np.dot(ltf[:3, :3], lpos)
                # gradient_handles.append(self._robot.GetEnv().drawarrow(pos, pos + 0.1 * sgradient, 0.005))
        if num_colliding_balls > 0:
            total_value /= num_colliding_balls
            gradient /= num_colliding_balls
            # self.visualize_balls()
        else:
            self.hide_balls()
        return total_value, gradient

    def visualize_balls(self):
        """
            If approximating balls are available, this function issues rendering these
            in OpenRAVE.
        """
        self.hide_balls()
        env = self._robot.GetEnv()
        color = np.array((1, 0, 0, 0.6))
        for link in self._links:
            ball_info = self._balls[link.GetName()]
            ltf = link.GetTransform()
            positions = np.dot(ball_info[:, :3], ltf[:3, :3].T) + ltf[:3, 3]
            for b in range(ball_info.shape[0]):
                handle = env.plot3(positions[b], ball_info[b, 3], color, 1)
                self._handles.append(handle)

    def hide_balls(self):
        """
            If the approximating balls are currently rendered, this function
            removes those renderings, else it does nothing.
        """
        self._handles = []
