"""
    This module provides SDF-based information for an OpenRAVE robot. In particular,
    this module provides an SDF-based obstacle cost function.
"""
import yaml
import rospy
import numpy as np
from itertools import izip
from hfts_grasp_planner.sdf.core import SceneSDF
from hfts_grasp_planner.sdf.kinbody import OccupancyOctree


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
        self._occupancy_trees = []
        self._total_volume = 0.0
        self._total_num_occupied_cells = 0
        self._link_names = [link.GetName() for link in self._robot.GetLinks()]
        if link_names:
            self._link_names = [name for name in self._link_names if name in link_names]
        links = [robot.GetLink(name) for name in self._link_names]
        if occupancy_trees is None:
            self._occupancy_trees = OccupancyOctree.create_occupancy_octrees(cell_size, links)
        else:
            self._occupancy_trees = occupancy_trees
        self._total_volume = sum([tree.get_volume() for tree in self._occupancy_trees])
        self._total_num_occupied_cells = sum([tree.get_num_occupied_cells() for tree in self._occupancy_trees])

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
        for link_name, octree in zip(self._link_names, self._occupancy_trees):
            octree.save(base_file_name + '.' + link_name)

    def compute_intersection(self, robot_pose, robot_config, scene_sdf):
        """
            Computes the intersection between the octrees of the robot's links
            and the geometry in the scene described by the provided scene sdf.
            ---------
            Arguments
            ---------
            robot_pose, numpy array of shape (4, 4) - pose of the robot
            robot_config, numpy array of shape (q,) - configuration of active DOFs
            scene_sdf, signed distance field
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
            for tree in self._occupancy_trees:
                tv, _, tdc, _, _ = tree.compute_intersection(scene_sdf)
                v += tv
                dc += tdc
            return v, v / self._total_volume, dc, dc / self._total_num_occupied_cells

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
            for tree, name in izip(self._occupancy_trees, self._link_names):
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
        for tree in self._occupancy_trees:
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


class RobotSDF(object):
    """
        This class provides signed distance information for a robot.
        It utilizes a set of balls to approximate a robot and provides access functions
        to acquire the shortest distance of each ball to the closest obstacle.
        In order to instantiate an object of this class, you need a signed distance field
        for the OpenRAVE scene the robot is embedded in as well as a description file
        containing the definition of the approximating balls.
    """

    def __init__(self, robot, scene_sdf=None):
        """
            Creates a new RobotSDF.
            NOTE: Before you can use this object, you need to provide it with a ball approximation.
            @param robot - OpenRAVE robot this sdf should operate on
            @param scene_sdf (optional) - A SceneSDF for the given environment. If not provided,
                            at construction, it must be provided by calling set_sdf(..) before this
                            class can be used.
        """
        self._robot = robot
        if scene_sdf is not None and not isinstance(scene_sdf, SceneSDF):
            raise TypeError("The provided sdf object must be a SceneSDF")
        self._sdf = scene_sdf
        self._handles = []  # list of openrave handles for visualization
        # ball_positoins stores one large matrix of shape (n, 4), where n is the number of balls we have
        # in each row we have the homogeneous coordinates of on ball center w.r.t its link's frame
        self._ball_positions = None
        self._query_positions = None
        # ball_radii stores the radii for all balls
        self._ball_radii = None
        # saves the indices of links for which we have balls
        self._link_indices = None
        # saves tuples (num_balls, index_offset) for each link_idx in self._link_indices
        # index_offset refers to the row in self._ball_positions and ball_radii in which the first ball
        # for link i is stored. num_balls is the number of balls for this link
        self._ball_indices = None

    def load_approximation(self, filename):
        """
            Loads a ball approximation for the robot from the given file.
        """
        self._ball_positions = []
        self._ball_radii = []
        self._link_indices = []
        link_descs = None
        with open(filename, 'r') as in_file:
            link_descs = yaml.load(in_file)
        # first we need to know how many balls we have
        num_balls = 0
        for ball_descs in link_descs.itervalues():
            num_balls += len(ball_descs)
        # now we can create our data structures
        self._ball_positions = np.ones((num_balls, 4))
        self._query_positions = np.ones((num_balls, 3))
        self._ball_radii = np.zeros(num_balls)
        self._ball_indices = []
        index_offset = 0
        # run over all links
        links = self._robot.GetLinks()
        for link_idx in range(len(links)):  # we need the index
            link_name = links[link_idx].GetName()
            if link_name in link_descs:  # if we have some balls for this link
                ball_descs = link_descs[link_name]
                num_balls_link = len(ball_descs)
                index_offset_link = index_offset
                # save this link
                self._link_indices.append(link_idx)
                # save the offset and number of balls
                self._ball_indices.append((num_balls_link, index_offset_link))
                # save all balls
                for ball_idx in range(num_balls_link):
                    self._ball_positions[index_offset_link + ball_idx, :3] = np.array(ball_descs[ball_idx][:3])
                    self._ball_radii[index_offset_link + ball_idx] = ball_descs[ball_idx][3]
                index_offset += num_balls_link
            else:
                # for links that don't have any balls we need to store None so we can index ball_indices easily
                self._ball_indices.append(None)

    def set_sdf(self, sdf):
        """
            Sets the scene sdf to use.
            - :sdf: must be of type SceneSDF
        """
        if not isinstance(sdf, SceneSDF):
            raise TypeError("The provided sdf object must be a SceneSDF")
        self._sdf = sdf

    def get_distances(self):
        """
            Returns the distances of all balls to the respective closest obstacle.
        """
        if self._ball_positions is None or self._sdf is None:
            return None
        link_tfs = self._robot.GetLinkTransformations()
        for link_idx in self._link_indices:
            nb, off = self._ball_indices[link_idx]  # number of balls, offset
            self._query_positions[off:off +
                                  nb] = np.dot(self._ball_positions[off: off + nb, : 3], link_tfs[link_idx].transpose())
        return self._sdf.get_distances(self._query_positions) - self._ball_radii

    def visualize_balls(self):
        """
            If approximating balls are available, this function issues rendering these
            in OpenRAVE.
        """
        self.hide_balls()
        if self._ball_positions is None:
            return
        env = self._robot.GetEnv()
        link_tfs = self._robot.GetLinkTransformations()
        color = np.array((1, 0, 0, 0.6))
        for link_idx in self._link_indices:
            (nb, off) = self._ball_indices[link_idx]  # number of balls, offest
            positions = np.dot(self._ball_positions[off: off + nb], link_tfs[link_idx].transpose())
            for ball_idx in range(nb):
                handle = env.plot3(positions[ball_idx, : 3], self._ball_radii[off + ball_idx],
                                   color, 1)
                self._handles.append(handle)

    def hide_balls(self):
        """
            If the approximating balls are currently rendered, this function
            removes those renderings, else it does nothing.
        """
        self._handles = []
