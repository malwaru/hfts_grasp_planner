import numpy as np
import yaml
import os
import math
import sys
from pycaster import pycaster
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
import scipy.spatial
import hfts_grasp_planner.external.transformations as transformations
import hfts_grasp_planner.dmg.dmg_path as dmg_path_module


class DexterousManipulationGraph():
    """class to read a DMG from files and do basic search"""

    def __init__(self):
        self._adjacency_list = None
        self._node_to_component = None
        self._node_to_position = None
        self._component_to_normal = None
        self._component_to_zero_axis = None
        self._component_to_nodes = None
        self._node_to_angles = None
        self._node_to_angle_intervals = None
        self._supervoxel_angle_to_angular_component = None
        self._angle_res = None
        # data structures for the nearest neighbor
        self._kdtree = None
        self._idx_to_supervoxel = None
        self._supervoxel_to_angular_component = None
        # stuff for opposite finger component
        self._caster = None
        self._mesh_scale = 1.0
        self._object_shape_file = None
        # length of the gripper's finger
        self._finger_length = 0.04  # Yumi's finger
        # utility to compute push paths
        self._push_path_computer = None

    @staticmethod
    def loadFromYaml(yaml_file):
        """
            Load a DMG from yaml description file.
            ---------
            Arguments
            ---------
            yaml_file, string - path to yaml file containing all information to load a DMG
            -------
            Returns
            -------
            dmg, DexterousManipulationGraph - fully intialized graph
        """
        dmg = DexterousManipulationGraph()
        with open(yaml_file, 'r') as the_file:
            yaml_content = yaml.load(the_file)
        basepath = os.path.dirname(yaml_file) + '/'
        dmg.set_object_shape_file(basepath + yaml_content["mesh"], yaml_content["mesh_scale"])
        dmg.read_graph(basepath + yaml_content['graph_file'])
        dmg.read_nodes(basepath + yaml_content['node_position'])
        dmg.read_node_to_component(basepath + yaml_content['node_component'])
        dmg.read_component_to_normal(basepath + yaml_content['component_to_normal'])
        dmg.read_node_to_angles(basepath + yaml_content['node_angle'], yaml_content['angle_res'])
        dmg.read_supervoxel_angle_to_angular_component(basepath + yaml_content['node_angle_angle_component'])
        return dmg

    # @staticmethod
    # def initFromPath(basepath, name, voxel_res, ang_res, mesh_scale=1.0):
    #     """
    #         Create a DMG from files in folder basefolder assuming that the files follow
    #         the following naming convention:
    #         <type>_<name>_<voxel_res>_<ang_res>.txt
    #         ---------
    #         Arguments
    #         ---------
    #         basepath, string - path to a folder containing all files
    #         name, string - common substring of all filenames
    #         voxel_res, int - voxel resolution
    #         ang_res, int - angular resolution
    #         mesh_scale, float - scale to apply on mesh to scale it to meters
    #         -------
    #         Returns
    #         -------
    #         dmg, DexterousManipulationGraph - fully initialized graph
    #     """
    #     dmg = DexterousManipulationGraph()
    #     # read the dmg from files
    #     dmg.set_object_shape_file(basepath + '/' + name + '.stl', mesh_scale)
    #     dmg.read_graph("%s/graph_%s_%i_%i.txt" % (basepath, name, voxel_res, ang_res))
    #     dmg.read_nodes("%s/node_position_%s_%i_%i.txt" % (basepath, name, voxel_res, ang_res))
    #     dmg.read_node_to_component("%s/node_component_%s_%i_%i.txt" % (basepath, name, voxel_res, ang_res))
    #     dmg.read_component_to_normal("%s/component_normal_%s_%i_%i.txt" % (basepath, name, voxel_res, ang_res))
    #     dmg.read_node_to_angles("%s/node_angle_%s_%i_%i.txt" % (basepath, name, voxel_res, ang_res), ang_res)
    #     dmg.read_supervoxel_angle_to_angular_component(
    #         "%s/node_angle_angle_component_%s_%i_%i.txt" % (basepath, name, voxel_res, ang_res))
    #     return dmg

    def set_object_shape_file(self, filename, scale=1.0):
        '''
            Set the object shape file (stl mesh)
            ---------
            Arguments
            ---------
            filename, string - path to stl file
            scale, float - scale to transform from meters to the scale of the mesh's frame
        '''
        self._object_shape_file = filename
        self._mesh_scale = scale
        # the scale argument does not work
        self._caster = pycaster.rayCaster.fromSTL(filename, scale=1.0)
        self._push_path_computer = dmg_path_module.DMGPushPath(filename)

    def read_nodes(self, filename):
        '''reads the Cartesian positions of all the nodes'''
        nodes_to_position = dict()
        positions_list = list()
        f = open(filename, 'r')
        idx = 0
        idx_to_sv = dict()
        for x in f:
            y = x.split()
            p = np.array([float(y[2]), float(y[3]), float(y[4])])
            n = (int(y[0]), int(y[1]))
            nodes_to_position[n] = p
            positions_list.append(p)
            idx_to_sv[idx] = n[0]
            idx += 1

        self._node_to_position = nodes_to_position
        self._idx_to_supervoxel = idx_to_sv
        self._kdtree = scipy.spatial.KDTree(positions_list)

    def read_graph(self, filename):
        '''reads the adjacency list'''
        nodes_to_list_of_nodes = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            node = (int(y[0]), int(y[1]))
            nodes_to_list_of_nodes[node] = []
            for i in range(2, len(y), 2):
                nodes_to_list_of_nodes[node] += [(int(y[i]), int(y[i+1]))]
        self._adjacency_list = nodes_to_list_of_nodes

    def read_node_to_component(self, filename):
        '''reads the mapping from node id to connected component'''
        nodes_to_component = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            node = (int(y[0]), int(y[1]))
            nodes_to_component[node] = int(y[2])
        self._node_to_component = nodes_to_component

    def read_component_to_normal(self, filename):
        '''reads the normal associated to each component'''
        component_to_normal = dict()
        component_to_zero_axis = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            component = int(y[0])
            normal = np.array([float(y[1]), float(y[2]), float(y[3])])
            axis = np.array([float(y[4]), float(y[5]), float(y[6])])
            component_to_normal[component] = normal
            component_to_zero_axis[component] = axis
        self._component_to_normal = component_to_normal
        self._component_to_zero_axis = component_to_zero_axis

    def read_node_to_angles(self, filename, angle_res):
        '''reads the admissible angles in one node. These nodes are in degrees!'''
        self._angle_res = angle_res
        nodes_to_angles = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            node = (int(y[0]), int(y[1]))
            angles_list = list()
            for i in range(2, len(y)):
                angles_list += [int(y[i])]
            nodes_to_angles[node] = angles_list
        self._node_to_angles = nodes_to_angles
        # now compute angular intervals
        self._node_to_angle_intervals = {}
        for node, angles in self._node_to_angles.iteritems():
            if len(angles) == 1:
                self._node_to_angle_intervals[node] = (angles[0], angles[0])
            else:
                assert(len(angles) > 1)
                # search what the true min and max angles are
                # the min angle might be negative, if we wrap around 360
                min_angle = angles[0]
                max_angle = angles[-1]
                for i in range(1, len(angles)):
                    if angles[i] - angles[i - 1] > self._angle_res:
                        # there is a gap in angles here
                        max_angle = angles[i - 1]
                        min_angle = angles[i] - 360
                        break
                self._node_to_angle_intervals[node] = (min_angle, max_angle)

    def read_supervoxel_angle_to_angular_component(self, filename):
        '''reads the mapping from supervoxel id and angle to the node angular component'''
        node_angle_to_angle_component = dict()
        sv_to_angle_component = dict()
        f = open(filename, 'r')
        for x in f:
            y = x.split()
            supervoxel_id = int(y[0])
            if not node_angle_to_angle_component.has_key(supervoxel_id):
                node_angle_to_angle_component[supervoxel_id] = dict()
                sv_to_angle_component[supervoxel_id] = set()
            angle = int(y[1])
            node_angle_to_angle_component[supervoxel_id][angle] = int(y[2])
            sv_to_angle_component[supervoxel_id].add(int(y[2]))
        self._supervoxel_angle_to_angular_component = node_angle_to_angle_component
        self._supervoxel_to_angular_component = sv_to_angle_component

    def snap_to_angle_grid(self, angle):
        """
            Snap the given floating point angle to the closest angle on the grid.
            ---------
            Arguments
            ---------
            angle, float - floating point angle in range [0, 360)
            -------
            Returns
            -------
            angle, int - closest angle grid point to the input angle
        """
        return int(np.around(angle / self._angle_res) * self._angle_res)

    def get_closest_nodes(self, point):
        '''returns the nodes whose center is the closest to the given point.
           They can be two if there are nodes with different angular component'''
        (_, idx) = self._kdtree.query(point)
        supervoxel = self._idx_to_supervoxel[idx]
        # this supervoxel can correspond to more than one node
        angular_components = self._supervoxel_to_angular_component[supervoxel]
        nodes = list()
        for c in angular_components:
            nodes.append((supervoxel, c))
        return nodes

    def get_position(self, node):
        """
            Return the position of the given node.
        """
        return self._node_to_position[node]

    def get_angular_resolution(self):
        """
            Return angular resolution in degrees.
        """
        return self._angle_res

    def ray_shape_intersections(self, start_point, goal_point):
        '''get the intersection of a ray with the object's shape (used for checks on the opposite finger)'''
        source = self._mesh_scale * start_point
        destination = self._mesh_scale * goal_point
        intersections = self._caster.castRay(source, destination)
        intersection_points = list()
        for x in intersections:
            intersection_points.append(np.array(x) / self._mesh_scale)
        return intersection_points

    def is_valid_angle(self, node, angle, breturn_err=False):
        """
            Return whether the given angle is valid for the given node.
            ---------
            Arguments
            ---------
            node, tuple (int, int)
            angle, float or int - in degrees in range [0, 360)
            breturn_err, bool - if True, also return distance to interval
            -------
            Returns
            -------
            bvalid, bool
            error, float - optionally the minimal angular distance to the range of valid angles
        """
        error = 0.0
        min_angle, max_angle = self._node_to_angle_intervals[node]
        if min_angle < 0.0 and angle > max_angle:
            langle = angle - 360
            bvalid = langle >= min_angle
            if not bvalid and breturn_err:
                error = min(abs(min_angle - langle), abs(angle - max_angle))
        else:
            bvalid = angle >= min_angle and angle <= max_angle
            if not bvalid and breturn_err:
                if angle < min_angle:
                    error = min(min_angle - angle, angle + 360 - max_angle)
                else:
                    error = min(angle - max_angle, min_angle - angle + 360)
        if breturn_err:
            return bvalid, error
        return bvalid

    def get_finger_tf(self, node, angle):
        """
            Return the pose of the fingertip in object frame at the given node and angle.
            ---------
            Arguments
            ---------
            node, tuple (int, int) identifying a DMG node
            angle, int - angle in degrees
            -------
            Returns
            -------
            oTf, np array of shape (4, 4)
        """
        # get position of the node
        point = self._node_to_position[node]
        # the zero axis is opposite to the finger axis in the convention
        normal = self._component_to_normal[self._node_to_component[node]]
        zero_axis = self._component_to_zero_axis[self._node_to_component[node]]
        # tf of the node frame at node position
        oTn = np.eye(4)
        oTn[:3, :3] = np.c_[normal, -zero_axis, np.cross(normal, -zero_axis)]
        oTn[:3, 3] = point
        # rotate around normal (x axis in local frame)
        theta = angle * np.pi / 180.0
        nTa = transformations.rotation_matrix(theta, np.array([1.0, 0.0, 0.0]))
        return np.dot(oTn, nTa)

    def get_closest_node_angle(self, finger_tf):
        """
            Return the closest node and angle for a fingertip located at pose finger_tf.
            ---------
            Arguments
            ---------
            finger_tf, np.array of shape (4, 4) - tf matrix of finger frame in object frame oTf
            -------
            Returns
            -------
            node, tuple - node id
            angle, int - closest angle
            bvalid, bool - flag whether the angle is within the range of valid angles for the returned node
        """
        # first get closest node
        pos = finger_tf[:3, 3]
        neighbors = self.get_closest_nodes(pos)
        nearest_n = None
        nearest_angle = None
        angle_err = np.inf
        for n in neighbors:
            comp = self._node_to_component[n]
            normal = self._component_to_normal[comp]
            zero_axis = self._component_to_zero_axis[comp]
            cTo = np.array([normal, -zero_axis, np.cross(normal, -zero_axis)])
            cTf = np.dot(cTo, finger_tf[:3, :3])
            angle, _, _ = transformations.euler_from_matrix(cTf)
            angle = angle / np.pi * 180.0
            if angle < 0.0:
                angle += 360
            # angle, direct, point = transformations.rotation_from_matrix(cTf)
            # assert(np.isclose(point, np.zeros(3), atol=1e-3))
            # assert(np.isclose(direct, np.array([1.0, 0.0, 0.0]), atol=1e-3))
            bvalid, err = self.is_valid_angle(n, angle, True)
            if bvalid:
                return n, self.snap_to_angle_grid(angle), True
            if angle_err > err:
                nearest_n = n
                nearest_angle = angle
                angle_err = err
        return nearest_n, nearest_angle, False

    def get_finger_distance(self, node_a, node_b):
        """
            Return the distance between fingertips if finger a is placed at node_a and finger b at node_b
            ---------
            Arguments
            ---------
            node_a, tuple (int, int)
            node_b, tuple (int, int)
            -------
            Returns
            -------
            dist, float
        """
        pos_a = self._node_to_position[node_a]
        pos_b = self._node_to_position[node_b]
        return np.linalg.norm(pos_a - pos_b)

    def get_finger_openings(self, node, angle):
        """
            Return all possible distances between two fingertips on opposing sides of
            the object, if one fingertip is placed at the given node at the given angle.
            It is assumed that both fingertips need to be parallel, i.e. have the same angle.
            ---------
            Arguments
            ---------
            node, tuple (int, int) identifying a DMG node
            angle, int - angle in degrees
            -------
            Returns
            -------
            distances, np.array of type float - may be empty if there is no opposing surface
        """
        opposing_nodes = self.get_opposite_nodes(node)
        if opposing_nodes is None:
            return np.array([])
        distances = []
        node_pos = self._node_to_position[node]
        # iterate over all opposing nodes
        for on in opposing_nodes:
            # get angle for opposing node
            oangle = self.get_opposing_angle(node, angle, on)
            if not self.is_valid_angle(on, oangle):
                continue
            on_pos = self._node_to_position[on]
            distances.append(np.linalg.norm(node_pos - on_pos))
        return np.array(distances)

    def get_opposing_angle(self, node, angle, onode):
        """
            Return the angle at the opposing node, onode, that an opposing parallel
            finger must have when the other finger is placed at node with the given angle.
            ---------
            Arguments
            ---------
            node, tuple (int, int) - node where one finger is located
            angle, int - angle at which the finger is
            onode, tuple (int, int) - opposing node where the other finger is
            -------
            Returns
            -------
            oangle, float - angle of the opposing finger at the opposing node.
                The angle is within range [0, 360)
        """
        comp = self._node_to_component[node]
        ocomp = self._node_to_component[onode]
        # compute rotated y-axis
        zero_axis = self._component_to_zero_axis[comp]
        normal = self._component_to_normal[comp]
        tf = transformations.rotation_matrix(angle / 180.0 * np.pi, normal)
        rot_zero_axis = np.dot(tf[:3, :3], zero_axis)
        # project this rot_zero_axis to opposite component
        onormal = self._component_to_normal[ocomp]
        ozero_axis = self._component_to_zero_axis[ocomp]
        # rotational tf from object frame to opposite component
        ocTo = np.array([onormal, ozero_axis, np.cross(onormal, ozero_axis)])
        raxis = np.dot(ocTo, rot_zero_axis)
        # compute angle at opposing component in local frame (x is normal of plane)
        theta_prime = np.arctan2(raxis[2], raxis[1])
        theta_deg = theta_prime * 180.0 / np.pi
        if theta_deg < 0:
            theta_deg += 360
        return theta_deg

    def get_opposite_node(self, node, angle, dist=None, comp=None):
        """
            Return the node that is opposite of node and compatible
            with the given angle at node, and either
                a. closest to the given distance, or
                b. in component comp.
            You choose which option by providing only the respective argument.
            If both are provided, option a is returned.
            ---------
            Arguments
            ---------
            node, tuple - node key
            dist, float - distance
            comp, int - component
            -------
            Returns
            -------
            node, tuple - None if no such node exists
        """
        nodes = self.get_opposite_nodes(node)
        nodes = [n for n in nodes if self.is_valid_angle(n, self.get_opposing_angle(node, angle, n))]
        if nodes is None:
            return None
        if dist is not None:
            node_pos = self._node_to_position[node]
            idx = np.argmin([np.abs(np.linalg.norm(self._node_to_position[n] - node_pos) - dist) for n in nodes])
            return nodes[idx]
        else:
            comp_nodes = filter(lambda n: self._node_to_component[n] == comp, nodes)
            if len(comp_nodes) == 0:
                return None
            assert(len(comp_nodes) == 1)
            return comp_nodes[0]

    def get_component(self, node):
        """
            Return the connected component of the given node.
            ---------
            Arguments
            ---------
            node, tuple - node key
            -------
            Returns
            -------
            id, int - identifier of connected component
        """
        return self._node_to_component[node]

    def get_opposite_nodes(self, node):
        '''get the nodes that are on the opposite face of the object'''
        start_point = self._node_to_position[node]
        normal = self._component_to_normal[self._node_to_component[node]]
        goal_point = start_point - 10*normal  # 10 m
        points = self.ray_shape_intersections(start_point, goal_point)
        if len(points) < 1:
            # check in the other case, i.e. the normal is pointing inside the object
            goal_point = start_point + 10*normal  # 10 m
            points = self.ray_shape_intersections(start_point, goal_point)
            if len(points) < 1:
                return None
            if len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
                return None
        elif len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
            goal_point = start_point + 10*normal  # 10 m
            points = self.ray_shape_intersections(start_point, goal_point)
            if len(points) < 1:
                return None
            if len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
                return None

        # now we check if the first point corresponds to the starting point, and if yes we remove it
        if np.linalg.norm(points[0]-start_point) < 0.001:
            points = points[1:]

        # now get the nodes corresponding to the points
        opposite_nodes = list()
        for p in points:
            node_list = self.get_closest_nodes(p)
            opposite_nodes += node_list
        return list(set(opposite_nodes))

    def get_neighbors(self, node):
        """
            Return a list of all neighboring nodes of the given node.
        """
        return self._adjacency_list[node]

    def get_neighbor_angles(self, node, angle):
        """
            Get the neighboring angles of the given angle at the given node.
        """
        neighbor_angles = []
        lower_angle = angle - self._angle_res
        if lower_angle < 0:
            lower_angle += 360
        if self.is_valid_angle(node, lower_angle):
            neighbor_angles.append(lower_angle)
        upper_angle = angle + self._angle_res
        if self.is_valid_angle(node, upper_angle):
            neighbor_angles.append(upper_angle)
        return neighbor_angles

    def get_delta_angle(self, angle1, angle2):
        """
            Return the rotation required to move from angle1 to angle2, assuming that
            both angles are adjacent.
        """
        if abs(angle1 - angle2) <= self._angle_res:
            return angle2 - angle1
        if angle1 == 0:
            return -self._angle_res
        elif angle1 == 360 - self._angle_res:
            return self._angle_res
        raise ValueError("Invalid adjacent angles: %i, %i" % (angle1, angle2))

    def get_shortest_path(self, start, goal):
        '''finds the shortest path in only one component'''
        # sanity check on the component (if they are different no in-hand path can be found)
        if self._node_to_component[start] != self._node_to_component[goal]:
            return None
        # check what should happen in the fingers in the opposite component
        opposite_start_list = self.get_opposite_nodes(start)
        opposite_goal_list = self.get_opposite_nodes(goal)
        # get the desired component
        opposite_component_start = set()
        opposite_component_goal = set()
        for n in opposite_start_list:
            opposite_component_start.add(self._node_to_component[n])
        for n in opposite_goal_list:
            opposite_component_goal.add(self._node_to_component[n])
        admissible_opposite_components = opposite_component_start & opposite_component_goal
        if len(admissible_opposite_components) < 1:
            return None

        # initialization for Dijkstra
        distances = dict()
        q = self._adjacency_list.keys()
        prev = dict()
        for x in self._adjacency_list:
            distances[x] = sys.float_info[0]
        distances[start] = 0.0
        prev[start] = None

        # for the distance between the nodes, for now use simple Euclidian distance
        def nodes_distance(node1, node2):
            point1 = np.array(self._node_to_position[node1])
            point2 = np.array(self._node_to_position[node2])
            return np.linalg.norm(point1-point2)

        while len(q) > 0:
            current_node = min(q, key=lambda node: distances[node])
            if current_node == goal:
                break
            if distances[current_node] == sys.float_info[0]:
                return None  # no path can be found
            q.remove(current_node)
            # get the neighbors
            neighbor_nodes = self._adjacency_list[current_node]
            for n in neighbor_nodes:
                new_d = distances[current_node] + nodes_distance(n, current_node)
                # check if the opposite finger is valid
                valid = False
                opposite_nodes = self.get_opposite_nodes(n)
                for opposite_n in opposite_nodes:
                    if self._node_to_component[opposite_n] in admissible_opposite_components:
                        valid = True
                if not valid:
                    new_d = sys.float_info[0]

                if new_d < distances[n]:
                    distances[n] = new_d
                    prev[n] = current_node

        path = [goal]
        if prev[goal] is None:
            return path
        prev_node = prev[goal]
        path.append(prev_node)
        while prev[prev_node] is not None:
            prev_node = prev[prev_node]
            path.append(prev_node)
        return path

    def get_rotations(self, start_angle, goal_angle, path):
        '''return a sequence of angles to associate with the given path'''
        cc = self._node_to_component[path[0]]
        normal = self._component_to_normal[cc]
        zero_axis = self._component_to_zero_axis[cc]
        planar_zero_axis = -zero_axis[0:2]
        planar_zero_axis = planar_zero_axis/np.linalg.norm(planar_zero_axis)

        angles = [start_angle]
        angle = start_angle
        for i in range(len(path)-1, 0, -1):
            prev_node = path[i]
            next_node = path[i-1]
            next_valid_angles = self._node_to_angles[next_node]
            if angle in next_valid_angles:
                angles += [angle]
            else:
                # get an angle in common between the two (if there's the goal angle, put that)
                prev_valid_angles = self._node_to_angles[prev_node]
                angles_in_common = list(set(prev_valid_angles) & set(next_valid_angles))
                if goal_angle in angles_in_common:
                    next_angle = goal_angle
                else:
                    # get the angle closest to what we already have
                    _, next_angle = min(enumerate(angles_in_common), key=lambda x: abs(x[1]-angle))
                angles += [next_angle]
                angle = next_angle
        angles += [goal_angle]
        return angles

    def plot_graph(self):
        '''Use to visualize the shape and the graph'''
        object_mesh = mesh.Mesh.from_file(self._object_shape_file)
        self._figure = plt.figure()
        self._axes = mplot3d.Axes3D(self._figure)
        self._axes.set_xlabel('x [m]')
        self._axes.set_ylabel('y [m]')
        self._axes.set_zlabel('z [m]')
        self._axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self._mesh_scale * object_mesh.vectors))
        scale = object_mesh.points.flatten(-1)
        self._axes.auto_scale_xyz(scale, scale, scale)
        for n in self._adjacency_list:
            point1 = self._node_to_position[n]
            for m in self._adjacency_list[n]:
                point2 = self._node_to_position[m]
                xline = np.array([point1[0], point2[0]])
                yline = np.array([point1[1], point2[1]])
                zline = np.array([point1[2], point2[2]])
                self._axes.plot3D(xline, yline, zline, 'gray')
        self._figure.show()

    def plot_path(self, path, angles, color='yellow'):
        '''use to visualize the given path'''
        for i in range(len(path)-1, 0, -1):
            node1 = path[i]
            node2 = path[i-1]
            point1 = self._node_to_position[node1]
            point2 = self._node_to_position[node2]
            xline = np.array([point1[0], point2[0]])
            yline = np.array([point1[1], point2[1]])
            zline = np.array([point1[2], point2[2]])
            self._axes.plot3D(xline, yline, zline, color)
            # plot the current fingers
            curr_angle = angles[len(angles)-i-1]
            self.plot_finger(node2, curr_angle, color)
            self.plot_finger(node1, curr_angle, color)
        self._figure.show()

    def plot_finger(self, node, angle, color='yellow'):
        '''use to plot a finger in a given configuration'''
        point = self._node_to_position[node]
        # the zero axis is opposite to the finger axis in the convention
        normal = self._component_to_normal[self._node_to_component[node]]
        zero_axis = self._component_to_zero_axis[self._node_to_component[node]]
        # rotate the zero angle axis by the angle
        theta = angle*math.pi/180
        # get the axis-angle matrix
        rrt = np.outer(normal, normal)
        Sr = np.array([[0, -normal[2], normal[1]], [normal[2], 0, -normal[0]], [-normal[1], normal[0], 0]])
        R = rrt + (np.identity(3) - rrt)*math.cos(theta) + Sr*math.sin(theta)
        # R = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]]
        finger_axis = self._finger_length*np.dot(R, zero_axis)
        xline = [point[0], point[0] + finger_axis[0]]
        yline = [point[1], point[1] + finger_axis[1]]
        zline = [point[2], point[2] + finger_axis[2]]  # assumed no variation in z. To be improved
        self._axes.plot3D(xline, yline, zline, color)
        self._figure.show()

    def visualize(self, block=True):
        '''use to visualize the figures'''
        plt.draw()
        plt.show(block=block)

    def convert_path(self, inhand_path, initial_grasp_pose, fingertip_contact):
        """
            Convert a given inhand path into a sequence of pushing poses for the second end-effector.
            An inhand path is a sequence of translations and rotations. A translation is a directional
            vector in the original object frame. A rotation is an angle in radians around the normal
            at the fingertip contact.

            Arguments
            ---------
            inhand_path, tuple (translations, rotations), 
                where translations is a list of np.arrays of shape (3,)
                rotations is a list of floats  - Both lists are assumed to have the same length
            initial_grasp_pose, np.array of shape (4, 4) - the pose of the grasping gipper in the original object frame oTe 
            fingertip_contacts, np.array of shape (2, 3) - the initial positions of the figertips in the original object frame

            Returns
            -------
            pushes, a list of tuples (start, end, rot_center), 
                where start and end are np.array of shape (4, 4) representing the initial and final pose of the pusher
                rot_center is either None or a np.array of shape (3,) denoting the center of rotation for a rotational push
        """
        if self._push_path_computer is None:
            raise RuntimeError("Could not convert path. Object model not initialized.")
        return self._push_path_computer.convert_path(inhand_path, initial_grasp_pose, fingertip_contact)
