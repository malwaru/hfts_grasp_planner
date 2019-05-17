import numpy as np
import math
import sys
from pycaster import pycaster
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
import scipy.spatial
from collections import defaultdict

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
        self._supervoxel_angle_to_angular_component = None

        # data structures for the nearest neighbor
        self._kdtree = None
        self._idx_to_supervoxel = None
        self._supervoxel_to_angular_component = None

        #stuff for opposite finger component
        self._caster = None
        self._object_shape_file = None

        #length of the gripper's finger 
        self._finger_length = 0.04 #Yumi's finger
        self.current_node = None
        self.current_angle = None

    def set_current_node(self, node):
        self.current_node = node

    def get_current_node(self):
        if self.current_node is None:
            print("Current DMG Node not set")
            assert(not self.current_node is None)
        else:
            return self.current_node

    def set_current_angle(self, angle):
        self.current_angle = angle

    def get_current_angle(self):
        if self.current_angle is None:
            assert(not self.current_node is None)
            print("Current DMG Angle not set, taking default")
            self.current_angle = self._node_to_angles[self.current_node][0]
        return self.current_angle

    def vector_angle(self, v1, v2):
        # v1 is your first vector
        # v2 is your second vector
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)
        

    def set_object_shape_file(self, filename):
        '''read the object shape file'''
        self._object_shape_file = filename
        self._caster = pycaster.rayCaster.fromSTL(filename, scale=1.0) #the object used to be in mm

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
            nodes_to_list_of_nodes[node]=[]
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

    def read_node_to_angles(self, filename):
        '''reads the admissible angles in one node. These nodes are in degrees!'''
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

    def get_closest_nodes(self, point):
        '''returns the nodes whose center is the closest to the given point. They can be two if there are nodes with different angular component'''
        (_, idx) = self._kdtree.query(point)
        supervoxel = self._idx_to_supervoxel[idx]
        #this supervoxel can correspond to more than one node
        angular_components = self._supervoxel_to_angular_component[supervoxel]
        nodes = list()
        for c in angular_components:
            nodes.append((supervoxel, c))
        return nodes


    def ray_shape_intersections(self, start_point, goal_point):
        '''get the intersection of a ray with the object's shape (used for checks on the opposite finger)'''
        #important thing: the mesh is in mm, while the graph points are in m!
        source = 1000.0*start_point
        destination = 1000.0*goal_point
        intersections = self._caster.castRay(source, destination)
        intersection_points = list()
        for x in intersections:
            intersection_points.append(np.array(x)/1000.0)
        return intersection_points

    def get_opposite_nodes(self, node):
        '''get the nodes that are on the opposite face of the object'''
        start_point = self._node_to_position[node]
        normal = self._component_to_normal[self._node_to_component[node]]
        goal_point = start_point - 10*normal #10 m
        points = self.ray_shape_intersections(start_point, goal_point)
        if len(points) < 1:
            #check in the other case, i.e. the normal is pointing inside the object
            goal_point = start_point + 10*normal #10 m
            points = self.ray_shape_intersections(start_point, goal_point)
            if len(points) < 1:
                return None
            if len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
                return None
        elif len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
            goal_point = start_point + 10*normal #10 m
            points = self.ray_shape_intersections(start_point, goal_point)
            if len(points) < 1:
                return None
            if len(points) == 1 and np.linalg.norm(points[0]-start_point) < 0.001:
                return None

        #now we check if the first point corresponds to the starting point, and if yes we remove it
        if np.linalg.norm(points[0]-start_point) < 0.001:
            points = points[1:]

        #now get the nodes corresponding to the points
        opposite_nodes = list()
        for p in points:
            node_list = self.get_closest_nodes(p)
            opposite_nodes += node_list
        return opposite_nodes 


    def get_shortest_path(self, start, goal):
        '''finds the shortest path in only one component'''
        #sanity check on the component (if they are different no in-hand path can be found)
        if self._node_to_component[start] != self._node_to_component[goal]:
            return None
        #check what should happen in the fingers in the opposite component
        opposite_start_list = self.get_opposite_nodes(start)
        opposite_goal_list = self.get_opposite_nodes(goal)
        #get the desired component
        opposite_component_start = set()
        opposite_component_goal = set()
        for n in opposite_start_list:
            opposite_component_start.add(self._node_to_component[n])
        for n in opposite_goal_list:
            opposite_component_goal.add(self._node_to_component[n])
        admissible_opposite_components = opposite_component_start & opposite_component_goal
        if len(admissible_opposite_components) < 1:
            return None

        #initialization for Dijkstra
        distances = dict()
        q = self._adjacency_list.keys()
        prev = dict()
        for x in self._adjacency_list:
            distances[x] = sys.float_info[0]
        distances[start] = 0.0
        prev[start] = None
        
        #for the distance between the nodes, for now use simple Euclidian distance
        def nodes_distance(node1, node2):
            point1 = np.array(self._node_to_position[node1])
            point2 = np.array(self._node_to_position[node2])
            return np.linalg.norm(point1-point2)

        while len(q) > 0:
            current_node = min(q, key=lambda node: distances[node])
            if current_node == goal:
                break
            if distances[current_node] == sys.float_info[0]:
                return None #no path can be found
            q.remove(current_node)
            #get the neighbors
            neighbor_nodes = self._adjacency_list[current_node]
            for n in neighbor_nodes:
                new_d = distances[current_node] + nodes_distance(n, current_node)
                #check if the opposite finger is valid
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
                #get an angle in common between the two (if there's the goal angle, put that)
                prev_valid_angles = self._node_to_angles[prev_node]
                angles_in_common = list(set(prev_valid_angles) & set(next_valid_angles))
                if goal_angle in angles_in_common:
                    next_angle = goal_angle
                else:
                    #get the angle closest to what we already have
                    _ , next_angle = min(enumerate(angles_in_common), key=lambda x: abs(x[1]-angle))
                angles += [next_angle]
                angle = next_angle
        angles += [goal_angle]
        return angles

    def get_center_node(self, reference_node):
        ''' 
        Gets the center point between a reference and its opposite nodes 
        on the surface of an object. The Resulting node will be inside the object 
        '''

        opposite_nodes = self.get_opposite_nodes(reference_node)
        ref_node_component = self._node_to_component[reference_node]
        ref_node_normal = self._component_to_normal[ref_node_component]
        
        opposite_node = None
        best_angle = np.Inf
        for node in opposite_nodes:
            node_comp = self._node_to_component[node]
            node_normal = self._component_to_normal[node_comp]

            angle = self.vector_angle(ref_node_normal, node_normal)
            if (angle >= 0 and angle <= 5) or (angle <= 180 and angle >= 175):
                opposite_node = node
                break

        if opposite_node is None:
            # No opposite node found
            return None
        
        ref_node_position = self._node_to_position[reference_node]
        opp_node_position = self._node_to_position[opposite_node]

        return np.divide( np.add(ref_node_position, opp_node_position), 2.0)

    def make_transform_matrix(self, reference_node, target_angle=None):
        ''' 
        Transforms the object matrix to reference node matrix
        reference_node: the node from _adjecancy_list
        target_angle: one of the possible angles in degrees from _node_to_angles list of reference_node
        '''

        assert(reference_node in self._adjacency_list.keys())

        center = self.get_center_node(reference_node)
        if center is None:
            # No valid opposite node found
            return None

        component = self._node_to_component[reference_node]
        zero_axis = self._component_to_zero_axis[component]
        normal = self._component_to_normal[component]
        angles = self._node_to_angles[reference_node]
        
        if target_angle is None:
            angle = angles[0]-180
        else:
            assert(target_angle >= angles[0] and target_angle <= angles[len(angles)-1])
            angle = target_angle-180

        #rotate the zero angle axis by the angle
        theta = angle*np.pi/180.0
        Rot = np.array([ [1,0,0], [0, np.cos(theta), -np.sin(theta) ], [0, np.sin(theta), np.cos(theta)] ])
        #get the axis-angle matrix 
        rrt = np.outer(normal, normal)
        Sr = np.array([[0, -normal[2], normal[1]], [normal[2], 0, -normal[0]], [-normal[1], normal[0], 0]])
        R = rrt + (np.identity(3) - rrt)*np.cos(theta) + Sr*np.sin(theta)

        # Rotation transform according to end effector
        tR = np.array([ [1,0,0], [0,0,-1], [0,1,0] ] )
        
        finger_axis = 1.0*np.dot(R, zero_axis)
        finger_axis = finger_axis/np.linalg.norm(finger_axis)
        M = np.array([normal,  zero_axis, np.cross(normal,zero_axis)])
        M = np.dot(M, Rot)
        R = np.dot(tR , M)

        # Make the 4x4 transform matrix
        matrix = np.zeros(16).reshape(4,4)
        matrix[3][3] = 1
        matrix[:3,:3] = R
        matrix[:3,3] = center

        return matrix

    def create_node_angle_pairs(self, node_list):
        ''' 
        Converts a list of nodes into node angle pairs
        i.e ((x,y), angle)
        '''
        pair_list = list()
        for node in node_list:
            angles = self._node_to_angles[node]
            for angle in angles:
                pair_list.append( (node, angle) )
        return pair_list

    def _run_bfs_on_nodes(self, reference_node):
        ''' 
        Runs BFS over the self._adjacency_list
        Essentially serializing the nodes starting from reference_node 
        '''
        start = reference_node

        # keep track of all visited nodes
        explored = []
        # keep track of nodes to be checked
        queue = [start]
    
        # keep looping until there are nodes still to be checked
        while queue:
            # pop shallowest node (first node) from queue
            node = queue.pop(0)
            if node not in explored:
                # add node to list of checked nodes
                explored.append(node)
                neighbours = self._adjacency_list[node]
    
                # add neighbours of node to queue
                for neighbour in neighbours:
                    queue.append(neighbour)
        return explored

    def run_dijsktra(self, reference_node, reference_angle=None, angle_weight=1.0, to_matrix=True):
        ''' 
        Creates a list of search order for the possible grasps, using dijsktra algorithm
        reference_node: the node from _adjecancy_list
        reference_angle: one of the possible angles in degrees from _node_to_angles list of reference_node
        angle_weight: positive value, prioritizing angle over position.
        '''

        assert(reference_node in self._adjacency_list.keys())
        if reference_angle is None:
            reference_angle = self._node_to_angles[reference_node][0]

        # Create Graph
        edges = defaultdict(list)
        distances = {}

        #for the distance between the nodes, for now use simple Euclidian distance
        def nodes_distance(node1, node2):
            point1 = np.array(self._node_to_position[node1])
            point2 = np.array(self._node_to_position[node2])
            return abs(np.linalg.norm(point1-point2))

        def add_edge(from_node, to_node, distance):
            edges[from_node].append(to_node)
            edges[to_node].append(from_node)
            distances[(from_node, to_node)] = distance

        node_list = self._run_bfs_on_nodes(reference_node)
        nodes = set( self.create_node_angle_pairs(node_list) )
        for node in nodes:
            for next_node in nodes:
                if not next_node is node and (next_node[0] in self._adjacency_list[node[0]] or next_node[0] is node[0]):
                    add_edge(node, next_node, 
                            (nodes_distance(node[0], next_node[0])) + (angle_weight*abs(node[1]-next_node[1])) )
        
        # Dijsktra Start
        visited = { (reference_node, reference_angle) : 0}
        path = {}

        while nodes: 
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node

            if min_node is None:
                break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in edges[min_node]:
                weight = current_weight + distances[(min_node, edge)]
                
                if edge not in visited or weight < visited[edge]:
                    visited[edge] = weight
                    path[edge] = min_node

        grasp_order = sorted(visited, key=visited.get)
        grasp_order.remove((reference_node, reference_angle))
        grasp_order.insert(0, (reference_node, reference_angle))

        # Convert Each grasp in grasp order to transformation matrix
        if to_matrix:
            grasp_order_matrix = []
            for grasp_data in grasp_order:
                assert(len(grasp_data) == 2)
                node = grasp_data[0]
                angle = grasp_data[1]
                matrix = self.make_transform_matrix(node, angle)
                if not matrix is None:
                    grasp_order_matrix.append(matrix)
            return grasp_order_matrix
        else:
            return grasp_order

        return visited, path

    def plot_graph(self):
        '''Use to visualize the shape and the graph'''
        object_mesh = mesh.Mesh.from_file(self._object_shape_file)
        self._figure = plt.figure()
        self._axes = mplot3d.Axes3D(self._figure)
        self._axes.set_xlabel('x [mm]')
        self._axes.set_ylabel('y [mm]')
        self._axes.set_zlabel('z [mm]')
        self._axes.add_collection3d(mplot3d.art3d.Poly3DCollection(object_mesh.vectors))
        scale = object_mesh.points.flatten(-1)
        self._axes.auto_scale_xyz(scale, scale, scale)
        for n in self._adjacency_list:
            point1 = 1000*self._node_to_position[n]
            for m in self._adjacency_list[n]:
                point2 = 1000*self._node_to_position[m]
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
            point1 = 1000*self._node_to_position[node1]
            point2 = 1000*self._node_to_position[node2]
            xline = np.array([point1[0], point2[0]])
            yline = np.array([point1[1], point2[1]])
            zline = np.array([point1[2], point2[2]])
            self._axes.plot3D(xline, yline, zline, color)
            #plot the current fingers
            curr_angle = angles[len(angles)-i-1]
            self.plot_finger(node2, curr_angle, color)
            self.plot_finger(node1, curr_angle, color)
        self._figure.show()


    def plot_finger(self, node, angle, color = 'yellow'):
        '''use to plot a finger in a given configuration'''
        point = self._node_to_position[node]    
        #the zero axis is opposite to the finger axis in the convention
        normal = self._component_to_normal[self._node_to_component[node]]
        zero_axis = self._component_to_zero_axis[self._node_to_component[node]]      
        #rotate the zero angle axis by the angle
        theta = angle*math.pi/180
        #get the axis-angle matrix 
        rrt = np.outer(normal, normal)
        Sr = np.array([[0, -normal[2], normal[1]], [normal[2], 0, -normal[0]], [-normal[1], normal[0], 0]])
        R = rrt + (np.identity(3) - rrt)*math.cos(theta) + Sr*math.sin(theta)
        #R = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]]
        finger_axis = self._finger_length*np.dot(R, zero_axis)
        xline = [1000*point[0], 1000*point[0] + 1000*finger_axis[0]]
        yline = [1000*point[1], 1000*point[1] + 1000*finger_axis[1]]
        zline = [1000*point[2], 1000*point[2] + 1000*finger_axis[2]] #assumed no variation in z. To be improved
        self._axes.plot3D(xline, yline, zline, color)
        self._figure.show()

    def visualize(self, block=True):
        '''use to visualize the figures'''
        plt.draw()
        plt.show(block=block)

    