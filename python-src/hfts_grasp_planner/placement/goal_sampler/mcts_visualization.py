import igraph
import rospy
import numpy as np
from std_msgs.msg import String
from ast import literal_eval as make_tuple


class MCTSVisualizer(object):
    """ A debugging tool to visualize the current state
        of a Monte Carlo Tree """

    def __init__(self, robot, target_obj):
        self._graph = igraph.Graph()
        self._labels_to_ids = {}  # maps key to integer index for accessing node data in self._graph
        self._nodes_cache = {}  # maps key to MCTSNode
        self._prev_selected_node = (None, 0)  # tuple (node, solution id to show)
        self._root_node = None  # root MCTSNode
        if robot is not None:
            self.robot = robot
            self._env = self.robot.GetEnv()
        self.target_obj = target_obj
        self._mesh_handle = None
        self._graph_publisher = rospy.Publisher('/mct_graph', String, queue_size=1)
        rospy.Subscriber('/mct_node_select', String, self._ros_callback)

    def _ros_callback(self, msg):
        key = make_tuple(msg.data)
        if key in self._nodes_cache:
            node = self._nodes_cache[key]
            if len(node.solutions) == 0:
                rospy.loginfo("[MCTSVisualizer] Node with id %s has no solution associated with it." % str(key))
                return
            if node == self._prev_selected_node[0]:  # the node was selected before, so go to the next solution
                solution_id = self._prev_selected_node[1]
                next_sol_id = (self._prev_selected_node[1] + 1) % len(node.solutions)
                self._prev_selected_node = (node, next_sol_id)
            else:  # newly selected node
                solution_id = 0
                self._prev_selected_node = (node, solution_id)
            config = node.solutions[solution_id].arm_config  # may still be None!
            rospy.logdebug("[MCTSVisualizer::_ros_callback] Got a request for node %s" % str(key))
            rospy.logdebug("[MCTSVisualizer::_ros_callback: solution data: %s" % str(node.solutions[solution_id].data))
            if config is not None and self.robot is not None:
                rospy.logdebug('[MCTSVisualizer::_ros_callback] Request to show config ' + str(config))
                self.robot.SetActiveDOFs(node.solutions[solution_id].manip.GetArmIndices())
                self.robot.SetActiveDOFValues(config)
                b_in_collision = self._env.CheckCollision(self.robot)
                b_self_collision = self.robot.CheckSelfCollision()
                if not b_in_collision and not b_self_collision:
                    rospy.logdebug('[MCTSVisualizer::_ros_callback] Configuration is collision-free!')
                else:
                    if b_in_collision:
                        rospy.logdebug('[MCTSVisualizer::_ros_callback] Configuration is in collision.')
                    elif b_self_collision:
                        rospy.logdebug('[MCTSVisualizer::_ros_callback] Configuration is in self-collision.')
            if self.target_obj is not None:
                obj_pose = node.solutions[solution_id].obj_tf
                link = self.target_obj.GetLinks()[0]
                geom = link.GetGeometries()[0]
                mesh = geom.GetCollisionMesh()
                tf_points = np.dot(mesh.vertices, obj_pose[:3, :3].transpose()) + obj_pose[:3, 3]
                self._mesh_handle = self._env.drawtrimesh(tf_points, mesh.indices, np.array([0.0, 0.9, 0.02, 0.7]))
        else:
            rospy.logwarn('[FreeSpaceProximitySamplerVisualizer::_ros_callback] Received unknown node label.')

    def add_node(self, node):
        if node.key in self._labels_to_ids:
            node_id = self._labels_to_ids[node.key]
        else:
            self._graph.add_vertex(name=node.key)
            node_id = len(self._graph.vs) - 1
            self._labels_to_ids[node.key] = node_id
            self._graph.vs[node_id]['key'] = str(node.key)
            if node.parent is not None:
                if node.parent.key not in self._labels_to_ids:
                    raise ValueError("Parent node of %s has not been added to the visualizer!" % node.key)
                parent_id = self._labels_to_ids[node.parent.key]
                self._graph.add_edge(parent_id, node_id)
            else:
                self._root_node = node
        # self._graph.vs[node_id]['uct'] = node.last_uct_value
        # self._graph.vs[node_id]['fup'] = node.last_fup_value
        self._graph.vs[node_id]['num_visits'] = node.num_visits
        self._graph.vs[node_id]['acc_rewards'] = node.acc_rewards
        self._graph.vs[node_id]['num_constructions'] = node.num_constructions
        self._graph.vs[node_id]['num_new_valid_constr'] = node.num_new_valid_constr
        self._nodes_cache[node.key] = node
        return node_id

    def clear(self):
        rospy.loginfo('Clearing MCTSVisualizer')
        self._prev_selected_node = (None, 0)
        self._labels_to_ids = {}
        self._graph = igraph.Graph()
        self._nodes_cache = {}
        self._root_node = None

    def render(self, bupdate_data=True):
        if bupdate_data:
            self._update_data()
        pickled_graph = self._graph.write_pickle()
        self._graph_publisher.publish(pickled_graph)

    def _update_data(self):
        children = [self._root_node]
        while children:
            node = children.pop()
            node_id = self._labels_to_ids[node.key]
            # self._graph.vs[node_id]['uct'] = node.last_uct_value
            # self._graph.vs[node_id]['fup'] = node.last_fup_value
            self._graph.vs[node_id]['num_visits'] = node.num_visits
            self._graph.vs[node_id]['acc_rewards'] = node.acc_rewards
            self._graph.vs[node_id]['num_constructions'] = node.num_constructions
            self._graph.vs[node_id]['num_new_valid_constr'] = node.num_new_valid_constr
            children.extend(node.children.values())  # the order in which we update does not matter
