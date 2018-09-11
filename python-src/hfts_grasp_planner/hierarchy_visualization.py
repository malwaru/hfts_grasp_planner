import igraph
import rospy
import numpy as np
from std_msgs.msg import String


class FreeSpaceProximitySamplerVisualizer(object):
    """ A debugging tool to visualize the current state
        of a FreeSpaceProximitySampler """

    def __init__(self, robot):
        self._graph = igraph.Graph()
        self._labels_to_ids = {}
        self._nodes_cache = {}
        self.robot = robot
        self._env = self.robot.GetEnv()
        self._mesh_handle = None
        self._graph_publisher = rospy.Publisher('/goal_region_graph', String, queue_size=1)
        rospy.Subscriber('/node_select', String, self._ros_callback)

    def _ros_callback(self, msg):
        unique_label = msg.data
        if unique_label in self._nodes_cache:
            node = self._nodes_cache[unique_label]
            config = node.get_active_configuration()
            rospy.logdebug(
                '[HierarchyVisualizer::_ros_callback] Got a request for node ' + str(unique_label))
            rospy.logdebug('[HierarchyVisualizer::_ros_callback] Cost debug data: ' + str(node.cost_debug_data))
            if config is not None:
                rospy.logdebug('[HierarchyVisualizer::_ros_callback] Request to ' +
                               'show config ' + str(config))
                self.robot.SetActiveDOFValues(config)
                b_in_collision = self._env.CheckCollision(self.robot)
                b_self_collision = self.robot.CheckSelfCollision()
                if not b_in_collision and not b_self_collision:
                    rospy.logdebug('[HierarchyVisualizer::_ros_callback] Configuration' +
                                   ' is collision-free!')
                    if node.is_goal():
                        rospy.logdebug('HierarchyVisualizer::_ros_callback] The ' +
                                       ' selected config is a goal!')
                else:
                    if b_in_collision:
                        rospy.logdebug('[HierarchyVisualizer::_ros_callback] Configuration' +
                                       ' is in collision.')
                    elif b_self_collision:
                        rospy.logdebug('[HierarchyVisualizer::_ros_callback] Configuration' +
                                       ' is in self-collision.')
            # if we have an object pose, visualize it
            obj_pose = node.get_goal_sampler_hierarchy_node().get_additional_data()
            if obj_pose is not None:
                # rospy.logdebug("[HierarchyVisualizer::_ros_callback] Object pose: " + str(obj_pose))
                grabbed_objects = self.robot.GetGrabbed()
                if len(grabbed_objects) > 0:
                    grabbed_object = grabbed_objects[0]
                    link = grabbed_object.GetLinks()[0]
                    geom = link.GetGeometries()[0]
                    mesh = geom.GetCollisionMesh()
                    tf_points = np.dot(mesh.vertices, obj_pose[:3, :3].transpose()) + obj_pose[:3, 3]
                    self._mesh_handle = self._env.drawtrimesh(tf_points, mesh.indices, np.array([0.0, 0.9, 0.02, 0.7]))
                else:
                    rospy.logwarn("[HierarchyVisualizer::_ros_callback] No grabbed object!")
            else:
                rospy.logdebug('[FreeSpaceProximitySamplerVisualizer::_ros_callback] No object pose!')
        else:
            rospy.logwarn('[FreeSpaceProximitySamplerVisualizer::_ros_callback] Received unknown node label.')

    def _add_node(self, parent_id, node, b_is_active):
        label = node.get_hashable_label()
        if label in self._labels_to_ids:
            node_id = self._labels_to_ids[label]
        else:
            self._graph.add_vertex(name=label)
            node_id = len(self._graph.vs) - 1
            self._labels_to_ids[label] = node_id
            self._graph.vs[node_id]['unique_label'] = label
            if parent_id is not None:
                self._graph.add_edge(parent_id, node_id)
        self._graph.vs[node_id]['temperature'] = node.get_T()
        self._graph.vs[node_id]['isActive'] = b_is_active
        if node.get_max_num_children() != 0:
            self._graph.vs[node_id]['coverage'] = node.get_num_children() / float(node.get_max_num_children())
        else:
            self._graph.vs[node_id]['coverage'] = 1.0
        if node.get_max_num_leaves_in_branch() != 0:
            self._graph.vs[node_id]['branch_coverage'] = node.get_num_leaves_in_branch() / \
                float(node.get_max_num_leaves_in_branch())
        else:
            self._graph.vs[node_id]['branch_coverage'] = 1.0
        self._nodes_cache[label] = node
        return node_id

    def clear(self):
        rospy.loginfo('Clearing FreeSpaceProximitySamplerVisualizer')
        self._labels_to_ids = {}
        self._graph = igraph.Graph()
        self._nodes_cache = {}

    def draw_hierarchy(self, root_node):
        self.draw_hierarchy_recursively(None, root_node, True)
        pickled_graph = self._graph.write_pickle()
        self._graph_publisher.publish(pickled_graph)

    def draw_hierarchy_recursively(self, parent_id, node, b_is_active):
        node_id = self._add_node(parent_id, node, b_is_active)
        active_children_map = {}
        for child in node.get_active_children():
            active_children_map[child.get_hashable_label()] = True
        for child in node.get_children():
            self.draw_hierarchy_recursively(node_id, child, child.get_hashable_label() in active_children_map)
