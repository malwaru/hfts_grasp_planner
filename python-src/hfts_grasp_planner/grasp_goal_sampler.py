#!/usr/bin/env python
"""This module contains a wrapper class of the HFTS Grasp Sampler."""

from hfts_grasp_planner.core import HFTSSampler, HFTSNode
from sampler import GoalHierarchy
import rospy
import numpy


class HFTSNodeDataExtractor:
    def extractData(self, hierarchyInfo):
        return hierarchyInfo.get_hand_config()

    def getCopyFunction(self):
        return numpy.copy


# TODO this module is unnecessary overhead, we should let HFTS Grasp planner simply implement the required
# TODO interface itself.
class GraspGoalSampler(GoalHierarchy):
    """ Wrapper class for the HFTS Grasp Planner/Sampler that allows a full black box usage."""
    class GraspGoalNode(GoalHierarchy.GoalHierarchyNode):
        """
            Implementation of GoalHierarchyNode for HFTS grasp planner.
        """
        def __init__(self, hfts_node):
            self._hfts_node = hfts_node
            config = hfts_node.get_arm_configuration()
            if config is not None:
                config = numpy.concatenate((config, hfts_node.get_pre_grasp_config()))
            super(GraspGoalSampler.GraspGoalNode, self).__init__(config)

        def is_valid(self):
            return self._hfts_node.is_valid()

        def is_goal(self):
            """
                Return whether this node represents a goal. 
                In addition to being valid, a goal must represent a configuration that is 
                fulfilling all goal criteria.
            """
            return self._hfts_node.is_goal()

        def hierarchy_info_str(self):
            """
                Return a string representation of the hierarchy information for debugging and printing.
            """
            return str(self._hfts_node.get_labels())

        def get_num_possible_children(self):
            """
                Return the number of possible children of this node.
            """
            return self._hfts_node.get_num_possible_children()

        def get_num_possible_leaves(self):
            """
                Return the number of possible leaves in the subranch rooted at this node.
            """
            return self._hfts_node.get_num_possible_leaves()

        def get_hashable_label(self):
            """
                Return a unique and hashable identifier for this node.
            """
            return self._hfts_node.get_unique_label()

        def get_local_hashable_label(self):
            """
                Return a hashable identifier for this node that is unique with respect to its parent.
                I.e. it should uniquely identify it among its siblings.
            """
            return self.get_hashable_label()

        def get_label(self):
            """
                Return a unique identifier for this node.
            """
            return self._hfts_node.get_labels()

        def get_depth(self):
            """
                Return the depth of this node.
            """
            return self._hfts_node.get_depth()

        def get_additional_data(self):
            """
                Return optional additional data associated with this node.
                This may be any python object that stores additional information that an external
                caller may be interested in, such as the hand configuration of a grasp, or a hand-opening
                policy for dropping an object, etc.
            """
            return self._hfts_node.get_hand_config()

        def is_extendible(self):
            """
                Return whether this node is extendible, i.e. it has children.
            """
            return self._hfts_node.is_extendible()

        def get_quality(self):
            """
                Return the quality associated with this node.
                The quality is assumed to be floating point number in range (-infty, 1.0]
            """
            return self._hfts_node.get_quality()

        def get_white_list(self):
            """
                Return a white list of locally unique ids of this nodes children
                that can be sampled from this node.
            """
            # TODO what should this return??? Is this information even contained in a HFTSNode?
            return []

    def __init__(self,
                 object_io_interface,
                 hand_path,
                 hand_cache_file,
                 hand_config_file,
                 hand_ball_file,
                 planning_scene_interface,
                 visualize=False,
                 open_hand_offset=0.1):
        """ Creates a new wrapper.
            @param object_io_interface IOObject Object that handles IO requests
            @param hand_path Path to OpenRAVE hand file
            :hand_cache_file: Path to where the hand specific data is/can be stored.
            :hand_config_file: Path to hand configuration file containing required additional hand information
            :hand_ball_file: Path to hand ball file containing ball approximations of hand
            @param planning_scene_interface OpenRAVE environment with some additional information
                                            containing the robot and its surroundings.
            @param visualize If true, the internal OpenRAVE environment is set to be visualized
            (only works if there is no other OpenRAVE viewer in this process)
            @param open_hand_offset Value to open the hand by. A grasp is in contact with the target object,
            hence a grasping configuration is always in collision. To enable motion planning to such a
            configuration we open the hand by some constant offset.
            """
        self.grasp_planner = HFTSSampler(object_io_interface=object_io_interface,
                                         vis=visualize,
                                         scene_interface=planning_scene_interface)
        self.grasp_planner.set_max_iter(100)
        self.open_hand_offset = open_hand_offset
        self.root_node = GraspGoalSampler.GraspGoalNode(self.grasp_planner.get_root_node())
        self.load_hand(hand_path, hand_cache_file, hand_config_file, hand_ball_file)

    def sample(self, depth_limit, post_opt=True):
        """ Samples a grasp from the root level. """
        return self.sample_warm_start(self.root_node, depth_limit, post_opt=post_opt)

    def sample_warm_start(self, hierarchy_node, depth_limit, label_cache=None, post_opt=False):
        """ Samples a grasp from the given node on. """
        rospy.logdebug('[GoalSamplerWrapper] Sampling a grasp from hierarchy depth ' + str(hierarchy_node.get_depth()))
        sampled_node = self.grasp_planner.sample_grasp(node=hierarchy_node._hfts_node,
                                                       depth_limit=depth_limit,
                                                       post_opt=post_opt,
                                                       label_cache=label_cache,
                                                       open_hand_offset=self.open_hand_offset)
        return GraspGoalSampler.GraspGoalNode(sampled_node)

    def load_hand(self, hand_path, hand_cache_file, hand_config_file, hand_ball_file):
        """ Reset the hand being used. @see __init__ for parameter description. """
        self.grasp_planner.load_hand(hand_file=hand_path,
                                     hand_cache_file=hand_cache_file,
                                     hand_config_file=hand_config_file,
                                     hand_ball_file=hand_ball_file)

    def set_object(self, obj_id, model_id=None):
        """ Set the object.
            @param obj_id String identifying the object.
            @param model_id (optional) Name of the model data. If None, it is assumed to be identical to obj_id
        """
        self.grasp_planner.load_object(obj_id=obj_id, model_id=model_id)
        self.root_node = GraspGoalSampler.GraspGoalNode(self.grasp_planner.get_root_node())

    def set_max_iter(self, iterations):
        self.grasp_planner.set_max_iter(iterations)

    def get_max_depth(self):
        return self.grasp_planner.get_maximum_depth()

    def get_root(self):
        return self.root_node

    def set_parameters(self, **kwargs):
        self.grasp_planner.set_parameters(**kwargs)
