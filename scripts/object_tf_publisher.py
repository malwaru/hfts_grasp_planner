#! /usr/bin/env python

"""
    This file publishes the tf of the object to be manipulated given the frames of the markers on the object (4 markers)

    @author: Silvia Cruciani (cruciani@kth.se)
"""

import rospy
import tf
import yaml
import numpy as np
from apriltags_ros.msg import AprilTagDetectionArray

def inverse_transform(transform):
    """
        Returns the inverse transformation matrix of the given matrix.
        The given matrix is assumed to be an affine 4x4 transformation matrix (type numpy array)
    """
    inv_transform = np.eye(4)
    inv_transform[:3, :3] = np.transpose(transform[:3, :3])
    inv_transform[:3, 3] = np.dot(-1.0 * inv_transform[:3, :3], transform[:3, 3])
    return inv_transform

class ObjectTFPublisher():
    """docstring for ObjectTFPublisher"""
    def __init__(self, object_names, model_path, topic_name):
        self._marker_info = {}
        self._marker_to_object = {}
        self._publisher = tf.broadcaster.TransformBroadcaster() 
        self._object_names = object_names
        self._load_marker_data(model_path)
        rospy.Subscriber(topic_name, AprilTagDetectionArray, self._receive_tags, queue_size=1)

    def _load_marker_data(self, model_path):
        for obj_name in self._object_names:
            # open yaml containing marker information
            info_file = model_path + '/' + obj_name + '/markers.yaml'
            try:
                with open(info_file, 'r') as yaml_file:
                    info = yaml.load(yaml_file)
                    marker_tfs = {}
                    for entry in info:
                        # in our file we use the convention [w, x, y, z], ROS uses [x, y, z, w] though
                        quat = entry['rot']
                        oTm = tf.transformations.quaternion_matrix((quat[1], quat[2], quat[3], quat[0]))
                        oTm[:3, 3] = entry['pos']
                        marker_tfs[entry['id']] = inverse_transform(oTm)
                        self._marker_to_object[entry['id']] = obj_name
                    self._marker_info[obj_name] = marker_tfs
            except IOError as e:
                rospy.logerr("Could not find marker information for object %s at path %s. Error: %s" % (obj_name, model_path, str(e)))

    def _receive_tags(self, data):
        if len(data.detections) < 1:
            return
        # keep track which tfs we have already published in this iteration
        published_object_tfs = set()
        for entry in data.detections:
            if entry.id in self._marker_to_object:
                obj_name = self._marker_to_object[entry.id]
                if obj_name in published_object_tfs:
                    continue
                # get marker tf
                mTo = self._marker_info[obj_name][entry.id]
                # construct marker pose from detection
                quat = entry.pose.pose.orientation
                bTm = tf.transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
                pos = entry.pose.pose.position
                bTm[:3, 3] = pos.x, pos.y, pos.z
                # compute pose in base frame
                bTo = np.dot(bTm, mTo)
                # publish pose
                pos = bTo[:3, 3]
                quat = tf.transformations.quaternion_from_matrix(bTo)
                self._publisher.sendTransform(pos, quat,
                                              rospy.Time.now(), obj_name, entry.pose.header.frame_id)
                published_object_tfs.add(obj_name)

if __name__ == '__main__':
    rospy.init_node('ObjectTFPublisher')
    object_names = rospy.get_param('~object_names')
    model_path = rospy.get_param('~model_path')
    rate = rospy.get_param('~tf_publisher_rate', 100)
    tag_topic = rospy.get_param('~tag_topic')
    publisher = ObjectTFPublisher(object_names, model_path, tag_topic)
    rospy.spin()