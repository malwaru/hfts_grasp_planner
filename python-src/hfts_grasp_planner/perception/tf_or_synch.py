#!/usr/bin/env python

# Import required Python code.
import roslib, rospy, tf, tf_conversions, threading, numpy
import os

class TFSynchronizer:
    def __init__(self, or_env, object_names, data_folder, tf_base_frame, wTb=None):
        """
            Create a new TFSynchronizer for the given OpenRAVE environment.
            ---------
            Arguments
            ---------
            or_env, orpy.Environment
            object_name, list of strings - object names to query poses from tf for
            data_folder, string - path to folder containing subfolders with the object name,
                which in turn, contain a kinbody file for the object
            tf_base_frame, string - the name of the frame in which the OpenRAVE environment
                is set
            wTb(optional), np.array (4, 4) - optionally a transformation matrix to apply
                on each pose after querying from tf - default is np.eye(4)
        """
        self._env = or_env
        self._object_names = object_names
        self._data_folder = data_folder
        self._is_running = False
        self._listener = None
        self._tf_base_frame = tf_base_frame
        self._thread = None
        if wTb is None:
            wTb = numpy.eye(4)
        self._transform = wTb
        self.ignore_set = set()

    def __del__(self):
        self.end()

    def start(self):
        if self._is_running:
            rospy.logwarn('Perception is already running')
            return
        # rospy.init_node('TFSynchr', anonymous = True)
        self._listener = tf.TransformListener()
        self._is_running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        rate = rospy.Rate(80.0)
        while not rospy.is_shutdown() and self._is_running:
            for obj in self._object_names:
                if obj in self.ignore_set:
                    continue
                try:
                    (trans, rot) = self._listener.lookupTransform(self._tf_base_frame, obj, rospy.Time(0))
                    with self._env:
                        body = self._env.GetKinBody(obj)
                        if body is None:
                            # check whether we have a kinbody file that we can load
                            folder_path = self._data_folder + '/' + obj + '/'
                            kinbody_files = filter(lambda x: 'kinbody.xml' in x, os.listdir(folder_path))
                            if len(kinbody_files) > 0:
                                for body_file in kinbody_files:
                                    bloaded = self._env.Load(folder_path + body_file)
                                    if bloaded:
                                        body = self._env.GetKinBody(obj)
                                        break
                            if body is None:
                                continue
                        # Compute transform
                        matrix = tf_conversions.transformations.quaternion_matrix([rot[0], rot[1], rot[2], rot[3]])
                        matrix[:3, 3] = trans
                        matrix = numpy.dot(self._transform, matrix)
                        body.SetTransform(matrix)
                        # self.drawer.drawPose(self._orEnv, matrix)

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            rate.sleep()

    def enableObject(self, obj_name, enable = True):
        if enable:
            if obj_name in self.ignore_set:
                self.ignore_set.remove(obj_name)
        else:
            if obj_name not in self.ignore_set:
                self.ignore_set.add(obj_name)

    def end(self):
        if self._is_running:
            self._is_running = False
            self._thread.join()

# Main function.
if __name__ == '__main__':
    import openravepy, IPython
    rospy.init_node("TestORBridge")
    env = openravepy.Environment()
    env.SetViewer('qtcoin')
    object_names = ['elmers_glue', 'sugar_box', 'mustard', 'cabinet', 'cracker_box']
    data_path = '/home/joshua/projects/placement_ws/src/hfts_grasp_planner/models/objects/'
    perception = TFSynchronizer(env, object_names, data_path, 'kinect2_rgb_optical_frame')
    perception.start()
    IPython.embed()
    perception.end()



