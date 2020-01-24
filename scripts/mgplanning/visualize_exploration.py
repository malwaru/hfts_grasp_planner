#! /usr/bin/python
import argparse
import numpy as np
# import scipy.ndimage as scimg
from scipy import misc
from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
from mayavi import mlab
import os.path
import sys
import threading
import IPython


class MGRoadmapVisualizer(object):
    def __init__(self, roadmap_file, grasp_folder, log_file, grasp_distance=400):
        """
            Create a new MGRoadmapVisualizer.

            Arguments
            ---------
            roadmap_file : string
                path to roadmap file
            grasp_folder : string
                path to folder containing different grasp images.
            log_file : string
                path to log file to read evaluations of roadmap states from
        """
        self._showing_ui = False
        # stores positions of all roadmap vertices, shape: #vertices, 2
        self._vertices = None
        # stores for each vertex whether for which grasp it has been evaluated, shape #vertices, #grasps + 1
        # b, v0, v1, ..., vg; b - base check, vi - vertex check. It is vi = 0 if unchecked, vi < 0 if invalid, vi > 0 if valid
        self._vertices_state = None
        # self._edges_state = None
        self._last_rm_update = None
        self._grasp_images = {}
        self._roadmap_filename = roadmap_file
        self._log_filename = log_file
        self._log_line_counter = 0
        self._last_log_update = None
        self._logs = []
        self._grasp_folder = grasp_folder
        self._load_grasp_images()
        self._synch_roadmap()
        self._visualizer_thread = None
        self._img_actors = {}
        self.grasp_distance = grasp_distance

    def __del__(self):
        if self._visualizer_thread is not None:
            self._visualizer_thread.join()

    def _synch_roadmap(self):
        """
            Update the roadmap (self._vertices) to reflect the latest state of what is stored in self._roadmap_filename.
            Updates also self._last_rm_update.
        """
        # only update roadmap from file if file has changed
        if self._last_rm_update != os.path.getmtime(self._roadmap_filename):
            with open(self._roadmap_filename, 'r') as rfile:
                self._last_rm_update = os.path.getmtime(self._roadmap_filename)
                file_content = rfile.readlines()
                if len(file_content) == 0:
                    print "The roadmap file is empty. Nothing to show."
                    return
                # extract roadmap dimension
                dim = int(file_content[0].split(',')[1])
                if dim != 2:
                    print "Roadmap dimensions other than 2 are currently not supported"
                    return
                # create vertices
                # self._vertices = np.empty((len(file_content, dim + 1))

                def parse_vertex(s):
                    values = s.split(',')
                    return map(float, values[2:])
                self._vertices = np.array(map(parse_vertex, file_content))
                self._vertices_state = np.zeros((self._vertices.shape[0], len(self._grasp_images) + 1))

    def _load_grasp_images(self):
        """
            Load grasp images from self._grasp_folder.
        """
        for fname in os.listdir(self._grasp_folder):
            gid = int(os.path.basename(fname).split('.')[0])
            image_data = np.load(self._grasp_folder + '/' + fname)
            self._grasp_images[gid] = np.clip(image_data, 0.0, np.max(image_data))

    def show(self):
        """
            Show the visualizer.
        """
        if not self._showing_ui:
            # show grasps
            for (gid, grasp_image) in self._grasp_images.iteritems():
                grasp_z = float(gid * self.grasp_distance)
                # draw grasp cost space
                img_actor = mlab.imshow(grasp_image,
                                        extent=[0, grasp_image.shape[0], 0, grasp_image.shape[1], grasp_z, grasp_z],
                                        interpolate=False)
                # img_actor.actor.position[2] = grasp_z
                self._img_actors[gid] = img_actor
                # render roadmap for this grasp
                mlab.plot3d(self._vertices[:, 1], self._vertices[:, 0], np.repeat(grasp_z + 0.1, self._vertices.shape[0]),
                            color=(0, 0, 0), representation='points')
                self.explored_points = mlab.points3d([0], [0], [0], color=(1, 0, 0), scale_factor=10.0)
            self._showing_ui = True
            self._log_line_counter = 0
            self.animate_exploration_state()
            # self.fake_animation()
            mlab.show()
            # if os.fork() == 0:
            #     try:
            #         mlab.show()
            #     finally:
            #         os._exit(os.EX_OK)

    @mlab.animate(delay=10)
    def animate_exploration_state(self):  # , update_all=False):
        """
            Issue an update of the visualization of the exploration state of the roadmap.

            Arguments
            ---------
            # update_all : bool
            #     If True, update the exploration state until to the latest state within the log file.
            #     Otherwise, only update one step further.
        """
        log_line_counter = 0
        while True:
            # while self._log_line_counter < len(self._logs):
            # check if there are new logs
            if self._last_log_update != os.path.getmtime(self._log_filename):
                # reload log file
                with open(self._log_filename, 'r') as log_file:
                    self._logs = log_file.readlines()
            # update the exploration state
            if log_line_counter < len(self._logs):  # and (update_all or update_once):
                # update_once = False
                # read next line
                log_line = self._logs[log_line_counter]
                log_line_counter += 1
                line_args = log_line.split(',')
                event_type = line_args[0]
                if (event_type == "VAL_BASE"):
                    # base evaluation of a vertex
                    vid = int(line_args[1])
                    bvalid = bool(line_args[2])
                    self._vertices_state[vid, 0] = 1.0 if bvalid else -1.0
                    # print vid, bvalid
                elif (event_type == "VAL_GRASP"):
                    # evaluation of a vertex for a certain grasp
                    vid = int(line_args[1])
                    gid = int(line_args[2])
                    bvalid = bool(line_args[3])
                    self._vertices_state[vid, gid + 1] = 1.0 if bvalid else -1.0
                    # print vid, gid, bvalid
                elif (event_type == "EDGE_COST"):
                    # base evaluation of an edge
                    # TODO
                    pass
                elif (event_type == "EDGE_COST_GRASP"):
                    # evaluation of an edge for a certain grasp
                    # TODO
                    pass
            # TODO update rendering
            # print np.where(self._vertices_state[:, 0] > 0)[0]
            active_nodes = np.where(self._vertices_state > 0)
            if active_nodes[0].shape[0] > 0:
                xs = self._vertices[active_nodes[0], 1]
                ys = self._vertices[active_nodes[0], 0]
                zs = active_nodes[1] * self.grasp_distance
                self.explored_points.mlab_source.reset(x=xs, y=ys, z=zs)
                # print active_nodes
                # print xs, ys, zs
            print "Animation progress: ", log_line_counter, len(self._logs)
            yield


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'grasp_costs', help='Path to a folder containing numpy arrays storing cost spaces for a 2d robot.' +
        'The files in the folder should be named <id>.npy, where <id> is an integer grasp id.',
        type=str)
    parser.add_argument(
        'roadmap_file', help='Path to a single file containing the roadmap to visualize.', type=str)
    parser.add_argument(
        'log_file', help='Path to a single file containing the evaluation logs to visualize.', type=str)
    args = parser.parse_args()
    viewer = MGRoadmapVisualizer(args.roadmap_file, args.grasp_costs, args.log_file)
    viewer.show()
    IPython.embed()
