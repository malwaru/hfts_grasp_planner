#! /usr/bin/python
import argparse
import numpy as np
# from mayavi.core.api import
# f
from traits.api import Instance, Range, HasTraits, on_trait_change, Int, Property
from mayavi.core.ui.api import MlabSceneModel, SceneEditor, MayaviScene
from mayavi.core.api import PipelineBase
import traitsui.api
import os.path
import IPython


class MGRoadmapVisualizer(HasTraits):
    min_log_length = Int(0)
    # max_log_length = Property()
    max_log_length = Int(1)
    log_slide_range = Range(low='min_log_length', high='max_log_length')
    scene = Instance(MlabSceneModel, ())
    rm_vertices_plot = Instance(PipelineBase)
    valid_checked_vertices_plot = Instance(PipelineBase)
    invalid_checked_vertices_plot = Instance(PipelineBase)
    valid_checked_edges_plot = Instance(PipelineBase)
    invalid_checked_edges_plot = Instance(PipelineBase)

    view = traitsui.api.View(
        traitsui.api.Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                          height=400, width=400, show_label=False),
        traitsui.api.Item('_'),
        # TODO according to the documentation it should be possible to provide an editor here: editor=RangeEditor(mode='xslider'), it doesn't work though
        traitsui.api.Item('log_slide_range', label='LogStep'),
        traitsui.api.Item('max_log_length', label='Max log steps', style='readonly'),
        resizable=True)

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
        super(MGRoadmapVisualizer, self).__init__()
        # stores positions of all roadmap vertices, shape: #vertices, 2
        self._vertices = None
        self._last_rm_update = None  # last time the roadmap file was loaded
        self._grasp_images = {}
        self._roadmap_filename = roadmap_file
        self._log_filename = log_file
        self._last_log_update = None  # last time the logfile was loaded
        # parsed logs: contains tuples ((checked_node_id, grasp_id, state), (checked_edge, grasp_id, state));
        # grasp_id is None if base was only checked; state = 1 if valid, 0 if invalid
        self._logs = []
        self._grasp_folder = grasp_folder
        self._load_grasp_images()
        self._synch_roadmap()
        self._synch_logs()
        self._img_actors = {}
        self.grasp_distance = grasp_distance

    # def _get_max_log_length(self):
    #     return len(self._logs)

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

    def _synch_logs(self):
        def parse_log_entry(log_line):
            line_args = log_line.split(',')
            event_type = line_args[0]
            if (event_type == "VAL_BASE"):
                # base evaluation of a vertex
                vid = int(line_args[1])
                bvalid = bool(int(line_args[2]))
                return ((vid, None, bvalid), ())
                # print vid, bvalid
            elif (event_type == "VAL_GRASP"):
                # evaluation of a vertex for a certain grasp
                vid = int(line_args[1])
                gid = int(line_args[2])
                bvalid = bool(int(line_args[3]))
                return ((vid, gid, bvalid), ())
            elif (event_type == "EDGE_COST"):
                # base evaluation of an edge
                vid1 = int(line_args[1])
                vid2 = int(line_args[2])
                cost = float(line_args[3])
                return ((), (vid1, vid2, None, cost))
            elif (event_type == "EDGE_COST_GRASP"):
                # evaluation of an edge for a certain grasp
                vid1 = int(line_args[1])
                vid2 = int(line_args[2])
                gid = int(line_args[3])
                cost = float(line_args[4])
                return ((), (vid1, vid2, gid, cost))
        # check if there are new logs
        if self._last_log_update != os.path.getmtime(self._log_filename):
            self._last_log_update = os.path.getmtime(self._log_filename)
            # reload log file
            with open(self._log_filename, 'r') as log_file:
                self._logs = map(parse_log_entry, log_file.readlines())
            self.trait_set(max_log_length=len(self._logs))

    def _load_grasp_images(self):
        """
            Load grasp images from self._grasp_folder.
        """
        for fname in os.listdir(self._grasp_folder):
            gid = int(os.path.basename(fname).split('.')[0])
            image_data = np.load(self._grasp_folder + '/' + fname)
            self._grasp_images[gid] = np.clip(image_data, 0.0, np.max(image_data))

    def _create_base_plot(self):
        # create image views
        for (gid, grasp_image) in self._grasp_images.iteritems():
            grasp_z = float(gid * self.grasp_distance)
            # draw grasp cost space
            img_actor = self.scene.mlab.imshow(grasp_image, extent=[0, grasp_image.shape[0], 0,
                                                                    grasp_image.shape[1], grasp_z, grasp_z],
                                               interpolate=False)
            # img_actor.actor.position[2] = grasp_z
            self._img_actors[gid] = img_actor
            # render roadmap for this grasp
            self.rm_vertices_plot = self.scene.mlab.plot3d(self._vertices[:, 1], self._vertices[:, 0],
                                                           np.repeat(grasp_z + 0.1, self._vertices.shape[0]),
                                                           color=(0, 0, 0), representation='points')

    @on_trait_change('log_slide_range,scene.activated')
    def update_plot(self):
        """
            Issue an update of the visualization of the exploration state of the roadmap.
        """
        self._synch_roadmap()
        self._synch_logs()
        if self.rm_vertices_plot is None or self._img_actors is None:
            self._create_base_plot()
        # self.max_log_length = len(self._logs)
        # idx = float(self.log_slide_range) / 100.0 * len(self._logs)
        idx = self.log_slide_range
        # extract checked nodes (valid, invalid)
        nxs, nys, nzs = [[], []], [[], []], [[], []]
        # and checked edges (invalid valid)
        exs, eys, ezs, us, vs = [[], []], [[], []], [[], []], [[], []], [[], []]
        for i in xrange(int(idx)):
            node_check, edge_check = self._logs[i]
            if len(node_check):
                vid, gid, valid = node_check
                gid = 0 if gid is None else gid + 1
                nxs[valid].append(self._vertices[vid, 1])
                nys[valid].append(self._vertices[vid, 0])
                nzs[valid].append(gid * self.grasp_distance)
            if len(edge_check):
                vid1, vid2, gid, cost = edge_check
                valid = not np.isinf(cost)
                gid = 0 if gid is None else gid + 1
                exs[valid].append(self._vertices[vid1, 1])
                eys[valid].append(self._vertices[vid1, 0])
                ezs[valid].append(gid * self.grasp_distance)
                edge_dir = self._vertices[vid2] - self._vertices[vid1]
                us[valid].append(edge_dir[1])
                vs[valid].append(edge_dir[0])
        # render valid vertices
        if len(nxs[1]):
            if self.valid_checked_vertices_plot is None:
                self.valid_checked_vertices_plot = self.scene.mlab.points3d(
                    nxs[1], nys[1], nzs[1], color=(0, 1, 0), scale_factor=10.0, reset_zoom=False)
            else:
                self.valid_checked_vertices_plot.mlab_source.reset(x=nxs[1], y=nys[1], z=nzs[1])
        # render invalid vertices
        if len(nxs[0]):
            if self.invalid_checked_vertices_plot is None:
                self.invalid_checked_vertices_plot = self.scene.mlab.points3d(
                    nxs[0], nys[0], nzs[0], color=(1, 0, 0), scale_factor=10.0, reset_zoom=False)
            else:
                self.invalid_checked_vertices_plot.mlab_source.reset(x=nxs[0], y=nys[0], z=nzs[0])
        # render valid edges
        if len(exs[1]):
            if self.valid_checked_edges_plot is None:
                self.valid_checked_edges_plot = self.scene.mlab.quiver3d(
                    exs[1], eys[1], ezs[1], us[1], vs[1], np.zeros_like(us[1]), scale_factor=1, mode='2ddash', color=(0, 0.4, 0), reset_zoom=False)
            else:
                self.valid_checked_edges_plot.mlab_source.reset(
                    x=exs[1], y=eys[1], z=ezs[1], u=us[1], v=vs[1], w=np.zeros_like(us[1]))
        # TODO hide edges when there are none to show
        # render invalid edges
        if len(exs[0]):
            if self.invalid_checked_edges_plot is None:
                self.invalid_checked_edges_plot = self.scene.mlab.quiver3d(
                    exs[0], eys[0], ezs[0], us[0], vs[0], np.zeros_like(us[0]), scale_factor=1, mode='2ddash', color=(0.4, 0, 0), reset_zoom=False)
            else:
                self.invalid_checked_edges_plot.mlab_source.reset(
                    x=exs[0], y=eys[0], z=ezs[0], u=us[0], v=vs[0], w=np.zeros_like(us[0]))
        # TODO hide edges when there are none to show


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize roadmap exploration for multi-grasp configuration spaces. ' +
        'In case of visualization issues, try running with Qt4: 1. install python-qt4, 2. set environment variables ' +
        'ETS_TOOLKIT=qt4; QT_API=pyqt')
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
    viewer.configure_traits()
    # IPython.embed()
