#! /usr/bin/python
import argparse
import numpy as np
# from mayavi.core.api import
# f
from traits.api import Instance, Range, HasTraits, on_trait_change, Int, Property, String
from mayavi.core.ui.api import MlabSceneModel, SceneEditor, MayaviScene
from mayavi.core.api import PipelineBase
import mayavi.mlab as mlab
import traitsui.api
import os.path
import IPython


class MGRoadmapVisualizer(HasTraits):
    min_log_length = Int(0)
    # max_log_length = Property()
    max_log_length = Int(1)
    last_log_message = String("NOT READ YET")
    log_slide_range = Range(low='min_log_length', high='max_log_length')
    scene = Instance(MlabSceneModel, ())
    rm_vertices_plot = Instance(PipelineBase)
    valid_checked_vertices_plot = Instance(PipelineBase)
    invalid_checked_vertices_plot = Instance(PipelineBase)
    valid_checked_edges_plot = Instance(PipelineBase)
    invalid_checked_edges_plot = Instance(PipelineBase)
    num_extensions_plot = Instance(PipelineBase)
    solution_plot = Instance(PipelineBase)

    event_names = {
        'EdgeValidity': ['EDGE_COST', 'EDGE_COST_GRASP'],
        'VertexValidity': ['VAL_BASE', 'VAL_GRASP'],
        'VertexExpansion': ['BASE_EXPANSION', 'EXPANSION'],
        'Solution': ['SOLUTION_EDGE']
    }

    VERTEX_SIZE = 15.0
    EDGE_WIDTH = 5.0
    SOLUTION_EDGE_WIDTH = 15.0
    SOLUTION_OFFSET = 1.0

    view = traitsui.api.View(
        traitsui.api.Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=400, width=400,
                          show_label=False),
        traitsui.api.Item('_'),
        # TODO according to the documentation it should be possible to provide an editor here: editor=RangeEditor(mode='xslider'), it doesn't work though
        traitsui.api.Item('log_slide_range', label='LogStep'),
        traitsui.api.Item('max_log_length', label='Max log steps', style='readonly'),
        traitsui.api.Item('last_log_message', label='Last log message', style='readonly'),
        resizable=True)

    def __init__(self, roadmap_file, grasp_folder, log_file, max_num_visits, hide_list, grasp_distance=400):
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
            max_num_visits : int
                the maximal number of visits to expect for a node. This is used to normalize the color of the visit indicator.
            hide_list : list of string
                list of events to hide
        """
        super(MGRoadmapVisualizer, self).__init__()
        # stores positions of all roadmap vertices, shape: #vertices, 2
        self._vertices = None
        self._last_rm_update = None  # last time the roadmap file was loaded
        self._grasp_images = {}
        self._roadmap_filename = roadmap_file
        self._log_filename = log_file
        self._last_log_update = None  # last time the logfile was loaded
        self._hide_list = hide_list
        self._hide_list.extend(["GOAL_EXPANSION",
                                "GOAL_EXPENSION"])  # there is nothing to visualize for goal expansions
        # parsed logs: contains tuples ((checked_node_id, grasp_id, state), (checked_edge, grasp_id, state));
        # grasp_id is None if base was only checked; state = 1 if valid, 0 if invalid
        self._logs = []
        self._grasp_folder = grasp_folder
        self._load_grasp_images()
        self._synch_roadmap()
        self._synch_logs()
        self._img_actors = {}
        self.grasp_distance = grasp_distance
        self._max_num_visits = max_num_visits
        self._auto_adapt_max_num_vists = self._max_num_visits == 0

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
        def filter_entry(log_line):
            # line_args = log_line.split(',')
            return not reduce(lambda x, y: x or y, [s in log_line for s in self._hide_list], False)

        def parse_log_entry(log_line):
            line_args = log_line.split(',')
            event_type = line_args[0]
            if (event_type == "VAL_BASE"):
                # base evaluation of a vertex
                vid = int(line_args[1])
                bvalid = bool(int(line_args[2]))
                return ((vid, None, bvalid), (), (), (), log_line)
                # print vid, bvalid
            elif (event_type == "VAL_GRASP"):
                # evaluation of a vertex for a certain grasp
                vid = int(line_args[1])
                gid = int(line_args[2])
                bvalid = bool(int(line_args[3]))
                return ((vid, gid, bvalid), (), (), (), log_line)
            elif (event_type == "EDGE_COST"):
                # base evaluation of an edge
                vid1 = int(line_args[1])
                vid2 = int(line_args[2])
                cost = float(line_args[3])
                return ((), (vid1, vid2, None, cost), (), (), log_line)
            elif (event_type == "EDGE_COST_GRASP"):
                # evaluation of an edge for a certain grasp
                vid1 = int(line_args[1])
                vid2 = int(line_args[2])
                gid = int(line_args[3])
                cost = float(line_args[4])
                return ((), (vid1, vid2, gid, cost), (), (), log_line)
            elif (event_type == "EXPANSION"):
                # expanding a node (iterating over its neighbors)
                vid = int(line_args[1])
                gid = int(line_args[2])
                return ((), (), (vid, gid), (), log_line)
            elif (event_type == "BASE_EXPANSION"):
                # expanding a node (iterating over its neighbors)
                vid = int(line_args[1])
                return ((), (), (vid, None), (), log_line)
            elif (event_type == "SOLUTION_EDGE"):
                vid1 = int(line_args[1])
                vid2 = int(line_args[2])
                gid = int(line_args[3])
                return ((), (), (), (vid1, vid2, gid), log_line)
            else:
                raise ValueError("Unknown event type encountered %s" % event_type)

        # check if there are new logs
        if self._last_log_update != os.path.getmtime(self._log_filename):
            self._last_log_update = os.path.getmtime(self._log_filename)
            # reload log file
            with open(self._log_filename, 'r') as log_file:
                self._logs = map(parse_log_entry, filter(filter_entry, log_file.readlines()))
            self.trait_set(max_log_length=len(self._logs))

    def _load_grasp_images(self):
        """
            Load grasp images from self._grasp_folder.
        """
        for fname in os.listdir(self._grasp_folder):
            gid = int(os.path.basename(fname).split('.')[0])
            image_data = np.load(self._grasp_folder + '/' + fname)
            invalid_vals = np.isinf(image_data)
            valid_vals = np.logical_not(invalid_vals)
            image_data[invalid_vals] = np.nan  # np.max(image_data[valid_vals])
            self._grasp_images[gid] = image_data
            # np.clip(image_data, 0.0, np.max(image_data))

    def _create_base_plot(self):
        # create image views
        for (gid, grasp_image) in self._grasp_images.iteritems():
            grasp_z = float(gid * self.grasp_distance)
            # draw grasp cost space
            img_actor = self.scene.mlab.imshow(
                grasp_image,
                extent=[0, grasp_image.shape[0], 0, grasp_image.shape[1], grasp_z, grasp_z],
                interpolate=False)
            img_actor.module_manager.scalar_lut_manager.lut.nan_color = 0.2, 0.2, 0.2, 1
            img_actor.update_pipeline()
            # colormap=u'YlOrRd')
            # img_actor.actor.position[2] = grasp_z
            self._img_actors[gid] = img_actor
            # render roadmap for this grasp
            self.rm_vertices_plot = self.scene.mlab.plot3d(self._vertices[:, 1],
                                                           self._vertices[:, 0],
                                                           np.repeat(grasp_z + 0.1, self._vertices.shape[0]),
                                                           color=(0, 0, 0),
                                                           representation='points')

    @on_trait_change('log_slide_range,scene.activated')
    def update_plot(self):
        """
            Issue an update of the visualization of the exploration state of the roadmap.
        """
        # mlab.view(-90, 68.79285548404428, 3139.0443403849304, np.array([479.24520597, 500.59099056, 401.7819012]))
        self._synch_roadmap()
        self._synch_logs()
        if self.rm_vertices_plot is None or self._img_actors is None:
            self._create_base_plot()
        # self.max_log_length = len(self._logs)
        # idx = float(self.log_slide_range) / 100.0 * len(self._logs)
        idx = self.log_slide_range
        self.trait_set(last_log_message=self._logs[int(idx) - 1][-1].replace('\n', ''))
        # extract checked nodes (valid, invalid)
        nxs, nys, nzs = [[], []], [[], []], [[], []]
        # and checked edges (invalid valid)
        exs, eys, ezs, us, vs = [[], []], [[], []], [[], []], [[], []], [[], []]
        # solution edges
        solutions_edges = [[], [], [], [], []]  # xs, ys, zs, us, vs (u and v are directions)
        # map from (vid, layer_id) -> number of expansions
        num_expansions = {}
        for i in xrange(int(idx)):
            node_check, edge_check, node_expansion, sol_edge, _ = self._logs[i]
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
            if len(node_expansion):
                layer_idx = node_expansion[1] + 1 if node_expansion[1] is not None else 0
                key = (node_expansion[0], layer_idx)
                if key in num_expansions:
                    num_expansions[key] += 1
                else:
                    num_expansions[key] = 1
            if len(sol_edge):
                vid1, vid2, gid = sol_edge
                gid = 0 if gid is None else gid + 1
                solutions_edges[0].append(self._vertices[vid1, 1])
                solutions_edges[1].append(self._vertices[vid1, 0])
                solutions_edges[2].append(gid * self.grasp_distance + self.SOLUTION_OFFSET)
                edge_dir = self._vertices[vid2] - self._vertices[vid1]
                solutions_edges[3].append(edge_dir[1])
                solutions_edges[4].append(edge_dir[0])

        # render valid vertices
        if len(nxs[1]):
            if self.valid_checked_vertices_plot is None:
                self.valid_checked_vertices_plot = self.scene.mlab.points3d(nxs[1],
                                                                            nys[1],
                                                                            nzs[1],
                                                                            color=(0, 1, 0),
                                                                            scale_factor=self.VERTEX_SIZE,
                                                                            reset_zoom=False)
            else:
                self.valid_checked_vertices_plot.actor.visible = True
                self.valid_checked_vertices_plot.mlab_source.reset(x=nxs[1], y=nys[1], z=nzs[1])
        elif self.valid_checked_vertices_plot is not None:
            self.valid_checked_vertices_plot.actor.visible = False
        # render invalid vertices
        if len(nxs[0]):
            if self.invalid_checked_vertices_plot is None:
                self.invalid_checked_vertices_plot = self.scene.mlab.points3d(nxs[0],
                                                                              nys[0],
                                                                              nzs[0],
                                                                              color=(1, 0, 0),
                                                                              scale_factor=self.VERTEX_SIZE,
                                                                              reset_zoom=False)
            else:
                self.invalid_checked_vertices_plot.actor.visible = True
                self.invalid_checked_vertices_plot.mlab_source.reset(x=nxs[0], y=nys[0], z=nzs[0])
        elif self.invalid_checked_vertices_plot is not None:
            self.invalid_checked_vertices_plot.actor.visible = False
        # render valid edges
        if len(exs[1]):
            if self.valid_checked_edges_plot is None:
                self.valid_checked_edges_plot = self.scene.mlab.quiver3d(exs[1],
                                                                         eys[1],
                                                                         ezs[1],
                                                                         us[1],
                                                                         vs[1],
                                                                         np.zeros_like(us[1]),
                                                                         scale_factor=1,
                                                                         line_width=self.EDGE_WIDTH,
                                                                         mode='2ddash',
                                                                         color=(0, 0.4, 0),
                                                                         reset_zoom=False)
            else:
                self.valid_checked_edges_plot.actor.visible = True
                self.valid_checked_edges_plot.mlab_source.reset(x=exs[1],
                                                                y=eys[1],
                                                                z=ezs[1],
                                                                u=us[1],
                                                                v=vs[1],
                                                                w=np.zeros_like(us[1]))
        elif self.valid_checked_edges_plot is not None:
            self.valid_checked_edges_plot.actor.visible = False
        # render invalid edges
        if len(exs[0]):
            if self.invalid_checked_edges_plot is None:
                self.invalid_checked_edges_plot = self.scene.mlab.quiver3d(exs[0],
                                                                           eys[0],
                                                                           ezs[0],
                                                                           us[0],
                                                                           vs[0],
                                                                           np.zeros_like(us[0]),
                                                                           line_width=self.EDGE_WIDTH,
                                                                           scale_factor=1,
                                                                           mode='2ddash',
                                                                           color=(0.4, 0, 0),
                                                                           reset_zoom=False)
            else:
                self.invalid_checked_edges_plot.actor.visible = True
                self.invalid_checked_edges_plot.mlab_source.reset(x=exs[0],
                                                                  y=eys[0],
                                                                  z=ezs[0],
                                                                  u=us[0],
                                                                  v=vs[0],
                                                                  w=np.zeros_like(us[0]))
        elif self.invalid_checked_edges_plot is not None:
            self.invalid_checked_edges_plot.actor.visible = False
        # render number of expansions
        if len(num_expansions) > 0:
            xs, ys, zs, ss = [], [], [], []
            for key, num in num_expansions.iteritems():
                vid, layer_id = key
                xs.append(self._vertices[vid, 1])
                ys.append(self._vertices[vid, 0])
                zs.append(layer_id * self.grasp_distance)
                ss.append(num + 1)
            if self.num_extensions_plot is None:
                self.num_extensions_plot = self.scene.mlab.points3d(xs,
                                                                    ys,
                                                                    zs,
                                                                    ss,
                                                                    mode="2dcircle",
                                                                    reset_zoom=False,
                                                                    scale_mode='none',
                                                                    scale_factor=1.1 * self.VERTEX_SIZE,
                                                                    vmin=0.0,
                                                                    vmax=self._max_num_visits)
            else:
                self.num_extensions_plot.actor.visible = True
                self.num_extensions_plot.mlab_source.reset(x=xs, y=ys, z=zs, scalars=ss)
        # render solution
        if len(solutions_edges) > 0 and len(solutions_edges[0]) > 0:
            if self.solution_plot is None:
                self.solution_plot = self.scene.mlab.quiver3d(solutions_edges[0],
                                                              solutions_edges[1],
                                                              solutions_edges[2],
                                                              solutions_edges[3],
                                                              solutions_edges[4],
                                                              np.zeros(len(solutions_edges[3])),
                                                              scale_factor=1,
                                                              line_width=self.SOLUTION_EDGE_WIDTH,
                                                              mode='2ddash',
                                                              color=(0, 0, 0.5),
                                                              reset_zoom=False)
            else:
                self.solution_plot.actor.visible = True
                self.solution_plot.mlab_source.reset(x=solutions_edges[0],
                                                     y=solutions_edges[1],
                                                     z=solutions_edges[2],
                                                     u=solutions_edges[3],
                                                     v=solutions_edges[4],
                                                     w=np.zeros(len(solutions_edges[3])))
        elif self.solution_plot is not None:
            self.solution_plot.actor.visible = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize roadmap exploration for multi-grasp configuration spaces. ' +
        'In case of visualization issues, try running with Qt4: 1. install python-qt4, 2. set environment variables ' +
        'ETS_TOOLKIT=qt4; QT_API=pyqt')
    parser.add_argument('grasp_costs',
                        help='Path to a folder containing numpy arrays storing cost spaces for a 2d robot.' +
                        'The files in the folder should be named <id>.npy, where <id> is an integer grasp id.',
                        type=str)
    parser.add_argument('roadmap_file', help='Path to a single file containing the roadmap to visualize.', type=str)
    parser.add_argument('log_file', help='Path to a single file containing the evaluation logs to visualize.', type=str)
    parser.add_argument('--max_num_visits',
                        help='The maximal number of visits of a node used to normalize the color of the visit'
                        ' indicator.',
                        type=int,
                        default=100)
    parser.add_argument(
        '--hide',
        help='Provide a list of events to hide from the visualization. Options are: VertexValidity, EdgeValidity,'
        ' VertexExpansion, Solution',
        type=str,
        nargs='+')
    args = parser.parse_args()
    hide_list = []
    if args.hide:
        for hide_elem in args.hide:
            hide_list.extend(MGRoadmapVisualizer.event_names[hide_elem])

    viewer = MGRoadmapVisualizer(args.roadmap_file, args.grasp_costs, args.log_file, args.max_num_visits, hide_list)
    viewer.configure_traits()
    # IPython.embed()
