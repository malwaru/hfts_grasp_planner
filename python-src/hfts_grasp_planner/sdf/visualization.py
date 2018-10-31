import numpy as np
import time
from itertools import izip
# import matplotlib.pyplot as plt

# This registers the 3D projection, but it otherwise unneeded
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class ORVoxelGridVisualization(object):
    """
        This class allows to visualize a voxel grid using an OpenRAVE environment.
    """

    def __init__(self, or_env, voxel_grid):
        """
            Creates a new visualization of a voxel grid using openrave.
        """
        self._env = or_env
        self._voxel_grid = voxel_grid
        self._handles = []

    def update(self, min_sat_value=None, max_sat_value=None, style=0, color_fn=None):
        """
            Updates this visualization to reflect the latest state of the underlying voxel grid.
            The voxels are colored according to their values. By default the color of a voxel
            is computed using linear interpolation between two colors min_color and max_color.
            The voxel with maximum value is colored with max_color and the voxel with minimum value
            is colored with min_color. This behaviour can be changed by providing min_sat_value
            and max_sat_value. If these values are provided, any cell with value <= min_sat_value is
            colored min_color and any cell with value >= max_sat_value is colored with max_color.
            Cells with values in range (min_sat_value, max_sat_value) are colored
            using linear interpolation.

            @param min_sat_value (optional) - minimal saturation value
            @param max_sat_value (optional) - maximal saturation value
            @param style (optional) - if 0, renders cells using 2d sprites, if 1, renders cells using 3d balls, 
                                      if 2, renders cells as boxes
                                      WARNING: Rendering many balls(cells) will crash OpenRAVE
        """
        self._handles = []
        values = [x.get_value() for x in self._voxel_grid]
        if min_sat_value is None:
            min_sat_value = min(values)
        if max_sat_value is None:
            max_sat_value = max(values)

        blue_color = np.array([0.0, 0.0, 1.0, 0.05])
        red_color = np.array([1.0, 0.0, 0.0, 0.05])
        positions = np.array([cell.get_position() for cell in self._voxel_grid])

        def compute_color(value):
            """
                Computes the color for the given value
            """
            rel_value = np.clip((value - min_sat_value) / (max_sat_value - min_sat_value), 0.0, 1.0)
            return (1.0 - rel_value) * red_color + rel_value * blue_color
        if color_fn is None:
            color_fn = compute_color
        colors = np.array([color_fn(v) for v in values])
        # TODO we should read the conversion from pixels to workspace size from somwhere
        # and convert true cell size to it
        if style == 0:
            handle = self._env.plot3(positions, 20, colors)  # size is in pixel
            self._handles.append(handle)
        elif style == 1:
            handle = self._env.plot3(positions, self._voxel_grid.get_cell_size / 2.0, colors, 1)
            self._handles.append(handle)
        elif style == 2:
            extents = 3 * [self._voxel_grid.get_cell_size() / 2.0]
            for pos, color in izip(positions, colors):
                handle = self._env.drawbox(pos, extents, color)
                self._handles.append(handle)
        else:
            raise ValueError("Invalid style: " + str(style))

    def clear(self):
        """
            Clear visualization
        """
        self._handles = []


# class MatplotLibGridVisualization(object):
#     """
#         Visualize a Voxel grid using matplotlib.
#     """

#     def __init__(self):
#         self._fig = plt.figure()
#         self._ax = self._fig.gca(projection='3d')

#     def visualize_bool_grid(self, grid):
#         """
#             Visualize grid of bool values.
#             ---------
#             Arguments
#             ---------
#             grid, VoxelGrid of cell type bool
#         """
#         assert(grid.get_type() == bool)
#         xx, yy, zz = grid.get_cell_positions(b_center=False)
#         grid_data = grid.get_raw_data()
#         self._ax.voxels(xx, yy, zz, grid_data)
#         self._fig.show()

    # def visualize_float_grid(self, grid):

class ORSDFVisualization(object):
    """
        This class allows to visualize an SDF using an OpenRAVE environment.
    """

    def __init__(self, or_env):
        """
            Creates a new visualization of an SDF using openrave.
        """
        self._env = or_env
        self._handles = []

    def visualize(self, sdf, volume, resolution=0.1, min_sat_value=None, max_sat_value=None, alpha=0.1,
                  style='sprites'):
        """
            Samples the given sdf within the specified volume and visualizes the data.
            @param sdf - the signed distance field to visualize
            @param volume - the workspace volume in which the sdf should be visualized
            @param resolution (optional) - the resolution at which the sdf should be sampled.
            @param min_sat_value (optional) - all points with distance smaller than this will have the same color
            @param max_sat_value (optional) - all point with distance larger than this will have the same color
            @param alpha (optional) - alpha value for colors
            @param style (optional) - if sprites, renders cells using 2d sprites, if balls, renders cells using 3d balls,
                                     if boxes, renders cells using boxes.
                                      WARNING: Rendering many balls(cells) will crash OpenRAVE
        """
        # first sample values
        if type(volume) is tuple:
            volume = np.concatenate(volume)
        num_samples = (volume[3:] - volume[:3]) / resolution
        start_time = time.time()
        positions = np.array([np.array([x, y, z]) for x in np.linspace(volume[0], volume[3], num_samples[0])
                              for y in np.linspace(volume[1], volume[4], num_samples[1])
                              for z in np.linspace(volume[2], volume[5], num_samples[2])])
        print ('Computation of positions took %f s' % (time.time() - start_time))
        start_time = time.time()
        # values = [sdf.get_distance(pos) for pos in positions]
        values = sdf.get_distances(positions)
        print ('Computation of distances took %f s' % (time.time() - start_time))
        # compute min and max
        # draw
        start_time = time.time()
        if min_sat_value is None:
            min_sat_value = min(values)
        if max_sat_value is None:
            max_sat_value = max(values)

        blue_color = np.array([0.0, 0.0, 1.0, alpha])
        red_color = np.array([1.0, 0.0, 0.0, alpha])
        zero_color = np.array([0.0, 0.0, 0.0, alpha])

        def compute_color(value):
            """
                Computes the color for the given value
            """
            # rel_value = np.clip((value - min_sat_value) / (max_sat_value - min_sat_value), 0.0, 1.0)
            # return (1.0 - rel_value) * red_color + rel_value * blue_color
            base_color = blue_color if value > 0.0 else red_color
            sat_value = max_sat_value if value > 0.0 else min_sat_value
            rel_value = np.clip(value / sat_value, 0.0, 1.0)
            return rel_value * base_color + (1.0 - rel_value) * zero_color

        colors = np.array([compute_color(v) for v in values])
        if style == 'sprites':
            handle = self._env.plot3(positions[:, :3], 10, colors)  # size is in pixel
            self._handles.append(handle)
        elif style == 'balls':
            # Instead we can also render balls, but this can easily crash OpenRAVE if done for many cells
            handle = self._env.plot3(positions[:, :3], resolution / 2.0, colors, 1)
            self._handles.append(handle)
        elif style == 'boxes':
            extents = [resolution / 2.0, resolution / 2.0, resolution / 2.0]
            for pos, color in izip(positions, colors):
                handle = self._env.drawbox(pos, extents, color)
                self._handles.append(handle)
        else:
            raise ValueError('Could not render sdf. Unknown style %s' % style)
        print ('Rendering took %f s' % (time.time() - start_time))
        # Alternatively, the following code would render the sdf using meshes, but this also kills openrave
        # colors = np.reshape([12 * [compute_color(v)] for v in values], (12 * values.shape[0], 4))
        # box_extents = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]) * resolution
        # vertices = np.zeros((8 * positions.shape[0], 3))
        # indices = np.zeros((12 * positions.shape[0], 3), dtype=np.int64)
        # box_vertices, box_indices = ComputeBoxMesh(box_extents)
        # for pos_idx in xrange(len(positions)):
        #     vertices[pos_idx * 8 : (pos_idx + 1) * 8] = box_vertices + positions[pos_idx, :3]
        #     indices[pos_idx * 12 : (pos_idx + 1) * 12] = box_indices + pos_idx * 12
        # handle = self._env.drawtrimesh(points=vertices,
        #                                indices=indices)
        #                             #    colors=colors)

    def clear(self):
        """
            Clear visualization
        """
        self._handles = []


# def visualize_grid(grid, min_sat_value=None, max_sat_value=None):
#     """
#         Visualize the given grid using mlab.
#         @param grid - the grid to visualize
#         @param min_sat_value (optional) - all points with distance smaller than this will have the same color
#         @param max_sat_value (optional) - all point with distance larger than this will have the same color
#     """
#     if min_sat_value is None:
#         min_sat_value = np.min(grid.get_raw_data())
#     if max_sat_value is None:
#         max_sat_value = np.max(grid.get_raw_data())
#     mlab.pipeline.volume(mlab.pipeline.scalar_field(grid.get_raw_data()), vmin=min_sat_value, vmax=max_sat_value)

# def clear_visualization():
#     """
#         Clear visualization.
#     """
#     mlab.clf()
