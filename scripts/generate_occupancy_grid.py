#! /usr/bin/python
import openravepy as orpy
import argparse
import hfts_grasp_planner.sdf.grid
import hfts_grasp_planner.sdf.occupancy
import numpy as np


class GrowableAABB(object):
    def __init__(self, min_point, max_point):
        self._min_point = min_point
        self._max_point = max_point

    def union(self, other):
        """
            Join this AABB with the given one.
            ---------
            Arguments
            ---------
            other, either orpy.AABB or GrowableAABB - other AABB to unify this one with
        """
        if type(other) == orpy.AABB:
            self._min_point = np.min((self._min_point, other.pos() - other.extents()), axis=0)
            self._max_point = np.max((self._max_point, other.pos() + other.extents()), axis=0)
        else:
            self._min_point = np.min((self._min_point, other._min_point), axis=0)
            self._max_point = np.min((self._max_point, other._max_point), axis=0)

    def to_single_array(self):
        return np.array([self._min_point, self._max_point]).flatten()


def compute_aabb(env):
    aabb = GrowableAABB(np.zeros(3), np.zeros(3))
    for body in env.GetBodies():
        baabb = body.ComputeAABB()
        aabb.union(baabb)
    return aabb.to_single_array()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an occupancy grid of the given OpenRAVE environment and store it to disk")
    parser.add_argument('env_file', type=str, help='Path to an OpenRAVE environment')
    parser.add_argument('output_file', type=str, help='Path where to store occupancy grid')
    parser.add_argument('--aabb', type=float, nargs=6, required=False, metavar=("x_min", "y_min", "z_min", "x_max", "y_max", "z_max"),
                        help='Axis aligned bounding box specified by min point and max point')
    parser.add_argument('--cell_size', type=float, default=0.02, help="Cell size in meters")
    parser.add_argument('--disable_bodies', nargs='+', type=str, help="Body names to exclude from occupancy grid")
    args = parser.parse_args()
    env = orpy.Environment()
    env.Load(args.env_file)
    if args.aabb:
        workspace_aabb = np.array(args.aabb)
    else:
        workspace_aabb = compute_aabb(env)
    if args.disable_bodies:
        for body_name in args.disable_bodies:
            body = env.GetKinBody(body_name)
            if body:
                body.Enable(False)
                print "Disabled body ", body_name
            else:
                print "Could not find body with name ", body_name
    grid = hfts_grasp_planner.sdf.grid.VoxelGrid(workspace_aabb, args.cell_size)
    builder = hfts_grasp_planner.sdf.occupancy.OccupancyGridBuilder(env, args.cell_size)
    builder.compute_grid(grid)
    grid.save(args.output_file)
