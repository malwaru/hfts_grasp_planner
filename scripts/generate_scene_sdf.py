#! /usr/bin/python
import openravepy as orpy
import argparse
import hfts_grasp_planner.sdf.core as sdf_core
import hfts_grasp_planner.sdf.occupancy
import numpy as np
import sys


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
        description="Create a scene signed distance field for the given environment.")
    parser.add_argument('env_file', type=str, help='Path to an OpenRAVE environment')
    parser.add_argument('output_file', type=str, help='Path where to store scene signed distance field')
    parser.add_argument('--aabb', type=float, nargs=6, required=False, metavar=("x_min", "y_min", "z_min", "x_max", "y_max", "z_max"),
                        help='Axis aligned bounding box specified by min point and max point')
    parser.add_argument('--cell_size', type=float, default=0.02, help="Cell size in meters")
    parser.add_argument('--mcell_size', type=float, required=False,
                        help="If provided, cell size for sdfs of movable bodies. If not provided cell_size is used.")
    parser.add_argument('--movable_bodies', nargs='+', type=str,
                        help="Body names to flag as movable (i.e. they get their own distance fields)")
    parser.add_argument('--exclude_bodies', nargs='+', type=str, help="Body names to exclude from sdf")
    args = parser.parse_args()
    env = orpy.Environment()
    env.Load(args.env_file)
    if args.aabb:
        workspace_aabb = np.array(args.aabb)
    else:
        workspace_aabb = compute_aabb(env)
    excluded_bodies = []
    if args.exclude_bodies:
        excluded_bodies = args.exclude_bodies
    print "Will exclude bodies:", str(excluded_bodies)
    movable_bodies = []
    if args.movable_bodies:
        movable_bodies = args.movable_bodies
    print "Will generate separate sdfs for:", str(movable_bodies)
    mcell_size = args.cell_size
    if args.mcell_size:
        mcell_size = args.mcell_size
    scene_sdf = sdf_core.SceneSDF(env, movable_bodies, excluded_bodies=excluded_bodies)
    print "Generating SceneSDF"
    scene_sdf.create_sdf(workspace_aabb, static_resolution=args.cell_size, moveable_resolution=mcell_size)
    print "Saving"
    scene_sdf.save(args.output_file)
    print "Done"
    sys.exit(0)
