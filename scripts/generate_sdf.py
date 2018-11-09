#! /usr/bin/python
import openravepy as orpy
import argparse
import hfts_grasp_planner.sdf.core
import hfts_grasp_planner.sdf.grid
import numpy as np
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a signed distance field from an occupancy grid")
    parser.add_argument('grid_file', type=str, help='Path to file storing occupancy grid')
    parser.add_argument('output_file', type=str, help='Path to where to store signed distance field')
    args = parser.parse_args()
    print "Loading occupancy grid"
    occ_grid = hfts_grasp_planner.sdf.grid.VoxelGrid.load(args.grid_file, True)
    print "Computing distance field"
    distance_grid = hfts_grasp_planner.sdf.core.SDFBuilder.compute_sdf(occ_grid)
    sdf = hfts_grasp_planner.sdf.core.SDF(distance_grid)
    print "Saving field"
    sdf.save(args.output_file)
    print "Done."
    sys.exit(0)
