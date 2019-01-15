#! /usr/bin/python
import argparse
import hfts_grasp_planner.sdf.core
import hfts_grasp_planner.sdf.grid
import numpy as np
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a gradient field from an sdf")
    parser.add_argument('sdf_file', type=str, help='Path to signed distance field')
    parser.add_argument('output_file', type=str, help='Path to where to store gradient field')
    args = parser.parse_args()
    print "Loading sdf"
    sdf = hfts_grasp_planner.sdf.core.SDF.load(args.sdf_file)
    print "Computing gradients"
    gradient_field = hfts_grasp_planner.sdf.core.SDFBuilder.compute_gradient_field(sdf)
    print "Saving field"
    gradient_field.save(args.output_file)
    print "Done."
    sys.exit(0)
