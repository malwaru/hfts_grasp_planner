import os
import openravepy as orpy
import numpy as np
import hfts_grasp_planner.sdf.kinbody as kinbody_core
import hfts_grasp_planner.placement2.clearance as clearance_module
import hfts_grasp_planner.sdf.grid as grid_module
import hfts_grasp_planner.sdf.core as sdf_core
import IPython


def colorize_object(body, val, min_val, max_val):
    min_color = np.array([1.0, 0.0, 0.0, 1.0])
    max_color = np.array([0.0, 1.0, 0.0, 1.0])
    t = (val - min_val) / (max_val - min_val)
    color = t * min_color + (1.0 - t) * max_color
    for link in body.GetLinks():
        for geom in link.GetGeometries():
            geom.SetDiffuseColor(color)


def compute_clearance(val_range, clearance_map, link_grid):
    val = link_grid.sum(clearance_map)
    val_range[0] = np.min([val, val_range[0]])
    val_range[1] = np.max([val, val_range[1]])
    # colorize_object(link_grid.get_link().GetParent(), val, val_range[0], val_range[1])
    print val


if __name__ == "__main__":
    base_path = os.path.dirname(__file__) + '/../../'
    sdf_file = base_path + 'data/sdfs/placement_exp_0_low_res'
    sdf = sdf_core.SDF.load(sdf_file)
    occ_file = base_path + 'data/occupancy_grids/placement_exp_0_low_res'
    occ = grid_module.VoxelGrid.load(occ_file)
    clearance_map = clearance_module.compute_clearance_map(occ, sdf.get_grid())
    env = orpy.Environment()
    env.Load(base_path + 'data/environments/placement_exp_0.xml')
    env.SetViewer('qtcoin')
    body = env.GetKinBody('crayola')
    link = body.GetLinks()[0]
    link_grid = kinbody_core.RigidBodyOccupancyGrid(0.005, link)
    val_range = np.array([0.0, 0.0])
    compute_clearance(val_range, clearance_map, link_grid)
    IPython.embed()
