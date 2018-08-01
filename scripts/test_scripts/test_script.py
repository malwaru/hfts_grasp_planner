#! /usr/bin/python
# import hfts_grasp_planner.sdf.core as sdf_module
# import hfts_grasp_planner.sdf.robot as robot_sdf_module
# import hfts_grasp_planner.sdf.costs as costs_module
import openravepy as orpy
import hfts_grasp_planner.sdf.core as sdf_module
import numpy as np
# from mayavi import mlab
# import hfts_grasp_planner.placement.placement_planning as pp_module

ENV_PATH = '/home/ros-devel/workspace/catkin_ws/src/hfts_grasp_planner/data/environments/table_r850.xml'
SDF_PATH = '//home/ros-devel/workspace/catkin_ws/src/hfts_grasp_planner/data/sdfs/table_r850.sdf'
# ROBOT_BALL_PATH = '/home/joshua/projects/grasping_catkin/src/hfts_grasp_planner/models/r850_robotiq/ball_description.yaml'

def visualize():
    grid = sdf_module.VoxelGrid.load('/tmp/test_bunny_grid')
    x, y, z = [], [], []
    for cell in grid:
        if cell.get_value() > 0.0:
            pos = cell.get_position()
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
    mlab.points3d(x, y, z, mode='cube', opacity=0.5) 

def generate_grid():
    env = orpy.Environment()
    env.Load('/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/bunny/objectModel.ply')
    # env.Load(ENV_PATH)
    # env.SetViewer('qtcoin')
    # placement_planner = pp_module.PlacementGoalPlanner(None, env)
    # placement_volume = (np.array([-1.0, -2.0, 0.5]), np.array([0.2, 0.8, 0.9]))
    # placement_planner.set_placement_volume(placement_volume)
    # placement_planner.set_object('crayola')
    # IPython.embed()
    # placement_planner.sample(1)
    bunny = env.GetBodies()[0]
    aabb = bunny.ComputeAABB()
    vol = np.zeros(6)
    vol[:3] = aabb.pos() - aabb.extents()
    vol[3:] = aabb.pos() + aabb.extents()
    grid = sdf_module.VoxelGrid(vol, 0.01)
    builder = sdf_module.OccupancyGridBuilder(env, 0.01)
    builder.compute_grid(grid)
    grid.save('/tmp/test_bunny_grid')
    orpy.RaveDestroy()

if __name__=="__main__":
    # visualize()  # doesn't work properly, run it manually using ipython --gui=qt
    generate_grid()