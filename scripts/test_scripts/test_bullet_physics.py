import os
import IPython
import openravepy as orpy
import hfts_grasp_planner.placement.bullet as bullet

if __name__ == '__main__':
    env = orpy.Environment()
    my_dir = os.path.dirname(__file__)
    env.Load(my_dir + '/../../data/environments/placement_exp_1.xml')
    urdf_file = my_dir + '/../../models/yumi/yumi.urdf'
    urdf_path = my_dir + '/../../data/placement_problems/objects/'
    env.SetViewer('qtcoin')
    physics_model = bullet.PhysicsModel(env, {'Yumi': urdf_file}, urdf_path)
    physics_model.init_environment()
    IPython.embed()
