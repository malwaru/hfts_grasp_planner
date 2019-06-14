#! /usr/bin/python
import openravepy as orpy
import numpy as np
import IPython
from copy import deepcopy
from hfts_grasp_planner.dmg.dmg_class import DexterousManipulationGraph as DMG
from hfts_grasp_planner.utils import (is_dynamic_body, inverse_transform, get_manipulator_links,
                                      set_grasp, set_body_color, set_body_alpha, get_tf_interpolation, get_tf_gripper)

def check_floating_gripper_colision(gripper, target_object, grasp_obj_tf, grasp_tf_gripper, env):
    '''
    Checks the floating gripper for collisions with the environment
    '''
    target_pose = target_object.GetTransform()
    collision_grasp_tf = np.dot(target_pose, grasp_obj_tf)
    collision_eef_tf = np.dot(collision_grasp_tf, grasp_tf_gripper)
    gripper.SetTransform(collision_eef_tf)
    in_collision = env.CheckCollision(gripper)
    return in_collision

def get_grasp_tf(gripper):
    '''
    Creates a grasp tf, given a tf from the dmg nodes 
    '''
    grasp_tf = inverse_transform(get_tf_gripper(gripper=gripper))
    return grasp_tf

def get_grasp(dmg_node, dmg_angle, robot, dmg):
    grasp_tf = get_tf_gripper(gripper=robot.GetJoint('gripper_r_joint'))
    object_tf = dmg.make_transform_matrix(dmg_node, dmg_angle)
    if object_tf is None:
        raise ValueError("The provided DMG node has no valid opposite node. Aborting")
    return inverse_transform(np.dot(grasp_tf, inverse_transform(object_tf)))

def set_body_color(body, color):
    """
        Set the color of the given kinbody for visualization.
        ---------
        Arguments
        ---------
        body, OpenRAVE kinbody - kinbody to set color for (all links and geometries)
        color, np array of shape (3,) or (4,) - [r, g, b, a=1], with all values in range [0, 1]
    """
    links = body.GetLinks()
    for link in links:
        geoms = link.GetGeometries()
        for geom in geoms:
            geom.SetDiffuseColor(color[:3])
            geom.SetAmbientColor(color[:3])
            if color.shape[0] == 4:
                geom.SetTransparency(1.0 - color[3])

if __name__ == "__main__":
    env = orpy.Environment()
    env.Load("/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/environments/table_low_clutter.xml")
    # btarget_found = env.Load("../../data/crayola_24/crayola_24.kinbody.xml")
    btarget_found = env.Load("/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/elmers_glue/elmers_glue.kinbody.xml")
    if not btarget_found:
        raise ValueError("Could not load target object. Aborting")
    target_obj_name = "target_object"
    env.GetBodies()[-1].SetName(target_obj_name)

    robot = env.GetRobot("Yumi")

    robot1 = env.ReadRobotURI('/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/models/yumi/yumi_gripper_r.robot.xml')
    env.Add(robot1,True)

    gripper = env.GetRobot('yumi_gripper')
    gripper.SetJointValues([0.02], [0])
    target_object = env.GetKinBody(target_obj_name)

    dmg = DMG()
    dmg.set_object_shape_file('/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/elmers_glue/glue.stl')
    dmg.read_graph('/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/dmg_files/glue/graph_glue_12_20.txt')
    dmg.read_nodes('/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/dmg_files/glue/node_position_glue_12_20.txt')
    dmg.read_node_to_component('/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/dmg_files/glue/node_component_glue_12_20.txt')
    dmg.read_component_to_normal('/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/dmg_files/glue/component_normal_glue_12_20.txt')
    dmg.read_node_to_angles('/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/dmg_files/glue/node_angle_glue_12_20.txt')
    dmg.read_supervoxel_angle_to_angular_component('/home/rizwan/projects/placement_ws/src/hfts_grasp_planner/data/dmg_files/glue/node_angle_angle_component_glue_12_20.txt')

    initial_dmg_node = (130,0)
    initial_dmg_angle = 300

    dmg.set_current_node(initial_dmg_node)
    dmg.set_current_angle(initial_dmg_angle)
    grasp_pose = get_grasp(initial_dmg_node, initial_dmg_angle, robot, dmg)

    grasp_order = dmg.run_dijsktra(initial_dmg_node, initial_dmg_angle, angle_weight=0.0)

    # check here
    # grasp_tf = get_grasp_tf(robot.GetJoint('gripper_r_joint'))
    grasp_tf_gripper = get_grasp_tf(gripper.GetJoints()[0])

    

    env.SetViewer('qtcoin')
    
    IPython.embed()