#! /usr/bin/python
import openravepy as orpy
import numpy as np
import IPython
from copy import deepcopy

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
    env.Load("../../data/environments/table_low_clutter.xml")
    # btarget_found = env.Load("../../data/crayola_24/crayola_24.kinbody.xml")
    # btarget_found = env.Load("../../data/elmers_glue/elmers_glue.kinbody.xml")
    # if not btarget_found:
    #     raise ValueError("Could not load target object. Aborting")
    # target_obj_name = "target_object"
    # env.GetBodies()[-1].SetName(target_obj_name)

    robot = env.GetRobot("Yumi")

    robot1 = env.ReadRobotURI('../../models/yumi/yumi_gripper_r.robot.xml')
    env.Add(robot1,True)

    gripper = env.GetRobot('yumi_gripper')

    #Joints and Links
    # rjoint1 = robot.GetJoint('gripper_r_joint')
    # rlink1 = robot.GetLink('gripper_r_base')
    # rlink2 = robot.GetLink('gripper_r_finger_r')
    # rlink3 = robot.GetLink('gripper_r_finger_l')

    # #Infos
    # rj1Info = rjoint1.GetInfo()

    # rl1Info = rlink1.GetInfo()
    # rl1Geo = rlink1.GetGeometries()

    # rl2Info = rlink2.GetInfo()
    # rl2Geo = rlink2.GetGeometries()

    # rl3Info = rlink3.GetInfo()
    # rl3Geo = rlink3.GetGeometries()

    # #New Links
    # link0 = orpy.KinBody.LinkInfo()
    # link0._vgeometryinfos = [x.GetInfo() for x in rl1Geo]
    # link0._name = 'link0'
    # link0._mapFloatParameters = rl1Info._mapFloatParameters
    # link0._mapIntParameters = rl1Info._mapIntParameters

    # link1 = orpy.KinBody.LinkInfo()
    # link1._vgeometryinfos = [x.GetInfo() for x in rl2Geo]
    # link1._name = 'link1'
    # link1._mapFloatParameters = rl2Info._mapFloatParameters
    # link1._mapIntParameters = rl2Info._mapIntParameters

    # link2 = orpy.KinBody.LinkInfo()
    # link2._vgeometryinfos = [x.GetInfo() for x in rl3Geo]
    # link2._name = 'link2'
    # link2._mapFloatParameters = rl3Info._mapFloatParameters
    # link2._mapIntParameters = rl3Info._mapIntParameters

    # # New Joint
    # joint0 = orpy.KinBody.JointInfo()
    # joint0._name = 'j0'
    # joint0._linkname0 = 'link0'
    # # joint0._linkname1 = 'link1'
    # # joint0._linkname2 = 'link2'
    # joint0._type = rj1Info._type
    # joint0._vlowerlimit = rj1Info._vlowerlimit
    # joint0._vupperlimit = rj1Info._vupperlimit
    # joint0._vaxes = rj1Info._vaxes

    # # New Body
    # body = orpy.RaveCreateKinBody(env,'')
    # body.Init([link0,link1,link2],[joint0])
    # body.SetName('temp')
    # env.Add(body)

    env.SetViewer('qtcoin')
    
    IPython.embed()