#! /usr/bin/python
import openravepy as orpy
import numpy as np
import yaml
import xml.etree.cElementTree as ET
from hfts_grasp_planner.external.transformations import euler_from_matrix


def set_base_pose(urdf_content, pose):
    """
        Return a string containing a modified urdf, where the base pose of the robot
        is set to the given pose.
        ---------
        Arguments
        ---------
        urdf_content, string - a string containing the URDF description of a robot.
        pose, numpy array of shape (4, 4) - pose of the robot base link
    """
    root = ET.fromstring(urdf_content)
    robot_name = root.attrib["name"]
    # TODO instead of modifying the transformation from base_link to the first body,
    # TODO we should maybe instead just add a "world_frame" link
    base_link_name = '%s_base_link' % robot_name
    # get the joint from base link to the robot body
    base_joint_elem = root.find("./joint[@type='fixed']/parent[@link='%s']/.." % base_link_name)
    if base_joint_elem is None:
        raise ValueError("Could not locate a fixed joint from %s to any other link.")
    pose_elem = base_joint_elem.find('.origin')
    if pose_elem is None:
        pose_elem = ET.SubElement(base_joint_elem, u'origin', rpy=u'0,0,0', xyz=u'0, 0, 0')
    pose_elem.attrib[u'rpy'] = u'%f, %f, %f' % euler_from_matrix(pose[:3, :3])
    pose_elem.attrib[u'xyz'] = u'%f, %f, %f' % tuple(pose[:3, 3])
    return ET.tostring(root, 'UTF-8')


def add_grasped_obj(urdf_content, eef_name, obj_name, grasp_tf):
    """
        Return a string containing a modified urdf, where there is an additional
        link with name obj_name that is fixed at pose grasp_tf relative to the link with eef_name.
        ---------
        Arguments
        ---------
        urdf_content, string - a string containing the URDF description of a robot.
        eef_name, string - name of the end-effector link
        obj_name, string - name of the object
        grasp_tf, numpy array of shape (4, 4) - the pose of the object w.r.t. to the end-effector frame,
            i.e. x' = grasp_tf * x translates a point x in object frame into eef_frame
    """
    root = ET.fromstring(urdf_content)
    # first do sanity checks  - does the end effector actually exist?
    eef_el = root.find(".link[@name='%s']" % eef_name)
    if eef_el is None:
        raise ValueError("Could not locate link with name %s" % eef_name)
    # does the object already exist?
    obj_el = root.find(".link[@name='%s']" % obj_name)
    if obj_el is not None:
        raise ValueError("A link with name %s already exists" % obj_name)
    # if all is good, add object link
    ET.SubElement(root, u"link", name=obj_name)  # attrib={u"name": obj_name}
    # add joint
    new_joint_el = ET.SubElement(root, 'joint', name=u'grasp_%s_%s' % (eef_name, obj_name), type=u'fixed')
    # add parent
    ET.SubElement(new_joint_el, u'parent', link=unicode(eef_name))
    # add child
    ET.SubElement(new_joint_el, u'child', link=unicode(obj_name))
    # import IPython
    # IPython.embed()
    # add transform
    rpy = u'%f, %f, %f' % euler_from_matrix(grasp_tf[:3, :3])
    xyz = u'%f, %f, %f' % tuple(grasp_tf[:3, 3])
    ET.SubElement(new_joint_el, 'origin', rpy=rpy, xyz=xyz)
    return ET.tostring(root, encoding='UTF-8')


def load_robot(urdf_file, srdf_file, env):
    """
        Load a URDF robot directly from file.
        ---------
        Arguments
        ---------
        urdf_file, string - urdf file name
        srdf_file, string - srdf file name
    """
    plugin = orpy.RaveCreateModule(env, 'urdf')
    if plugin is None:
        raise RuntimeError(
            "Could not load OpenRAVE-URDF plugin. Please install it and make sure the environment variables are setup so it can be found.")
    with env:
        name = plugin.SendCommand('loadURI ' + urdf_file + ' ' + srdf_file)
        if name is None:
            raise IOError("Could not load robot from URDF:%s and SRDF:%s" % (urdf_file, srdf_file))


def load_robot_from_yaml(yaml_file_name, env):
    """
        Loads a URDF robot from a yaml file that contains a path to the robot
        URDF and SRDF and additional information.
        ---------
        Arguments
        ---------
        yaml_file_name, string - path to yaml file
        env, OpenRAVE environment
    """
    # TODO if we call this multiple times, it might be better to save this somewhere
    plugin = orpy.RaveCreateModule(env, 'urdf')
    if plugin is None:
        raise RuntimeError(
            "Could not load OpenRAVE-URDF plugin. Please install it and make sure the environment variables are setup so it can be found.")
    with open(yaml_file_name, 'r') as yaml_file:
        infos = yaml.load(yaml_file)
        with env:
            name = plugin.SendCommand('loadURI ' + infos['urdf'] + ' ' + infos['srdf'])
            if name is None:
                raise IOError("Could not load robot from URDF:%s and SRDF:%s" % (infos['urdf'], infos['srdf']))
            robot = env.GetRobot(name)
            robot.SetDOFResolution(np.ones(robot.GetDOF()))
            if 'dof_weights' in infos:
                robot.SetDOFWeights(infos['dof_weights'])
            else:
                robot.SetDOFWeights(np.ones(robot.GetDOF()))
            if 'acceleration_limits' in infos:
                robot.SetDOFAccelerationLimits(infos['acceleration_limits'])
            else:
                robot.SetDOFAccelerationLimits(np.ones(robot.GetDOF()))
            # TODO ik solver?
            manips = robot.GetManipulator()
            for manip in manips:
                if manip.GetName() + '_info' in infos:
                    manip_info = infos[manip.GetName() + '_info']
                    if 'closing_dir' in manip_info:
                        manip.SetClosingDirection(manip_info['closing_dir'])
                    if 'tool_dir' in manip_info:
                        manip.SetLocalToolDirection(manip_info['tool_dir'])
                    if 'tool_tf' in manip_info:
                        manip.SetLocalToolTransform(manip_info['tool_tf'])
