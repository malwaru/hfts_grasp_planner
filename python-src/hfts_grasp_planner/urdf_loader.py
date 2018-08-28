#! /usr/bin/python
import openravepy as orpy
import numpy as np
import yaml


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
