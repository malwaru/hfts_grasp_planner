#! /usr/bin/python
import openravepy as orpy
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an ikfast model for the given robot.")
    parser.add_argument('robot_file_name', type=str, help='Path to an OpenRAVE robot')
    parser.add_argument('--manipulator_name', type=str, help='Name of the manipulator to create the IK for')
    args = parser.parse_args()
    env = orpy.Environment()
    # env.SetViewer('qtcoin')
    if env.Load(args.robot_file_name):
        robot = env.GetRobots()[0]
        manip = robot.GetActiveManipulator()
        if args.manipulator_name:
            manip = robot.GetManipulator(args.manipulator_name)
            if manip:
                robot.SetActiveManipulator(args.manipulator_name)
            else:
                print("Could not find manipulator with name " + args.manipulator_name)
        print("Generating IK for manipulator " + manip.GetName())
        arm_ik = orpy.databases.inversekinematics.InverseKinematicsModel(robot,
                                                                         iktype=orpy.IkParameterization.Type.Transform6D)
        if not arm_ik.load():
            print('No IKFast solver found. Generating new one.')
            arm_ik.autogenerate()
            print("Generation complete. Goodbye!")
    else:
        print("You need to provide a valid path to an OpenRAVE robot")
    sys.exit(0)
