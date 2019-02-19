#! /usr/bin/python
import openravepy as orpy
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show the center of mass of the given body")
    parser.add_argument('body', type=str, help='Path to xml file of body to show')
    args = parser.parse_args()

    env = orpy.Environment()
    env.Load(args.body)
    env.SetViewer('qtcoin')

    body = env.GetBodies()[0]
    handle = env.drawbox(body.GetCenterOfMass(), np.array([0.005, 0.005, 0.005]))

    print "Press any key to terminate."
    raw_input()
    orpy.RaveDestroy()
