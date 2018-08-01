#! /usr/bin/python

import os
import IPython
import openravepy as orpy
import numpy as np

ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/simple_env.xml'

if __name__=="__main__":
    env = orpy.Environment()
    orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
    with env:
        env.Load(ENV_PATH)
        env.StopSimulation()
    # physics_engine = orpy.RaveCreatePhysicsEngine(env, 'ode')
    # env.SetPhysicsEngine(physics_engine)
    env.SetViewer('qtcoin')
    IPython.embed()
    with env:
        env.StartSimulation(timestep=0.01)
    IPython.embed()