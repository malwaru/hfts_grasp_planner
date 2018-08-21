#! /usr/bin/python

import os
import IPython
import openravepy as orpy
import timeit
import numpy as np


def simulate_fall(env, start_tf, body, duration=10.0, dt=0.001):
    body.SetTransform(start_tf)
    with env:
        t = 0.0
        while t < duration:
            env.StepSimulation(dt)
            t += dt
    # print body.GetTransform()
    return body.GetTransform()


def setup_env():
    ENV_PATH = '/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/simple_env.xml'
    env = orpy.Environment()
    env.Load(ENV_PATH)
    return env


def measure_time():
    runtime = timeit.timeit("simulate_fall(env, tf, body, duration=1.0)",
                            setup="from __main__ import setup_env, simulate_fall; env = setup_env(); body = env.GetKinBody('crayola'); tf = body.GetTransform()",
                            number=100)
    print runtime / 100.0


if __name__ == "__main__":
    measure_time()
    # env = setup_env()
    # orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
    # with env:
    # physics_engine = orpy.RaveCreatePhysicsEngine(env, 'ode')
    # physics_engine.SetGravity([0, 0, -9.81])
    # env.SetPhysicsEngine(physics_engine)
    # env.StopSimulation()
    # env.SetViewer('qtcoin')
    # IPython.embed()
    # with env:
    # env.StartSimulation(timestep=0.0001)
    # IPython.embed()
