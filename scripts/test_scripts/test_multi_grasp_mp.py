#!/usr/bin/python
import openravepy as orpy
import IPython

if __name__ == "__main__":
    env = orpy.Environment()
    orpy.RaveSetDebugLevel(orpy.DebugLevel.Debug)
    orpy.RaveLoadPlugin("ormultigraspmp")
    # planner = orpy.RaveCreatePlanner(env, "ParallelMGBiRRT")
    module = orpy.RaveCreateModule(env, "ParallelMGBiRRT")
    IPython.embed()
