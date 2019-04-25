#! /usr/bin/python
import openravepy as orpy
import numpy as np
import IPython

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
    btarget_found = env.Load("../../data/elmers_glue/elmers_glue.kinbody.xml")
    if not btarget_found:
        raise ValueError("Could not load target object. Aborting")
    target_obj_name = "target_object"
    env.GetBodies()[-1].SetName(target_obj_name)

    robot = env.GetRobot("Yumi")
    body = env.GetKinBody(target_obj_name)

    set_body_color(body, np.array([0.0, 0.5, 0.0]))


    env.SetViewer('qtcoin')
    
    IPython.embed()