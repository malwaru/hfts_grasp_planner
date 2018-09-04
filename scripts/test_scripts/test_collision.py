#! /usr/bin/python
import os
import sys
import time
import IPython
import numpy as np
import openravepy as orpy


def show_contact_pose(contact_info, body):
    tf = orpy.matrixFromQuat(contact_info[1][:4])
    tf[:3, 3] = contact_info[1][4:]
    body.SetTransform(tf)


def show_contacts(env, contacts, arrow_length=0.01, arrow_width=0.001):
    handles = []
    for contact in contacts:
        p1 = contact.pos
        p2 = p1 + arrow_length * contact.norm
        handles.append(env.drawarrow(p1, p2, arrow_width))
    return handles


def test_ccd(env, body, target_tf=None, b_overwrite_pose=True):
    link = body.GetLinks()[0]
    if target_tf is None:
        target_tf = np.array([[1.00000000e+00,   1.18930618e-07,   1.19208958e-07, 7.24631399e-02],
                              [1.18930646e-07,   2.33737825e-03,  -9.99997268e-01, 7.80003488e-01],
                              [-1.19208930e-07,   9.99997268e-01,   2.33737825e-03, 6.94022834e-01],
                              [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])
    if b_overwrite_pose:
        current_tf = np.array([[1.00000000e+00,   0.00000000e+00,   0.00000000e+00, 7.24631399e-02],
                               [0.00000000e+00,   1.00000000e+00,  -1.42108505e-14, 7.80003428e-01],
                               [0.00000000e+00,   1.42108505e-14,   1.00000000e+00, 6.94022834e-01],
                               [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 1.00000000e+00]])
        body.SetTransform(current_tf)
    report = orpy.ContinuousCollisionReport()
    result = env.CheckContinuousCollision(link, target_tf, report)
    print report
    return report


def test_ray_trace(env, positions, dirs, length):
    start_time = time.time()
    rays = np.empty((len(positions), 6))
    rays[:, :3] = positions
    rays[:, 3:] = length * dirs
    contacts = env.CheckCollisionRays(rays)
    print contacts, 'It took: ', time.time() - start_time


def test_dcd_contact(env, body, b_overwrite_pose=True):
    if b_overwrite_pose:
        tf = np.array([[1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                        7.03271255e-02],
                       [0.00000000e+00,   1.00000000e+00,  -2.84217010e-14,
                        7.80003488e-01],
                       [0.00000000e+00,   2.84217010e-14,   1.00000000e+00,
                        6.64296269e-01],
                       [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                        1.00000000e+00]])
        body.SetTransform(tf)
    now = time.time()
    env.GetCollisionChecker().SetCollisionOptions(orpy.CollisionOptions.Contacts |
                                                  orpy.CollisionOptions.AllGeometryContacts |
                                                  orpy.CollisionOptions.AllLinkCollisions)
    report = orpy.CollisionReport()
    env.CheckCollision(body, report)
    print report, time.time() - now
    return report


def visualize_contacts(env, contacts, arrow_length=0.01, arrow_width=0.0001, color=None):
    handles = []
    if color is None:
        color = [1, 0, 0]
    for contact in contacts:
        if type(contact) == np.ndarray:
            p1 = contact[:3]
            p2 = contact[:3] + arrow_length * contact[3:]
        else:
            p1 = contact.pos
            p2 = p1 + arrow_length * contact.norm
        handles.append(env.drawarrow(p1, p2, arrow_width, color=color))
    return handles


if __name__ == '__main__':
    env = orpy.Environment()
    env.SetDebugLevel(orpy.DebugLevel.Debug)
    path = os.path.abspath(os.path.dirname(__file__))
    env_path = path + '/../../data/environments/cluttered_env.xml'
    if env.Load(env_path):
        ode_col_checker = orpy.RaveCreateCollisionChecker(env, 'ode')
        ode_col_checker.SetCollisionOptions(orpy.CollisionOptions.Contacts |
                                            orpy.CollisionOptions.AllGeometryContacts |
                                            orpy.CollisionOptions.AllLinkCollisions)
        env.SetCollisionChecker(ode_col_checker)
        env.SetViewer('qtcoin')
        robot = env.GetRobots()[0]
        body = env.GetKinBody('cube')
        body.SetTransform(np.array([[1.00000000e+00,   6.05844673e-28,   2.13162719e-14,
                                     -9.53442529e-02],
                                    [-3.02922320e-28,   1.00000000e+00,  -1.42108505e-14,
                                     6.41140580e-01],
                                    [-2.13162719e-14,   1.42108505e-14,   1.00000000e+00,
                                     4.67702389e-01],
                                    [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                     1.00000000e+00]]))
        # report = test_ccd(env, body)
        # report = test_dcd_contact(env, body)
        # handles = visualize_contacts(env, report.contacts)
        IPython.embed()
    else:
        print "Could not load environment %s" % env_path
        sys.exit(-1)
