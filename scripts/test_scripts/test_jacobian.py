import numpy as np
import openravepy as orpy
import hfts_grasp_planner.external.transformations as tf_mod
import IPython


def get_state(link, lpos):
    link_pose = link.GetTransform()
    link_pose[:3, 3] = np.dot(link_pose[:3, :3], lpos) + link_pose[:3, 3]
    tex, tey, tez = tf_mod.euler_from_matrix(link_pose[:3, :3])
    return np.array([link_pose[0, 3], link_pose[1, 3], link_pose[2, 3], tex, tey, tez])


def move_line_all(manip, link, lpos, displacement, m_err=0.004, rot_err=0.01, max_counter=10000):
    robot = manip.GetRobot()
    robot.SetActiveDOFs(manip.GetArmIndices())
    current_state = get_state(link, lpos)
    target_state = current_state + displacement
    jac = np.zeros((6, manip.GetArmDOF()))
    print 'Current state is:', current_state
    print 'Target state is:', target_state
    counter = 0
    error = target_state - current_state
    while (np.linalg.norm(error[:3], np.inf) > m_err or np.linalg.norm(error[3:], np.inf) > rot_err) and counter < max_counter:
        # compute jacobian
        # jac[:3] = manip.CalculateJacobian()
        # jac[3:] = manip.CalculateAngularVelocityJacobian()
        link_pose = link.GetTransform()
        pos = np.dot(link_pose[:3, :3], lpos) + link_pose[:3, 3]
        jac[:3] = robot.CalculateActiveJacobian(link.GetIndex(), pos)
        jac[3:] = robot.CalculateActiveAngularVelocityJacobian(link.GetIndex())
        inv_jac = np.linalg.pinv(jac)
        errordir = error[:3] / np.linalg.norm(error[:3])
        dq = np.dot(inv_jac, 0.01 * errordir)
        q = robot.GetActiveDOFValues() + dq
        robot.SetActiveDOFValues(q)
        current_state = get_state(link, lpos)
        error = target_state - current_state
        print "iteration: %i, error: %s" % (counter, str(error))
        counter += 1


def move_line_cart(manip, link, lpos, displacement, m_err=0.004, max_counter=10000):
    robot = manip.GetRobot()
    robot.SetActiveDOFs(manip.GetArmIndices())
    current_state = get_state(link, lpos)
    target_state = current_state + displacement
    jac = np.zeros((3, manip.GetArmDOF()))
    print 'Current state is:', current_state
    print 'Target state is:', target_state
    counter = 0
    error = target_state - current_state
    while np.linalg.norm(error[:3], np.inf) > m_err and counter < max_counter:
        # compute jacobian
        # jac[:3] = manip.CalculateJacobian()
        # jac[3:] = manip.CalculateAngularVelocityJacobian()
        link_pose = link.GetTransform()
        pos = np.dot(link_pose[:3, :3], lpos) + link_pose[:3, 3]
        jac = robot.CalculateActiveJacobian(link.GetIndex(), pos)
        inv_jac = np.linalg.pinv(jac)
        errordir = error[:3] / np.linalg.norm(error[:3])
        dq = np.dot(inv_jac, 0.01 * errordir)
        q = robot.GetActiveDOFValues() + dq
        robot.SetActiveDOFValues(q)
        current_state = get_state(link, lpos)
        error = target_state - current_state
        print "iteration: %i, error: %s" % (counter, str(error[:3]))
        counter += 1


if __name__ == "__main__":
    env = orpy.Environment()
    env.Load('/home/joshua/projects/placement_planning/src/hfts_grasp_planner/data/environments/placement_exp_0.xml')
    yumi = env.GetRobots()[0]
    env.SetViewer('qtcoin')
    manip = yumi.GetActiveManipulator()
    link = yumi.GetLink("yumi_link_5_r")
    lpos = np.array([0, 0, 0])
    displacement = np.array([0, 0, 0.1, 0, 0, 0])
    IPython.embed()
