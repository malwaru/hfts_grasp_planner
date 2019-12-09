from hfts_grasp_planner.placement.goal_sampler.interfaces import PlacementObjective
from hfts_grasp_planner.placement.clearance import ClearanceObjective
from hfts_grasp_planner.utils import inverse_transform
import numpy as np


class DeepShelfObjective(PlacementObjective):
    """
        Objective function that is hardcoded to increase along the world's y-axis
        in addition to clearance objective.
    """

    def __init__(self, object_body, clearance_obj):
        """
            Create a new DeepShelfObjective.
        """
        self._clearance_objective = clearance_obj
        with object_body:
            object_body.SetTransform(np.eye(4))
            aabb = object_body.ComputeAABB()
            self._local_pos = aabb.pos()

    def __call__(self, obj_tf):
        cval = self._clearance_objective(obj_tf)
        center_pos = np.dot(obj_tf[:3, :3], self._local_pos) + obj_tf[:3, 3]
        pval = center_pos[1]  # - int(center_pos[2] / 0.02)  # z makes a categorical difference
        # print "Clearance: ", cval
        # print "Center value:", pval
        return cval + pval

    def evaluate(self, obj_tf):
        return self(obj_tf)

    def get_num_evaluate_calls(self, b_reset=True):
        return self._clearance_objective.get_num_evaluate_calls(b_reset)

    def get_gradient(self, obj_tf, state, to_ref_pose):
        cval_grad = self._clearance_objective.get_gradient(obj_tf, state, to_ref_pose)
        # compute gradient of deep in shelf objective
        center_pos_in_ref = np.dot(to_ref_pose[:3, :3], self._local_pos) + to_ref_pose[:3, 3]
        # get object state
        _, _, theta = state
        # compute gradient w.r.t to state
        r = np.array([[-np.sin(theta), np.cos(theta)], [-np.cos(theta), -np.sin(theta)]])
        dxi_dtheta = np.matmul(center_pos_in_ref[:2], r)
        lcart_grad = np.array([0.0, 1.0, 0.0])
        lcart_grad[2] = np.dot(lcart_grad[:2], dxi_dtheta)
        return cval_grad + lcart_grad
