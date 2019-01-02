#include <stdlib.h>
#include <math.h>

#define DELTA_EPSILON 0.00001
#define YUMI_SE3_WEIGHT 0.07131
#define YUMI_MAX_RADIUS 0.07131
#define YUMI_MIN_RADIUS 0.02
#define SQ_YUMI_MAX_RADIUS (YUMI_MAX_RADIUS * YUMI_MAX_RADIUS)
#define YUMI_DELTA_SQ_RADII (YUMI_MIN_RADIUS*YUMI_MIN_RADIUS-YUMI_MAX_RADIUS*YUMI_MAX_RADIUS)

// def yumi_distance_fn(a, b):
//     # extract quaternions
//     q_a, q_b = a[3:], b[3:]
//     # express everything in frame a
//     q_a_inv = np.array(q_a)
//     q_a_inv[1:] *= -1.0
//     q_b_in_a = orpy.quatMult(q_b, q_a_inv)
//     # rotation distance is determined by the radius of the end-effector. We approximate the end-effector by an ellipse
//     angle = 2.0 * np.arccos(np.clip(np.abs(q_b_in_a[0]), 0.0, 1.0))
//     radius = 0.0
//     if not np.isclose(angle, 0.0):
//         # compute dot product of angle between rotation axis and z axis
//         cos_alpha = q_b_in_a[3] / np.sin(angle)
//         radius = np.sqrt(max(YUMI_MAX_RADIUS**2 + cos_alpha**2 * YUMI_DELTA_SQ_RADII, 0.0))
//     return np.linalg.norm(a[:3] - b[:3]) + radius * angle


void invert_quat(double* in, double* out) {
    out[0] = in[0];
    out[1] = -in[1];
    out[2] = -in[2];
    out[3] = -in[3];
}

void quat_mult(double* a, double* b, double* r) {
    r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    r[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    r[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    r[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

double pose_distance(double* pose_a, double* pose_b) {
    double eucl_dist = 0.0;
    for (int i = 0; i < 3; ++i){
        eucl_dist += (pose_a[i] - pose_b[i]) * (pose_a[i] - pose_b[i]);
    }
    eucl_dist = sqrt(eucl_dist);
    double q_b_in_a[4];
    double q_a_inv[4];
    invert_quat(pose_a + 3, q_a_inv);
    quat_mult(pose_b + 3, q_a_inv, q_b_in_a);
    // rotation distance is determined by the radius of the end-effector. We approximate the end-effector by an ellipse
    double clipped_value = fmax(fmin(abs(q_b_in_a[0]), 1.0), 0.0);
    double angle = 2.0 * acos(clipped_value);
    double radius = 0.0;
    if (angle > DELTA_EPSILON) {
        double cos_alpha = q_b_in_a[3] / sin(angle);
        // radius = sqrt()
        radius = sqrt(fmax(SQ_YUMI_MAX_RADIUS + cos_alpha * cos_alpha * YUMI_DELTA_SQ_RADII, 0.0));
    }
    return eucl_dist + radius * angle;
}