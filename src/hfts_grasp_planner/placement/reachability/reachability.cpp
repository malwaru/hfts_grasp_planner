#include <cmath>
#include <hfts_grasp_planner/placement/reachability/reachability.h>

namespace py = pybind11;

#define DELTA_EPSILON 0.00001
#define YUMI_SE3_WEIGHT 0.07131
#define YUMI_MAX_RADIUS 0.07131
#define YUMI_MIN_RADIUS 0.02
#define SQ_YUMI_MAX_RADIUS (YUMI_MAX_RADIUS * YUMI_MAX_RADIUS)
#define YUMI_DELTA_SQ_RADII (YUMI_MIN_RADIUS * YUMI_MIN_RADIUS - YUMI_MAX_RADIUS * YUMI_MAX_RADIUS)

void invert_quat(const double* in, double* out)
{
    out[0] = in[0];
    out[1] = -in[1];
    out[2] = -in[2];
    out[3] = -in[3];
}

void quat_mult(const double* a, const double* b, double* r)
{
    r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    r[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    r[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    r[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

double eef_pose_distance_fn(const EndEffectorPose& a, const EndEffectorPose& b)
{
    double eucl_dist = 0.0;
    for (int i = 0; i < 3; ++i) {
        eucl_dist += (a.pos[i] - b.pos[i]) * (a.pos[i] - b.pos[i]);
    }
    eucl_dist = sqrt(eucl_dist);
    double q_b_in_a[4];
    double q_a_inv[4];
    invert_quat(a.quat, q_a_inv);
    quat_mult(b.quat, q_a_inv, q_b_in_a);
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

YumiReachabilityMap::YumiReachabilityMap(py::array_t<double, py::array::c_style | py::array::forcecast>& data)
{
    if (data.ndim() != 2) {
        throw std::runtime_error("Input array data must have dimension 2");
    }
    if (data.shape(1) != 7 or data.shape(0) == 0) {
        throw std::runtime_error("Input array data must have shape (n, 7) with n >= 1");
    }
    using namespace std::placeholders;
    _gnat.setDistanceFunction(std::bind(eef_pose_distance_fn, _1, _2));
    // auto raw_data = data.unchecked<2>();
    for (size_t i = 0; i < data.shape(0); ++i) {
        auto row_pointer = data.data(i);
        _gnat.add(EndEffectorPose(row_pointer[0], row_pointer[1], row_pointer[2], row_pointer[3],
            row_pointer[4], row_pointer[5], row_pointer[6], i));
    }
}

YumiReachabilityMap::~YumiReachabilityMap() = default;

py::tuple YumiReachabilityMap::query(py::array_t<double, py::array::c_style | py::array::forcecast>& poses)
{
    if (poses.shape(1) != 7) {
        throw std::runtime_error("Invalid input array. Input array must have shape (n, 7) with n >= 0");
    }
    std::vector<size_t> output_shape(1);
    output_shape[0] = poses.shape(0);
    py::array_t<double, py::array::c_style> distances(output_shape);
    py::array_t<size_t, py::array::c_style> indices(output_shape);
    double* distances_ptr = distances.mutable_data();
    size_t* indices_ptr = indices.mutable_data();
    // for each input pose
    for (size_t p = 0; p < poses.shape(0); ++p) {
        auto pose_ptr = poses.data(p);
        EndEffectorPose query_pose(pose_ptr[0], pose_ptr[1], pose_ptr[2], pose_ptr[3], pose_ptr[4], pose_ptr[5], pose_ptr[6], 0);
        auto nearest_pose = _gnat.nearest(query_pose);
        distances_ptr[p] = eef_pose_distance_fn(query_pose, nearest_pose);
        indices_ptr[p] = nearest_pose.id;
    }
    return py::make_tuple(distances, indices);
}