#include <Eigen/Core>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/ORCostsAndValidity.h>

// TODO this is probably too large for practical uses, but it's too slow with pqp otherwise
// TODO make both parameters settable from outside!?
#define STEP_SIZE 0.1
#define SAFE_DISTANCE 0.05

namespace pmp = ::placement::mp;
using namespace placement::mp::mgsearch;

double ::placement::mp::mgsearch::cSpaceDistance(const pmp::Config& a, const pmp::Config& b)
{
    const Eigen::Map<const Eigen::VectorXd> avec(a.data(), a.size());
    const Eigen::Map<const Eigen::VectorXd> bvec(b.data(), b.size());
    return (avec - bvec).norm();
}

ORSceneInterface::ORSceneInterface(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id)
    : _penv(penv)
    , _distance_check_enabled(false)
{
    _robot = _penv->GetRobot(_penv->GetBodyFromEnvironmentId(robot_id)->GetName());
    assert(_robot);
    _object = _penv->GetBodyFromEnvironmentId(obj_id);
    assert(_object);
    _col_checker = _penv->GetCollisionChecker();
    // OpenRAVE::CollisionOptions col_options = OpenRAVE::CollisionOptions::CO_Distance;
    // if (not _col_checker->SetCollisionOptions(col_options)) {
    //     RAVELOG_WARN("Collision checker does not support distance queries. Changing to pqp");
    //     auto pqp_checker = OpenRAVE::RaveCreateCollisionChecker(penv, "pqp");
    //     assert(pqp_checker);
    //     _penv->SetCollisionChecker(pqp_checker);
    //     _col_checker = pqp_checker;
    // }
    _report = boost::shared_ptr<OpenRAVE::CollisionReport>(new OpenRAVE::CollisionReport());
}

ORSceneInterface::~ORSceneInterface() = default;

void ORSceneInterface::addGrasp(const MultiGraspMP::Grasp& g)
{
    auto iter = _grasps.find(g.id);
    if (iter != _grasps.end()) {
        RAVELOG_ERROR("New grasp has the same id as a previously added grasp. Can not add new grasp");
        throw std::logic_error("New grasp has the same id as a previously added grasp. Can not add new grasp");
    }
    _grasps.insert(std::make_pair(g.id, g));
}

void ORSceneInterface::removeGrasp(unsigned int gid)
{
    auto iter = _grasps.find(gid);
    if (iter != _grasps.end()) {
        _grasps.erase(iter);
    } else {
        RAVELOG_WARN("Trying to remove grasp that doesn't exist");
    }
}

bool ORSceneInterface::isValid(const pmp::Config& c) const
{
    boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
    OpenRAVE::RobotBase::RobotStateSaver state_saver(_robot);
    OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
    _robot->ReleaseAllGrabbed();
    _robot->SetActiveDOFValues(c);
    _object->Enable(false);
    enableDistanceCheck(false);
    if (_robot->CheckSelfCollision()) {
        return false;
    }
    return !_penv->CheckCollision(_robot);
}

bool ORSceneInterface::isValid(const pmp::Config& c, unsigned int grasp_id, bool only_obj) const
{
    boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
    OpenRAVE::RobotBase::RobotStateSaver state_saver(_robot);
    OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
    setGrasp(grasp_id);
    _robot->SetActiveDOFValues(c);
    // kinbody should be part of the robot now
    bool bvalid = false;
    enableDistanceCheck(false);
    if (only_obj) {
        bvalid = !_penv->CheckCollision(_object);
    } else {
        bvalid = !_robot->CheckSelfCollision() and !_penv->CheckCollision(_robot);
    }
    _robot->ReleaseAllGrabbed();
    return bvalid;
}

double ORSceneInterface::lowerBound(const pmp::Config& a, const pmp::Config& b) const
{
    return cSpaceDistance(a, b);
}

double ORSceneInterface::cost(const pmp::Config& a, const pmp::Config& b) const
{
    boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
    OpenRAVE::RobotBase::RobotStateSaver rob_state_saver(_robot);
    OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
    return integrateCosts(a, b);
}

double ORSceneInterface::cost(const pmp::Config& a, const pmp::Config& b, unsigned int grasp_id) const
{
    boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
    OpenRAVE::RobotBase::RobotStateSaver rob_state_saver(_robot);
    OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
    setGrasp(grasp_id);
    double val = integrateCosts(a, b);
    _robot->ReleaseAllGrabbed();
    return val;
}

void ORSceneInterface::setGrasp(unsigned int gid) const
{
    auto iter = _grasps.find(gid);
    if (iter == _grasps.end()) {
        throw std::runtime_error(std::string("Could not retrieve grasp with id ") + std::to_string(gid));
    }
    auto grasp = iter->second;
    // set grasp tf
    auto manip = _robot->GetActiveManipulator();
    OpenRAVE::Transform wTe = manip->GetEndEffectorTransform();
    OpenRAVE::Transform oTe(grasp.quat, grasp.pos);
    OpenRAVE::Transform eTo = oTe.inverse();
    OpenRAVE::Transform wTo = wTe * eTo;
    _object->SetTransform(wTo);
    // set hand_config
    auto gripper_indices = manip->GetGripperIndices();
    _robot->SetDOFValues(grasp.gripper_values, 1, gripper_indices);
    // set grasped
    _robot->Grab(_object);
}

double ORSceneInterface::costPerConfig(const pmp::Config& c) const
{
    OpenRAVE::RobotBase::RobotStateSaver state_saver(_robot);
    _robot->SetActiveDOFValues(c);
    enableDistanceCheck(false);
    if (_robot->CheckSelfCollision())
        return INFINITY;
    if (_penv->CheckCollision(_robot))
        return INFINITY;
    return 1.0;
    // enableDistanceCheck(true);
    // _penv->CheckCollision(_robot, _report);
    // double clearance = _report->minDistance;
    // return SAFE_DISTANCE / std::min(clearance, SAFE_DISTANCE);
}

double ORSceneInterface::integrateCosts(const pmp::Config& a, const pmp::Config& b) const
{
    assert(a.size() == b.size());
    Eigen::Map<const Eigen::VectorXd> avec(a.data(), a.size());
    Eigen::Map<const Eigen::VectorXd> bvec(b.data(), b.size());
    Eigen::VectorXd delta = bvec - avec;
    pmp::Config q(delta.size());
    Eigen::Map<Eigen::VectorXd> qvec(q.data(), q.size());
    double norm = delta.norm();
    if (norm == 0.0) {
        return 0.0;
    }
    // iterate over path
    double integral_cost = 0.0;
    unsigned int num_steps = std::ceil(norm / STEP_SIZE);
    for (size_t t = 0; t < num_steps; ++t) {
        qvec = std::min(t * STEP_SIZE / norm, 1.0) * delta + avec;
        double dc = costPerConfig(q); // qvec is a view on q's data
        if (std::isinf(dc))
            return INFINITY;
        integral_cost += dc * STEP_SIZE;
    }
    return integral_cost;
}

void ORSceneInterface::enableDistanceCheck(bool enable) const
{
    if (enable and not _distance_check_enabled) {
        _col_checker->SetCollisionOptions(OpenRAVE::CollisionOptions::CO_Distance | OpenRAVE::CollisionOptions::CO_ActiveDOFs);
    } else if (not enable and _distance_check_enabled) {
        _col_checker->SetCollisionOptions(OpenRAVE::CollisionOptions::CO_ActiveDOFs);
    }
    _distance_check_enabled = enable;
}
