#include <boost/shared_ptr.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/ORCostsAndValidity.h>

#define STEP_SIZE 0.001

using namespace placement::mp::mgsearch;

ORSceneInterface::ORSceneInterface(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id)
    : _penv(penv)
{
    _robot = _penv->GetRobot(_penv->GetBodyFromEnvironmentId(robot_id)->GetName());
    assert(_robot);
    _object = _penv->GetBodyFromEnvironmentId(obj_id);
    assert(_object);
    // TODO we could have two separate collision checkers; one for distance one for collision only (fcl is maybe faster!)
    auto col_checker = _penv->GetCollisionChecker();
    OpenRAVE::CollisionOptions col_options = OpenRAVE::CollisionOptions::CO_Distance;
    if (not col_checker->SetCollisionOptions(col_options)) {
        RAVELOG_WARN("Collision checker does not support distance queries. Changing to pqp");
        auto pqp_checker = OpenRAVE::RaveCreateCollisionChecker(penv, "pqp");
        assert(pqp_checker);
        _penv->SetCollisionChecker(pqp_checker);
        pqp_checker->SetCollisionOptions(col_options);
    }
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

bool ORSceneInterface::isValid(const Config& c) const
{
    boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
    OpenRAVE::RobotBase::RobotStateSaver state_saver(_robot);
    _robot->ReleaseAllGrabbed();
    _robot->SetActiveDOFValues(c);
    if (_robot->CheckSelfCollision()) {
        return false;
    }
    return !_penv->CheckCollision(_robot);
}

bool ORSceneInterface::isValid(const Config& c, unsigned int grasp_id) const
{
    boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
    OpenRAVE::RobotBase::RobotStateSaver state_saver(_robot);
    setGrasp(grasp_id);
    _robot->SetActiveDOFValues(c);
    // kinbody should be part of the robot now
    bool bvalid = !_robot->CheckSelfCollision() and !_penv->CheckCollision(_robot);
    _robot->ReleaseAllGrabbed();
    return bvalid;
}

double ORSceneInterface::lowerBound(const Config& a, const Config& b) const
{
    assert(a.size() == b.size());
    // just Euclidean distance
    double delta = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        delta += (a.at(i) - b.at(i)) * (a.at(i) - b.at(i));
    }
    return sqrt(delta);
}

double ORSceneInterface::cost(const Config& a, const Config& b) const
{
}

double ORSceneInterface::cost(const Config& a, const Config& b, unsigned int grasp_id) const
{
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

double ORSceneInterface::costPerConfig(const Config& c) const
{
    boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
    OpenRAVE::RobotBase::RobotStateSaver state_saver(_robot);
    _robot->SetActiveDOFValues(c);
    if (_robot->CheckSelfCollision())
        return INFINITY;
    _penv->CheckCollision(_robot, _report);
    double clearance = _report->minDistance;
    // TODO define clearance cost (should only be > 0 if clearance < some threshhold)
    return 1.0 / clearance;
}

double ORSceneInterface::integrateCosts(const Config& a, const Config& b) const
{
    assert(a.size() == b.size());
    std::vector<double> deltas(a.size());
    double norm = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        deltas.at(i) = a.at(i) - b.at(i);
        norm += deltas.at(i) * deltas.at(i);
    }
    norm = sqrt(norm);
    // iterate over path
    unsigned int num_steps = norm / STEP_SIZE;
    for (size_t t = 0; t < num_steps; ++t) {
        // TODO
    }
}