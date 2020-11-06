#include <Eigen/Core>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/ORStateSpace.h>
#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>

#define SAFE_DISTANCE 0.05

namespace pmp = ::placement::mp;
using namespace placement::mp::mgsearch;

double ::placement::mp::mgsearch::cSpaceDistance(const pmp::Config& a, const pmp::Config& b)
{
  const Eigen::Map<const Eigen::VectorXd> avec(a.data(), a.size());
  const Eigen::Map<const Eigen::VectorXd> bvec(b.data(), b.size());
  return (avec - bvec).norm();
}

ORStateSpace::ORStateSpace(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id)
  : _penv(penv), _distance_check_enabled(false)
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

ORStateSpace::~ORStateSpace() = default;

void ORStateSpace::addGrasp(const MultiGraspMP::Grasp& g)
{
  auto iter = _grasps.find(g.id);
  if (iter != _grasps.end())
  {
    RAVELOG_ERROR("New grasp has the same id as a previously added grasp. Can not add new grasp");
    throw std::logic_error("New grasp has the same id as a previously added grasp. Can not add new grasp");
  }
  _grasps.insert(std::make_pair(g.id, g));
}

void ORStateSpace::removeGrasp(unsigned int gid)
{
  auto iter = _grasps.find(gid);
  if (iter != _grasps.end())
  {
    _grasps.erase(iter);
  }
  else
  {
    RAVELOG_WARN("Trying to remove grasp that doesn't exist");
  }
}

bool ORStateSpace::isValid(const pmp::Config& c) const
{
  pmp::utils::ScopedProfiler profiler("StateSpace::isValid");
  boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
  OpenRAVE::RobotBase::RobotStateSaver state_saver(_robot);
  OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
  _robot->ReleaseAllGrabbed();
  _robot->SetActiveDOFValues(c);
  _object->Enable(false);
  enableDistanceCheck(false);
  if (_robot->CheckSelfCollision())
  {
    return false;
  }
  return !_penv->CheckCollision(_robot);
}

bool ORStateSpace::isValid(const pmp::Config& c, unsigned int grasp_id, bool only_obj) const
{
  pmp::utils::ScopedProfiler profiler("StateSpace::isValidWithGrasp");
  boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
  OpenRAVE::RobotBase::RobotStateSaver state_saver(_robot);
  OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
  setGrasp(grasp_id);
  _robot->SetActiveDOFValues(c);
  // kinbody should be part of the robot now
  bool bvalid = false;
  enableDistanceCheck(false);
  if (only_obj)
  {
    bvalid = !_penv->CheckCollision(_object);
  }
  else
  {
    bvalid = !_robot->CheckSelfCollision() and !_penv->CheckCollision(_robot);
  }
  _robot->ReleaseAllGrabbed();
  return bvalid;
}

void ORStateSpace::setGrasp(unsigned int gid) const
{
  auto iter = _grasps.find(gid);
  if (iter == _grasps.end())
  {
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
  _object->Enable(true);
  // set hand_config
  auto gripper_indices = manip->GetGripperIndices();
  _robot->SetDOFValues(grasp.gripper_values, 1, gripper_indices);
  // set grasped
  _robot->Grab(_object);
}

double ORStateSpace::cost(const pmp::Config& c) const
{
  pmp::utils::ScopedProfiler profiler("StateSpace::cost");
  boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
  OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
  _object->Enable(false);
  return computeCost(c);
}

double ORStateSpace::cost(const pmp::Config& c, unsigned int grasp_id) const
{
  pmp::utils::ScopedProfiler profiler("StateSpace::costWithGrasp");
  boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
  OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
  setGrasp(grasp_id);
  _object->Enable(true);
  double val = computeCost(c);
  _robot->ReleaseAllGrabbed();
  return val;
}

unsigned int ORStateSpace::getDimension() const
{
  return _robot->GetActiveDOF();
}

void ORStateSpace::getBounds(Config& lower, Config& upper) const
{
  _robot->GetActiveDOFLimits(lower, upper);
}

void ORStateSpace::getValidGraspIds(std::vector<unsigned int>& grasp_ids) const
{
  grasp_ids.clear();
  for (auto elem : _grasps)
  {
    grasp_ids.push_back(elem.first);
  }
}

unsigned int ORStateSpace::getNumGrasps() const
{
  return _grasps.size();
}

double ORStateSpace::distance(const Config& a, const Config& b) const
{
  return cSpaceDistance(a, b);
}

void ORStateSpace::enableDistanceCheck(bool enable) const
{
  if (enable and not _distance_check_enabled)
  {
    _col_checker->SetCollisionOptions(OpenRAVE::CollisionOptions::CO_Distance |
                                      OpenRAVE::CollisionOptions::CO_ActiveDOFs);
  }
  else if (not enable and _distance_check_enabled)
  {
    _col_checker->SetCollisionOptions(OpenRAVE::CollisionOptions::CO_ActiveDOFs);
  }
  _distance_check_enabled = enable;
}

double ORStateSpace::computeCost(const Config& c) const
{
  boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_penv->GetMutex());
  OpenRAVE::RobotBase::RobotStateSaver rob_state_saver(_robot);
  OpenRAVE::KinBody::KinBodyStateSaver obj_state_saver(_object);
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
