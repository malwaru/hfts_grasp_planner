#include <hfts_grasp_planner/placement/mp/MultiGraspBiRRT.h>

using namespace placement::mp;
using namespace OpenRAVE;

/********************************* SequentialMGBiRRT **************************************/
SequentialMGBiRRT::SequentialMGBiRRT(EnvironmentBasePtr penv,
    unsigned int robot_id,
    unsigned int obj_id)
    : _base_env(penv)
    , _robot_id(robot_id)
    , _obj_id(obj_id)
{
}

SequentialMGBiRRT::~SequentialMGBiRRT() = default;

void SequentialMGBiRRT::plan(std::vector<std::pair<unsigned int, WaypointPath>>& new_paths, double time_limit)
{
    new_paths.clear();
    unsigned int num_active_planners = 0;
    double per_grasp_time = 0.0f;
    if (time_limit == 0.0f)
        time_limit = 5.0f;

    // query first how many grasps with goals we have
    for (auto& key_planner : _planners) {
        num_active_planners += key_planner.second->getNumGoals() > 0;
    }
    if (num_active_planners == 0) {
        return;
    }
    per_grasp_time = time_limit / num_active_planners;
    // plan
    for (auto& key_planner : _planners) {
        WaypointPath path;
        unsigned int gid = 0;
        if (key_planner.second->plan(per_grasp_time, gid)) {
            // we have a new path, so retrieve it
            WaypointPath path;
            key_planner.second->getPath(gid, path);
            new_paths.emplace_back(std::make_pair(gid, path));
        }
    }
}

void SequentialMGBiRRT::addGrasp(const Grasp& grasp)
{
    auto iter = _planners.find(grasp.id);
    if (iter == _planners.end()) {
        // clone base env
        auto new_env = _base_env->CloneSelf(OpenRAVE::Clone_Bodies);
        auto robot = new_env->GetRobot(new_env->GetBodyFromEnvironmentId(_robot_id)->GetName());
        assert(robot != nullptr);
        // release any object in case it is set to grasp any
        robot->ReleaseAllGrabbed();
        auto obj = new_env->GetBodyFromEnvironmentId(_obj_id);
        assert(obj != nullptr);
        // set grasp tf
        auto manip = robot->GetActiveManipulator();
        Transform wTe = manip->GetEndEffectorTransform();
        Transform oTe(grasp.quat, grasp.pos);
        Transform eTo = oTe.inverse();
        Transform wTo = wTe * eTo;
        obj->SetTransform(wTo);
        // set hand_config
        auto gripper_indices = manip->GetGripperIndices();
        robot->SetDOFValues(grasp.gripper_values, 1, gripper_indices);
        // set grasped
        robot->Grab(obj);
        // create a new planner from this env
        _planners[grasp.id] = std::make_shared<ompl::ORRedirectableBiRRT>(robot, new_env);
    } else {
        RAVELOG_WARN("Attempting to add a grasp that has already been added! Ignoring request.");
    }
}

void SequentialMGBiRRT::addGoal(const Goal& goal)
{
    {
        auto iter = _goals.find(goal.id);
        if (iter != _goals.end()) {
            std::string error_msg("Cannot add goal: goal id already exists.");
            RAVELOG_ERROR(error_msg);
            throw std::logic_error(error_msg);
        }
    }
    // first get the planner for the respective grasp
    auto iter = _planners.find(goal.grasp_id);
    if (iter == _planners.end()) {
        std::string error_msg("Could not add the given goal. The corresponding grasp has not been defined");
        RAVELOG_ERROR(error_msg);
        throw std::logic_error(error_msg);
    }
    // add the goal to the respective planner
    iter->second->addGoal(goal.config, goal.id);
    _goals[goal.id] = goal;
}

void SequentialMGBiRRT::removeGoals(const std::vector<unsigned int>& goal_ids)
{
    for (unsigned int gid : goal_ids) {
        auto iter = _goals.find(gid);
        if (iter == _goals.end()) {
            RAVELOG_ERROR("Could not remove goal " + std::to_string(gid) + ". Goal does not exist");
            continue;
        }
        auto& goal = iter->second;
        auto planner_iter = _planners.find(goal.grasp_id);
        if (planner_iter == _planners.end()) {
            throw std::logic_error("There is no planner for goal " + std::to_string(goal.id) + " with grasp id " + std::to_string(goal.grasp_id));
        }
        planner_iter->second->removeGoal(goal.id);
        _goals.erase(goal.id);
    }
}

/********************************* ParallelMGBiRRT **************************************/
ParallelMGBiRRT::ParallelMGBiRRT(EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id)
{
}

ParallelMGBiRRT::~ParallelMGBiRRT() = default;

void ParallelMGBiRRT::plan(std::vector<std::pair<unsigned int, WaypointPath>>& new_paths, double time_limit)
{
}

void ParallelMGBiRRT::addGrasp(const Grasp& grasp)
{
}

void ParallelMGBiRRT::addGoal(const Goal& goal)
{
}

void ParallelMGBiRRT::removeGoals(const std::vector<unsigned int>& goal_ids)
{
}
