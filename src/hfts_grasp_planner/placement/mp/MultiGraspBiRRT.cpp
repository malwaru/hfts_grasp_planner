// we use
#include <boost/thread/locks.hpp>
#include <functional>
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

SequentialMGBiRRT::~SequentialMGBiRRT()
{
    _base_env->Destroy();
}

void SequentialMGBiRRT::plan(std::vector<Solution>& new_paths, double time_limit)
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
        unsigned int gid = 0;
        if (key_planner.second->plan(per_grasp_time, gid)) {
            // we have a new path, so retrieve it
            WaypointPathPtr path_ptr = std::make_shared<WaypointPath>();
            WaypointPath& path = *path_ptr;
            key_planner.second->getPath(gid, path);
            new_paths.emplace_back(Solution(gid, path_ptr, 0.0)); // TODO could store path length as cost here
            _goals.erase(gid);
        }
    }
}

void SequentialMGBiRRT::pausePlanning()
{
    // nothing to do, we only plan in plan()
}

void SequentialMGBiRRT::addGrasp(const Grasp& grasp)
{
    auto iter = _planners.find(grasp.id);
    if (iter == _planners.end()) {
        // clone base env
        boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_base_env->GetMutex());
        auto new_env = _base_env->CloneSelf(OpenRAVE::Clone_Bodies);
        new_env->StopSimulation();
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
        {
            std::stringstream ss;
            ss << "Name of the manipulator " << manip->GetName();
            std::vector<double> dof_values;
            robot->GetDOFValues(dof_values, manip->GetArmIndices());
            ss << " DoF values: ";
            for (auto v : dof_values) {
                ss << v << ", ";
            }
            RAVELOG_DEBUG(ss.str());
            ss.str("");
            ss << "quat: " << eTo.rot;
            ss << "trans: " << eTo.trans;
            RAVELOG_DEBUG("Object pose in eef frame " + ss.str());
            ss.str("");
            ss << "quat: " << oTe.rot;
            ss << "trans: " << oTe.trans;
            RAVELOG_DEBUG("EEF pose in object frame " + ss.str());
            ss.str("");
            ss << "quat: " << wTe.rot;
            ss << "trans: " << wTe.trans;
            RAVELOG_DEBUG("EEF pose in world frame " + ss.str());
            ss.str("");
            ss << "quat: " << wTo.rot;
            ss << "trans: " << wTo.trans;
            RAVELOG_DEBUG("Setting object to pose " + ss.str());
        }
        // set hand_config
        auto gripper_indices = manip->GetGripperIndices();
        robot->SetDOFValues(grasp.gripper_values, 1, gripper_indices);
        // set grasped
        robot->Grab(obj);
        // set arm dofs
        robot->SetActiveDOFs(manip->GetArmIndices());
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

/********************************* ParallelMGBiRRT::AsynchPlanner **************************************/
ParallelMGBiRRT::AsynchPlanner::AsynchPlanner(ompl::ORRedirectableBiRRTPtr planner)
    : _planner(planner)
    , _terminate(false)
    , _paused(false)
{
    _thread = std::thread(std::bind(&AsynchPlanner::run, this));
}

ParallelMGBiRRT::AsynchPlanner::~AsynchPlanner()
{
    _terminate = true; // tell motion planner to abort in case it is planning
    _goal_modification_cv.notify_all(); // and wake it up in case it is waiting for goals
    _thread.join();
}

void ParallelMGBiRRT::AsynchPlanner::addGoal(const Goal& goal)
{
    {
        std::lock_guard<std::mutex> lock(_goal_modification_mutex);
        _goals_to_add.push_back(goal);
    }
    _goal_modification_cv.notify_one();
}

void ParallelMGBiRRT::AsynchPlanner::addGoals(const std::vector<Goal>& goals)
{
    {
        std::lock_guard<std::mutex> lock(_goal_modification_mutex);
        _goals_to_add.insert(_goals_to_add.end(), goals.begin(), goals.end());
    }
    _goal_modification_cv.notify_one();
}

void ParallelMGBiRRT::AsynchPlanner::removeGoal(unsigned int goal_id)
{
    std::lock_guard<std::mutex> lock(_goal_modification_mutex);
    _goals_to_remove.push_back(goal_id);
}

void ParallelMGBiRRT::AsynchPlanner::removeGoals(const std::vector<unsigned int>& goals)
{
    std::lock_guard<std::mutex> lock(_goal_modification_mutex);
    _goals_to_remove.insert(_goals_to_remove.end(), goals.begin(), goals.end());
}

void ParallelMGBiRRT::AsynchPlanner::getNewPaths(std::vector<Solution>& paths)
{
    std::lock_guard<std::mutex> lock(_path_list_mutex);
    paths.insert(paths.end(), _new_paths.begin(), _new_paths.end());
    _new_paths.clear();
}

void ParallelMGBiRRT::AsynchPlanner::run()
{
    std::stringstream ss;
    auto my_id = std::this_thread::get_id();
    ss << my_id;
    std::string thread_id = ss.str();
    RAVELOG_DEBUG("Launching thread " + thread_id);
    while (not _terminate) {
        // synchronize goals
        {
            std::lock_guard<std::mutex> lock(_goal_modification_mutex);
            for (const Goal& goal : _goals_to_add) {
                RAVELOG_DEBUG("Thread " + thread_id + " adds goal " + std::to_string(goal.id));
                _planner->addGoal(goal.config, goal.id);
            }
            _goals_to_add.clear();
            for (unsigned int id : _goals_to_remove) {
                RAVELOG_DEBUG("Thread " + thread_id + " removes goal " + std::to_string(id));
                _planner->removeGoal(id);
            }
            _goals_to_remove.clear();
        }
        // if we have goals left, plan
        if (_planner->getNumGoals() > 0 and not _paused) {
            // we plane for some time or until termination is requested
            // std::time_t end_time_t = std::chrono::steady_clock::to_time_t(end_point);
            // RAVELOG_DEBUG("Thread " + thread_id + " plans until " << std::put_time(std::local_time(end_time_t), "%F, %T"));
            RAVELOG_DEBUG("Thread " + thread_id + " plans");
            std::chrono::steady_clock::time_point end_point = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
            std::function<bool()> interrupt_fn = [this, end_point] { return _terminate || std::chrono::steady_clock::now() > end_point; };
            unsigned int gid = 0;
            if (_planner->plan(interrupt_fn, gid)) {
                RAVELOG_DEBUG("Thread " + thread_id + " found a new path to " + std::to_string(gid));
                WaypointPathPtr new_path = std::make_shared<WaypointPath>();
                WaypointPath& tpath = *new_path;
                _planner->getPath(gid, tpath);
                std::lock_guard<std::mutex> lock(_path_list_mutex);
                _new_paths.push_back(Solution(gid, new_path, 0.0)); // TODO could provide true path cost
            }
        } else {
            // sleep until further notice
            RAVELOG_DEBUG("Thread " + thread_id + " either has no goal or was asked to pause, going to sleep");
            std::unique_lock<std::mutex> lock(_goal_modification_mutex);
            _goal_modification_cv.wait(lock);
            lock.unlock();
            RAVELOG_DEBUG("Thread " + thread_id + " woke up.");
            // in case we are awoken spuriously, we will just go to sleep in the next iteration again
        }
    }
}

void ParallelMGBiRRT::AsynchPlanner::pause(bool bpause)
{
    bool before_paused = _paused;
    _paused = bpause;
    if (!bpause and before_paused) {
        _goal_modification_cv.notify_all(); // wake thread up in case it was asleep
    }
}

/********************************* ParallelMGBiRRT **************************************/
ParallelMGBiRRT::ParallelMGBiRRT(EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id)
    : _base_env(penv)
    , _robot_id(robot_id)
    , _obj_id(obj_id)
{
}

ParallelMGBiRRT::~ParallelMGBiRRT()
{
    _base_env->Destroy();
}

void ParallelMGBiRRT::plan(std::vector<Solution>& new_paths, double time_limit)
{
    new_paths.clear();
    if (not _goals.empty()) {
        // wake up all threads in case they were paused
        for (auto& key_planner : _planners) {
            key_planner.second->pause(false);
        }
        // sleep for the duration of time_limit to give asynchronous motion planners some time
        std::this_thread::sleep_for(std::chrono::duration<double>(time_limit));
        // collect paths
        for (auto& key_planner : _planners) {
            std::vector<Solution> paths_for_grasp;
            key_planner.second->getNewPaths(paths_for_grasp);
            // filter goals that we are not longer interested in
            for (auto& sol : paths_for_grasp) {
                if (_goals.find(sol.goal_id) != _goals.end()) {
                    new_paths.push_back(sol);
                    _goals.erase(sol.goal_id);
                } // else ignore
            }
        }
    }
}

void ParallelMGBiRRT::pausePlanning()
{
    // pause all planners
    for (auto& key_planner : _planners) {
        key_planner.second->pause(true);
    }
}

void ParallelMGBiRRT::addGrasp(const Grasp& grasp)
{
    auto iter = _planners.find(grasp.id);
    if (iter == _planners.end()) {
        // clone base env
        boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_base_env->GetMutex());
        auto new_env = _base_env->CloneSelf(OpenRAVE::Clone_Bodies);
        new_env->StopSimulation();
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
        // set arm dofs
        robot->SetActiveDOFs(manip->GetArmIndices());
        // create a new planner from this env
        _planners[grasp.id] = std::make_shared<AsynchPlanner>(std::make_shared<ompl::ORRedirectableBiRRT>(robot, new_env));
    } else {
        RAVELOG_WARN("Attempting to add a grasp that has already been added! Ignoring request.");
    }
}

void ParallelMGBiRRT::addGoal(const Goal& goal)
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
    iter->second->addGoal(goal);
    _goals[goal.id] = goal;
}

void ParallelMGBiRRT::removeGoals(const std::vector<unsigned int>& goal_ids)
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
