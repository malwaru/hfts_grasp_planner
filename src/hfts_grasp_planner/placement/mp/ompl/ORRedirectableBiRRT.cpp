#include <hfts_grasp_planner/placement/mp/ompl/ORRedirectableBiRRT.h>
#include <ompl/base/goals/GoalStates.h>
#include <or_ompl/OMPLConversions.h>
#include <or_ompl/OMPLPlannerParameters.h>

using namespace placement::mp::ompl;

ORRedirectableBiRRT::ORRedirectableBiRRT(OpenRAVE::RobotBasePtr probot, OpenRAVE::EnvironmentBasePtr penv)
{
    typedef ::ompl::base::ScopedState<::ompl::base::StateSpace> ScopedState;
    _robot = probot;
    _dirty_goals = true;
    // dummy ompl planner parameters
    or_ompl::OMPLPlannerParameters params;
    // init state space
    _state_space = or_ompl::CreateStateSpace(_robot, params);
    if (!_state_space) {
        RAVELOG_ERROR("Failed creating state space");
        throw std::runtime_error("Failed creating state space");
    }
    // setup simple setup
    _simple_setup.reset(new ::ompl::geometric::SimpleSetup(_state_space));
    auto dof_indices = _robot->GetActiveDOFIndices();
    // setup validity checker
    if (_state_space->isCompound()) {
        _or_validity_checker.reset(new or_ompl::OrStateValidityChecker(
            // TODO what does baked collision checking mean?
            _simple_setup->getSpaceInformation(), _robot, dof_indices, false));
    } else {
        _or_validity_checker.reset(new or_ompl::RealVectorOrStateValidityChecker(
            _simple_setup->getSpaceInformation(), _robot, dof_indices, false));
    }
    _simple_setup->setStateValidityChecker(
        std::static_pointer_cast<::ompl::base::StateValidityChecker>(_or_validity_checker));
    // start validity checker
    _or_validity_checker->start();
    // set start state
    ScopedState q_start(_state_space);
    std::vector<double> vals;
    _robot->GetActiveDOFValues(vals);
    assert(vals.size() == _state_space->getDimension());
    for (size_t i = 0; i < _state_space->getDimension(); ++i) {
        q_start[i] = vals.at(i);
    }
    if (!_or_validity_checker->isValid(q_start.get())) {
        _or_validity_checker->stop();
        throw std::runtime_error("Invalid start state.");
    }
    _simple_setup->addStartState(q_start);
    // create planning algorithm
    _planner = std::make_shared<::ompl::geometric::RedirectableRRTConnect>(_simple_setup->getSpaceInformation(), true);
    _simple_setup->setPlanner(_planner);
    _or_validity_checker->stop();
    _simple_setup->setup();
}

ORRedirectableBiRRT::~ORRedirectableBiRRT() = default;

bool ORRedirectableBiRRT::plan(double timeout, unsigned int& gid)
{
    if (getNumGoals() == 0)
        return false;
    _synchronizeGoals();
    // start validity checker
    _or_validity_checker->start();
    ::ompl::base::PlannerStatus status = _simple_setup->solve(timeout);
    _or_validity_checker->stop();
    return _handlePlanningStatus(status, gid);
}

bool ORRedirectableBiRRT::plan(std::function<bool()> interrupt_fn, unsigned int& gid)
{
    if (getNumGoals() == 0)
        return false;
    _synchronizeGoals();
    // start validity checker
    _or_validity_checker->start();
    ::ompl::base::PlannerStatus status = _simple_setup->solve(interrupt_fn);
    _or_validity_checker->stop();
    return _handlePlanningStatus(status, gid);
}

unsigned int ORRedirectableBiRRT::getNumGoals() const
{
    return _goal_storage.size();
}

void ORRedirectableBiRRT::getPath(unsigned int id, std::vector<std::vector<double>>& path) const
{
    auto iter = _path_storage.find(id);
    if (iter != _path_storage.end()) {
        path = iter->second;
    }
}

void ORRedirectableBiRRT::addGoal(const std::vector<double>& config, unsigned int id)
{
    if (_goal_storage.contains(id)) {
        throw std::logic_error("A goal with id " + std::to_string(id) + " already exists!");
    }
    auto config_with_id = std::make_shared<GoalWithId>(config, id);
    _goal_storage.add(config_with_id);
    _dirty_goals = true;
}

void ORRedirectableBiRRT::removeGoal(unsigned int id)
{
    auto goal = _goal_storage.getGoal(id);
    if (!goal) {
        RAVELOG_WARN("Could not remove goal with id " + std::to_string(id) + ". There is no such goal");
        return;
    }
    _goal_storage.remove(goal);
    _dirty_goals = true;
}

bool ORRedirectableBiRRT::_handlePlanningStatus(::ompl::base::PlannerStatus status, unsigned int& gid)
{
    if (status == ::ompl::base::PlannerStatus::EXACT_SOLUTION) {
        // extract path
        auto ompl_path = _simple_setup->getSolutionPath();
        std::vector<std::vector<double>> path;
        for (size_t i = 0; i < ompl_path.getStateCount(); ++i) {
            std::vector<double> config;
            _state_space->copyToReals(config, ompl_path.getState(i));
            path.push_back(config);
        }
        RAVELOG_DEBUG("New path leads to ");
        _simple_setup->getSpaceInformation()->printState(ompl_path.getState(ompl_path.getStateCount() - 1));
        // get the goal id that we connected
        auto goal = _goal_storage.getGoal(path.back());
        if (!goal) {
            throw std::logic_error("Found a path to a non-existing goal");
        }
        gid = goal->id;
        RAVELOG_DEBUG("Found a path to goal " + std::to_string(gid));
        // remove goal from goal storage so we don't plan to it again
        removeGoal(gid);
        // store path
        _path_storage[gid] = path;
        // reset ompl path, so we can continue searching
        auto pdef = _planner->getProblemDefinition();
        pdef->clearSolutionPaths();
        return true;
    } else if (status == ::ompl::base::PlannerStatus::APPROXIMATE_SOLUTION) {
        // do nothing
        return false;
    } else {
        // this shouldn't happen, so warn the user
        RAVELOG_ERROR("Planner returned %s.\n", status.asString().c_str());
        return false;
    }
}

void ORRedirectableBiRRT::_synchronizeGoals()
{
    if (!_dirty_goals)
        return;
    // init new ompl goals
    auto problem_def = _planner->getProblemDefinition();
    problem_def->clearGoal();
    auto ompl_goals = std::make_shared<::ompl::base::GoalStates>(_simple_setup->getSpaceInformation());
    // first get goals that the planner has already added
    std::vector<::ompl::base::ScopedState<>> in_tree_goals;
    _planner->getGoals(in_tree_goals);
    // determine which of those we have to remove
    std::vector<::ompl::base::ScopedState<>> to_remove;
    // and which we want to keep
    std::unordered_set<unsigned int> to_keep;
    for (auto& goal : in_tree_goals) {
        std::vector<double> config;
        _state_space->copyToReals(config, goal.get());
        auto goal_with_id = _goal_storage.getGoal(config);
        if (goal_with_id == nullptr) {
            to_remove.push_back(goal);
        } else {
            to_keep.insert(goal_with_id->id);
        }
    }
    _planner->removeGoals(to_remove);
    // run over goals in goal storage and add those that aren't in to_keep to ompl goal
    std::vector<std::shared_ptr<GoalWithId>> all_goals;
    _goal_storage.getGoals(all_goals);
    for (auto& g : all_goals) {
        if (to_keep.find(g->id) == to_keep.end()) {
            // add to ompl goal
            ::ompl::base::ScopedState<> state(_state_space);
            _state_space->copyFromReals(state.get(), g->config);
            ompl_goals->addState(state);
        }
    }
    problem_def->setGoal(ompl_goals);
    _dirty_goals = false;
}