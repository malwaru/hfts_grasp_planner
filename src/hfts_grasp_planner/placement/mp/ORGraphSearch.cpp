#include <hfts_grasp_planner/placement/mp/ORGraphSearch.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Algorithms.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>
#include <queue>

namespace mg = ::placement::mp::mgsearch;
using namespace ::placement::mp;

ORGraphSearch::ORGraphSearch(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id,
    const mg::MGGraphSearchMP::Parameters& iparams)
    : _env(penv)
{
    RAVELOG_DEBUG("ORGraphSearch constructor!");
    _scene_interface = std::make_shared<mg::ORStateSpace>(penv, robot_id, obj_id);
    _robot = penv->GetRobot(penv->GetBodyFromEnvironmentId(robot_id)->GetName());
    // get start config
    _robot->GetActiveDOFValues(_start_config);
    // create planner
    _planner = std::make_shared<mg::MGGraphSearchMP>(_scene_interface, _start_config, iparams);
}

ORGraphSearch::~ORGraphSearch()
{
    // nothing to do
}

void ORGraphSearch::plan(std::vector<Solution>& new_paths, double time_limit)
{
    Solution sol;
    if (_planner->plan(sol)) {
        new_paths.push_back(sol);
    }
}

void ORGraphSearch::pausePlanning()
{
    // nothing to do
}

void ORGraphSearch::addGrasp(const Grasp& grasp)
{
    _scene_interface->addGrasp(grasp);
}

void ORGraphSearch::addGoal(const Goal& goal)
{
    _planner->addGoal(goal);
}

void ORGraphSearch::removeGoals(const std::vector<unsigned int>& goal_ids)
{
    _planner->removeGoals(goal_ids);
}
