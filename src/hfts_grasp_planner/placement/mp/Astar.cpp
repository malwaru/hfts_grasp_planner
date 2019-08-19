#include <hfts_grasp_planner/placement/mp/Astar.h>

using namespace placement::mp;

Astar::Astar(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id)
{
    RAVELOG_INFO("Astar constructor!");
}

Astar::~Astar()
{
}

void Astar::plan(std::vector<std::pair<unsigned int, WaypointPathPtr>>& new_paths, double time_limit)
{
    RAVELOG_INFO("Astar plan!");
}

void Astar::pausePlanning()
{
}

void Astar::addGrasp(const Grasp& grasp)
{
}

void Astar::addGoal(const Goal& goal)
{
}

void Astar::removeGoals(const std::vector<unsigned int>& goal_ids)
{
}
