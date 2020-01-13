#include <hfts_grasp_planner/placement/mp/ImgGraphSearch.h>

using namespace placement::mp;

ImgGraphSearch::ImgGraphSearch(const std::experimental::filesystem::path& root_path,
    const Config& start, const mgsearch::MGGraphSearchMP::Parameters& params)
{
    _state_space = std::make_shared<mgsearch::ImageStateSpace>(root_path);
    _planner = std::make_shared<mgsearch::MGGraphSearchMP>(_state_space, start, params);
}

ImgGraphSearch::~ImgGraphSearch() = default;

void ImgGraphSearch::plan(std::vector<MultiGraspMP::Solution>& new_paths, double time_limits)
{
    MultiGraspMP::Solution sol;
    _planner->plan(sol);
}

void ImgGraphSearch::pausePlanning()
{
    // do nothing
}

void ImgGraphSearch::addGrasp(const Grasp& grasp)
{
    throw std::logic_error("ImgGraphSearch does not support adding grasps.");
}

void ImgGraphSearch::addGoal(const Goal& goal)
{
    _planner->addGoal(goal);
}

void ImgGraphSearch::removeGoals(const std::vector<unsigned int>& goal_ids)
{
    _planner->removeGoals(goal_ids);
}