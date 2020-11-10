#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>

using namespace placement::mp;

MultiGraspMP::~MultiGraspMP() = default;

void MultiGraspMP::savePlanningStats(const std::experimental::filesystem::path& statsfile_path) const
{
  bool file_exists = std::experimental::filesystem::exists(statsfile_path);
  if (not file_exists)
  {
    // ensure all parent folders exist
    std::experimental::filesystem::create_directories(statsfile_path.parent_path());
  }
  std::ofstream fstream(statsfile_path.c_str(), std::ios_base::app);
  utils::ScopedProfiler::dumpProfiles(fstream, not file_exists);
  fstream.close();
}

void MultiGraspMP::saveSolutions(std::vector<Solution>& sols, const std::experimental::filesystem::path& file_path)
{
  // std::experimental::filesystem::path results_file_path(vm["results_file"].as<)
  bool new_file = not std::experimental::filesystem::exists(file_path);
  if (new_file)
  {
    std::experimental::filesystem::create_directories(file_path.parent_path());
  }
  std::ofstream fstream(file_path.c_str(), std::ios_base::app);
  if (new_file)
  {
    fstream << "goal_id,cost" << std::endl;
  }
  for (auto& sol : sols)
  {
    // TODO save path?
    fstream << sol.goal_id << "," << sol.cost << std::endl;
  }
  fstream.close();
}