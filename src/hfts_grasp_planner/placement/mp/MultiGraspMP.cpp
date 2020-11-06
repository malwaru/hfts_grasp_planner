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