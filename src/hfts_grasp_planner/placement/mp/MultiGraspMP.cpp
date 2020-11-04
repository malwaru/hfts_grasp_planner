#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>

using namespace placement::mp;

MultiGraspMP::~MultiGraspMP() = default;

void MultiGraspMP::savePlanningStats(const std::experimental::filesystem::path& statsfile_path) const
{
  std::ofstream fstream(statsfile_path.c_str());
  utils::ScopedProfiler::dumpProfiles(fstream);
  fstream.close();
}