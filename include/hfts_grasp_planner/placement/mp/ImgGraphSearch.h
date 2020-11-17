#pragma once
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/ImageStateSpace.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MGGraphSearchMP.h>
#include <set>
#include <vector>

namespace placement
{
namespace mp
{
class ImgGraphSearch : public MultiGraspMP
{
public:
  // A wrapper around MGGraphSearchMP for 2d image-based state spaces that implements the MultiGraspMP interface
  ImgGraphSearch(const std::experimental::filesystem::path& root_path, const Config& start,
                 const mgsearch::MGGraphSearchMP::Parameters& params);
  ~ImgGraphSearch();

  void plan(std::vector<Solution>& new_paths, double time_limit) override;
  void pausePlanning() override;
  void addGrasp(const Grasp& grasp) override;
  void addGoal(const Goal& goal) override;
  void removeGoals(const std::vector<unsigned int>& goal_ids) override;
  void addWaypoints(const std::vector<Config>& configs) override;

private:
  mgsearch::MGGraphSearchMPPtr _planner;
  mgsearch::ImageStateSpacePtr _state_space;
};
}  // namespace mp
}  // namespace placement
