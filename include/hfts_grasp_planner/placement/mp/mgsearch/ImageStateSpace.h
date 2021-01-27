#pragma once
#include <experimental/filesystem>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <unordered_map>

namespace placement
{
namespace mp
{
namespace mgsearch
{
class ImageStateSpace : public StateSpace
{
public:
  /**
   * Create a new image state space from numpy arrays stored in the file system in the given folder.
   * The folder is expected to contain files of name <id>.npy, where <id> denotes the grasp id.
   * The numpy array with name 0.npy is considered to be the base case, i.e. the state space of
   * only the robot without any grasp. The remaining files represent the state space of the robot
   * with grasp <id> - 1.
   * Each image is expected to store state-based costs, where a state should have infinite cost if it is
   * unreachable and otherwise a finite cost >= 1.
   */
  ImageStateSpace(const std::experimental::filesystem::path& root_path);
  ~ImageStateSpace();
  // State validity
  bool isValid(const Config& c) const override;
  bool isValid(const Config& c, unsigned int grasp_id, bool only_obj = false) const override;
  // state cost
  double cost(const Config& a) const override;
  double cost(const Config& a, unsigned int grasp_id) const override;
  // distance
  double distance(const Config& a, const Config& b) const override;
  // space information
  unsigned int getDimension() const override;
  void getBounds(Config& lower, Config& upper) const override;
  // grasp query
  void getValidGraspIds(std::vector<unsigned int>& grasp_ids) const override;
  unsigned int getNumGrasps() const override;

protected:
  struct Image
  {
    unsigned int width;
    unsigned int height;
    std::vector<double> data;
    Image() : width(0), height(0)
    {
    }
    Image(unsigned int w, unsigned int h, const std::vector<double>& d) : width(w), height(h), data(d)
    {
    }
    ~Image() = default;

    double& operator()(unsigned int i, unsigned int j)
    {
      return data.at(i + j * width);
    }

    const double& operator()(unsigned int i, unsigned int j) const
    {
      return data.at(i + j * width);
    }

    double& at(unsigned int i, unsigned int j)
    {
      assert(not data.empty());
      assert(i < width && j < height);
      return operator()(i, j);
    }

    const double& at(unsigned int i, unsigned int j) const
    {
      assert(not data.empty());
      assert(i < width && j < height);
      return operator()(i, j);
    }
  };
  typedef std::shared_ptr<Image> ImagePtr;

  std::vector<ImagePtr> _images;

  void init(const std::experimental::filesystem::path& root_path);
};

typedef std::shared_ptr<ImageStateSpace> ImageStateSpacePtr;
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement
