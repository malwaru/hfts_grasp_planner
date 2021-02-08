#include <hfts_grasp_planner/placement/mp/mgsearch/LPAStar.h>

using namespace placement::mp::mgsearch::lpastar;

bool placement::mp::mgsearch::lpastar::operator<(const Key& a, const Key& b)
{
  // TODO figure out what epsilon to use here
  bool equal_key = std::abs(a.first - b.first) < 1e-6;
  return (not equal_key and a.first < b.first) or (equal_key and a.second < b.second);
}

bool placement::mp::mgsearch::lpastar::operator<=(const Key& a, const Key& b)
{
  // TODO figure out what epsilon to use here
  bool equal_key = std::abs(a.first - b.first) < 1e-6;
  return (not equal_key and a.first < b.first) or (equal_key and a.second <= b.second);
  // return a.first < b.first || (std::abs(a.first - b.first) < 1e-6 and a.second <= b.second);
}