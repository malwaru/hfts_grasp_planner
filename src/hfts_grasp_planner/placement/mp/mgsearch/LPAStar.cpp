#include <hfts_grasp_planner/placement/mp/mgsearch/LPAStar.h>

using namespace placement::mp::mgsearch::lpastar;

bool placement::mp::mgsearch::lpastar::operator<(const Key& a, const Key& b)
{
  return a.first < b.first || (a.first == b.first and a.second < b.second);
}

bool placement::mp::mgsearch::lpastar::operator<=(const Key& a, const Key& b)
{
  return a.first < b.first || (a.first == b.first and a.second <= b.second);
}