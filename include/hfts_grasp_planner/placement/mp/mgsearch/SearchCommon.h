#pragma once
#include <algorithm>
#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>
#include <vector>

/***
 * This header contains data structures and functions that search-based algorithms require.
 */
namespace placement
{
namespace mp
{
namespace mgsearch
{
struct SearchResult
{
  bool solved;
  std::vector<unsigned int> path;  // starting from start node
  double path_cost;
  double goal_cost;
  unsigned int goal_node;
  double cost() const
  {
    return path_cost + goal_cost;
  }
};

// Struct to communicate a cost change
struct EdgeChange
{
  unsigned int u;  // edge goes from u to v
  unsigned int v;
  bool cost_increased;  // true = cost increased, false = cost decreased
  EdgeChange(unsigned int u_, unsigned int v_, bool cost_increased_) : u(u_), v(v_), cost_increased(cost_increased_)
  {
  }
};

typedef std::pair<unsigned int, unsigned int> Edge;

/***
 * Extract the path from result.goal_node to v_start from the given vertex data map.
 *
 * Type expectations:
 * VertexDataMap:
 *  const VertexData& at(unsigned int vidx) : returns a reference to VertexData for vertex vidx
 * VertexData (struct/class):
 *  unsigned int p : a member that stores the parent id of the vertex this data belongs to.
 */
template <typename VertexDataMap>
void extractPath(unsigned int v_start, const VertexDataMap& vertex_data, SearchResult& result)
{
  result.path.clear();
  // extract path
  unsigned int v = result.goal_node;
  while (v != v_start)
  {
    result.path.push_back(v);
    v = vertex_data.at(v).p;
  }
  result.path.push_back(v_start);
  std::reverse(result.path.begin(), result.path.end());
}
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement
