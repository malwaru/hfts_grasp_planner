#include <hfts_grasp_planner/placement/mp/mgsearch/LazySP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>

namespace placement
{
namespace mp
{
namespace mgsearch
{
namespace lazysp
{
// Lazy-SP specializations for MultiGraspRoadmapGraph<lazy_grasp_check = True>

/**
 * Compute true edge cost and return it plus the previously best-known cost.
 * This is the default implementation for normal graphs.
 * @param graph: the graph
 * @param v1: first vertex id
 * @param v2: second vertex_id
 * @return: {old_Cost, new_cost}
 */
template <>
inline std::pair<double, double> getEdgeCost(MultiGraspRoadmapGraph<true>& graph, unsigned int v1, unsigned int v2)
{
  double old_cost = graph.getEdgeCost(v1, v2, true);
  double new_cost = graph.getEdgeCostWithGrasp(v1, v2);
  return {old_cost, new_cost};
}

/**
 * Check whether the true edge cost (taking the grasp into account) from u to v is known.
 * @param graph: the graph
 * @param u: first vertex id
 * @param v: second vertex id
 * @return: whether the true cost is known
 */
template <>
inline bool trueEdgeCostKnown(MultiGraspRoadmapGraph<true>& graph, unsigned int u, unsigned int v)
{
  return graph.trueEdgeCostWithGraspKnown(u, v);
}

}  // namespace lazysp
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement