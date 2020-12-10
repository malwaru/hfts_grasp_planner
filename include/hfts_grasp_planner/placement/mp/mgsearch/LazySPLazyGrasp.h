#pragma once
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
// Lazy-SP specializations for MultiGraspRoadmapGraph<cost_checking_type != WithGrasp>

/**
 * Compute true edge cost and return it with the previously best-known cost.
 * This is an override for MutliGraspRoadmapGraph with CostCheckingType::EdgeWithoutGrasp
 * @param graph: the graph
 * @param v1: first vertex id
 * @param v2: second vertex_id
 * @return: {old_Cost, new_cost}
 */
template <>
inline std::pair<double, double> getEdgeCost(MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>& graph,
                                             unsigned int v1, unsigned int v2)
{
  double old_cost = graph.getEdgeCost(v1, v2, true);
  double new_cost = graph.getEdgeCostWithGrasp(v1, v2);
  return {old_cost, new_cost};
}

/**
 * Compute true edge cost and return it with the previously best-known cost.
 * This is an override for MutliGraspRoadmapGraph with CostCheckingType::VertexEdgeWithoutGrasp
 * @param graph: the graph
 * @param v1: first vertex id
 * @param v2: second vertex_id
 * @return: {old_Cost, new_cost}
 */
template <>
inline std::pair<double, double> getEdgeCost(MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>& graph,
                                             unsigned int v1, unsigned int v2)
{
  double old_cost = graph.getEdgeCost(v1, v2, true);
  double new_cost = graph.getEdgeCostWithGrasp(v1, v2);
  return {old_cost, new_cost};
}

/**
 * Check the edge from v1 to v2 and add EdgeChanges to edge_change if they occurred.
 * This is an override for the MultiGraspRoadmapGraph with cost checking type VertexEdgeWithoutGrasp.
 * It can handle the situation that v1 or v2 may found to be invalid in the edge cost computation. If this occurs,
 * all edges into v1 and v2 are added as edge_changes.
 *
 * TODO: the number of reported edge_changes could be reduced if the parent of v1 is known
 */
template <>
inline void checkEdge(MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>& graph, unsigned int v1,
                      unsigned int v2, std::vector<EdgeChange>& edge_changes)
{
  std::vector<unsigned int> v1_neighbors;
  if (!graph.trueValidityWithGraspKnown(v1))
  {  // save v1's neighbors
    graph.getSuccessors(v1, v1_neighbors, true);
  }
  std::vector<unsigned int> v2_neighbors;
  if (!graph.trueValidityWithGraspKnown(v2))
  {  // save v2's neighbors
    graph.getSuccessors(v2, v2_neighbors, true);
  }
  // compute true cost and retrieve whether v1 and v2 are valid
  auto [old_cost, new_cost] = getEdgeCost(graph, v1, v2);
  bool v1_valid = graph.checkValidity(v1);
  bool v2_valid = graph.checkValidity(v2);
  if (!v1_valid)
  {  // add all edges into v1 as edge changes (they are all at infinite cost)
    for (unsigned int n : v1_neighbors)
    {  // we need to inform the algorithm about a bidirectional change
      // edge_changes.emplace_back(n, v1, true);
      edge_changes.emplace_back(v1, n, true);
    }
  }
  if (!v2_valid)
  {  // add all edges into v2 as edge changes (all at infinite cost)
    for (unsigned int n : v2_neighbors)
    {  // we need to inform the algorithm about a bidirectional change
      // edge_changes.emplace_back(n, v2, true);
      edge_changes.emplace_back(v2, n, true);
    }
  }
  if (v1_valid and v2_valid and old_cost != new_cost)
  {  // if vertices are valid, add change depending on actual edge cost change
    edge_changes.emplace_back(EdgeChange(v1, v2, old_cost < new_cost));
  }
}

/**
 * Check whether the true edge cost (taking the grasp into account) from u to v is known.
 * @param graph: the graph
 * @param u: first vertex id
 * @param v: second vertex id
 * @return: whether the true cost is known
 */
template <>
inline bool trueEdgeCostKnown(MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>& graph, unsigned int u,
                              unsigned int v)
{
  return graph.trueEdgeCostWithGraspKnown(u, v);
}

/**
 * Check whether the true edge cost (taking the grasp into account) from u to v is known.
 * @param graph: the graph
 * @param u: first vertex id
 * @param v: second vertex id
 * @return: whether the true cost is known
 */
template <>
inline bool trueEdgeCostKnown(MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>& graph, unsigned int u,
                              unsigned int v)
{
  return graph.trueEdgeCostWithGraspKnown(u, v);
}

}  // namespace lazysp
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement