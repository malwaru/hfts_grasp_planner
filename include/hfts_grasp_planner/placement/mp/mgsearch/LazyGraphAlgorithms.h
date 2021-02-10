#pragma once
#include <hfts_grasp_planner/placement/mp/mgsearch/LazySP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/LPAStar.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/LazyGraph.h>

/**
 * This header contains template specializations for LazySP and LPA* to correctly operate with
 * LazyLayeredMultiGraspRoadmapGraph.
 */

namespace placement
{
namespace mp
{
namespace mgsearch
{
namespace lazysp
{
/**
 * Check the true cost of the edge from v1 and v2 and add it to edge_changes if there is a change.
 * @param graph: the graph
 * @param v1: first vertex
 * @param v2: second vertex
 * @param edge_changes: vector to store edge changes in
 */
template <CostCheckingType ctype>
void checkEdge(LazyLayeredMultiGraspRoadmapGraph<ctype>& graph, unsigned int v1, unsigned int v2, unsigned int gid,
               std::vector<EdgeChange>& edge_changes)
{
  std::vector<unsigned int> v1_neighbors;
  if (!graph.isGraspSpecificValidityKnown(v1, gid))
  {  // save v1's neighbors
    graph.getSuccessors(v1, v1_neighbors, true);
  }
  std::vector<unsigned int> v2_neighbors;
  if (!graph.isGraspSpecificValidityKnown(v2, gid))
  {  // save v2's neighbors
    graph.getSuccessors(v2, v2_neighbors, true);
  }
  // compute true cost and retrieve whether v1 and v2 are valid
  double old_cost = graph.getEdgeCost(v1, v2, true);
  double new_cost = graph.getGraspSpecificEdgeCost(v1, v2, gid);

  // check validity of both vertices post edge computation.
  // this validity will reflect the validity with grasp for grasp-specific layers and without grasp on the base layer
  // we only need to add edges to former vertices on grasp-specific layers
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

template <CostCheckingType ctype, typename EdgeSelectorType, typename SearchAlgorithmType>
void lazySPLazyLayered(LazyLayeredMultiGraspRoadmapGraph<ctype>& graph, SearchAlgorithmType& algorithm,
                       EdgeSelectorType& edge_selector, SearchResult& result)
{
  // TODO: VertexEdgeWithoutGrasp probably needs some special treatment (validity check is delayed)
  utils::ScopedProfiler profiler("lazySPLazyLayered");
  bool all_path_edges_valid = false;
  // repeat as long as we have a path but do not know all of its edge costs
  do
  {
    algorithm.computeShortestPath(result);
    if (result.solved)
    {
      assert(result.path.size() > 2);
      // get goal for this solution (to have access to the grasp)
      auto [goal, cost] = graph.getBestGoal(result.path.at(result.path.size() - 2));
      // identify edges for which the true cost is not known yet
      std::list<Edge> unknown_edges;
      unsigned int u = result.path.front();
      for (size_t vidx = 1; vidx < result.path.size(); ++vidx)
      {
        unsigned int v = result.path.at(vidx);
        if (not graph.isGraspSpecificEdgeCostKnown(u, v, goal.grasp_id))
        {
          unknown_edges.push_back(Edge(u, v));
        }
        u = v;
      }
      // query true costs for these edges until we are done or observed a cost increase
      std::vector<EdgeChange> edge_changes;
      while (not unknown_edges.empty() and edge_changes.empty())
      {
        // ask edge selector which edges to evaluate next (might be multiple at once)
        std::list<std::list<Edge>::iterator> edges_to_evaluate;
        edge_selector.selectNextEdges(unknown_edges, edges_to_evaluate);
        for (auto& edge_iter : edges_to_evaluate)
        {
          // evaluate the given edge
          checkEdge(graph, edge_iter->first, edge_iter->second, goal.grasp_id, edge_changes);
          unknown_edges.erase(edge_iter);
        }
      }
      // add additional new edges to edge changes
      graph.getHiddenEdgeChanges(edge_changes, true);
      algorithm.updateEdges(edge_changes);
      all_path_edges_valid = edge_changes.empty() and unknown_edges.empty();
    }
  } while (result.solved and not all_path_edges_valid);
}

template <typename EdgeSelectorType, typename SearchAlgorithmType>
void lazySP(LazyLayeredMultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>& graph,
            SearchAlgorithmType& algorithm, EdgeSelectorType& edge_selector, SearchResult& result)
{
  lazySPLazyLayered<CostCheckingType::EdgeWithoutGrasp, EdgeSelectorType, SearchAlgorithmType>(graph, algorithm,
                                                                                               edge_selector, result);
}

template <typename EdgeSelectorType, typename SearchAlgorithmType>
void lazySP(LazyLayeredMultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>& graph,
            SearchAlgorithmType& algorithm, EdgeSelectorType& edge_selector, SearchResult& result)
{
  lazySPLazyLayered<CostCheckingType::VertexEdgeWithoutGrasp, EdgeSelectorType, SearchAlgorithmType>(
      graph, algorithm, edge_selector, result);
}

template <typename EdgeSelectorType, typename SearchAlgorithmType>
void lazySP(LazyLayeredMultiGraspRoadmapGraph<CostCheckingType::WithGrasp>& graph, SearchAlgorithmType& algorithm,
            EdgeSelectorType& edge_selector, SearchResult& result)
{
  lazySPLazyLayered<CostCheckingType::WithGrasp, EdgeSelectorType, SearchAlgorithmType>(graph, algorithm, edge_selector,
                                                                                        result);
}

template <typename Graph, typename EdgeSelectorType, typename SearchAlgorithmType>
void lazySPLazyLayered(Graph& graph, SearchResult& result)
{
  SearchAlgorithmType algorithm(graph);
  EdgeSelectorType es;
  lazySPLazyLayered(graph, algorithm, es, result);
}
}  // namespace lazysp
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement
