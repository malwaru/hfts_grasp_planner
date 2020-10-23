#pragma once
#include <list>
#include <cassert>
#include <hfts_grasp_planner/placement/mp/mgsearch/SearchCommon.h>

/**
 *  Defines LazySP and required data types.
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
 * Defines the interface that a search algorithm needs to provide in order to be compatible with LazySP.
 */
struct SearchAlgorithm
{
  void updateEdges(const std::vector<EdgeChange>& edge_changes);
  void computeShortestPath(SearchResult& result);
};

struct EdgeSelector
{
  /**
   * Select the next edges for evaluation.
   * @param unknown_edges: A list of edges for which the true cost is not known yet.
   * @param edge_iters: Output list of edge iterators to test next.
   */
  void selectNextEdges(std::list<Edge>& unknown_edges, std::list<std::list<Edge>::iterator>& edges);
};

/**
 * FirstUnknownEdgeSelector selects the first unknown edge.
 */
struct FirstUnknownEdgeSelector
{
  void selectNextEdges(std::list<Edge>& unknown_edges, std::list<std::list<Edge>::iterator>& edges)
  {
    assert(not unknown_edges.empty());
    edges.clear();
    edges.push_back(unknown_edges.begin());
  }
};

/**
 * LastUnknownEdgeSelector selects the last unknown edge.
 */
struct LastUnknownEdgeSelector
{
  void selectNextEdges(std::list<Edge>& unknown_edges, std::list<std::list<Edge>::iterator>& edges)
  {
    assert(not unknown_edges.empty());
    edges.clear();
    edges.push_back(std::prev(unknown_edges.end()));
  }
};

/**
 * Compute true edge cost and return it plus the previously best-known cost.
 * This is the default implementation for normal graphs.
 * @param graph: the graph
 * @param v1: first vertex id
 * @param v2: second vertex_id
 * @return: {old_Cost, new_cost}
 */
template <typename G>
std::pair<double, double> getEdgeCost(G& graph, unsigned int v1, unsigned int v2)
{
  double old_cost = graph.getEdgeCost(v1, v2, true);
  double new_cost = graph.getEdgeCost(v1, v2, false);
  return {old_cost, new_cost};
}

/**
 * Check the true cost of the edge from v1 and v2 and add it to edge_changes if there is a change.
 * @param graph: the graph
 * @param v1: first vertex
 * @param v2: second vertex
 * @param edge_changes: vector to store edge changes in
 */
template <typename G>
void checkEdge(G& graph, unsigned int v1, unsigned int v2, std::vector<EdgeChange>& edge_changes)
{
  auto [old_cost, new_cost] = getEdgeCost(graph, v1, v2);
  if (old_cost != new_cost)
  {
    edge_changes.emplace_back(EdgeChange(v1, v2, old_cost < new_cost));
  }
}

/**
 * Check whether the true edge cost from u to v is known.
 * This is the default implementation.
 * @param graph: the graph
 * @param u: first vertex id
 * @param v: second vertex id
 * @return: whether the true cost is known
 */
template <typename G>
bool trueEdgeCostKnown(G& graph, unsigned int u, unsigned int v)
{
  return graph.trueEdgeCostKnown(u, v);
}

/**
 * LazySP algorithm to plan a path on the given graph.
 * Template arguments:
 * type G: the type of the graph
 * type EdgeSelectorType: the type of the edge selector to use. Must fulfill the interface of an EdgeSelector
 * type SearchAlgorithmType: the type of the search algorithm to use. Must provide the interface of a SearchAlgorithm
 * @param graph: The graph to search on.
 * @param algorithm: The search algorithm to use.
 * @param edge_selector: The edge selector to use.
 * @param result: Will contain the search result.
 */
template <typename G, typename EdgeSelectorType, typename SearchAlgorithmType>
void lazySP(G& graph, SearchAlgorithmType& algorithm, EdgeSelectorType& edge_selector, SearchResult& result)
{
  bool all_path_edges_valid = false;
  // repeat as long as we have a path but do not know all of its edge costs
  do
  {
    algorithm.computeShortestPath(result);
    if (result.solved)
    {
      // identify edges for which the true cost is not known yet
      std::list<Edge> unknown_edges;
      unsigned int u = result.path.front();
      for (size_t vidx = 1; vidx < result.path.size(); ++vidx)
      {
        unsigned int v = result.path.at(vidx);
        if (not trueEdgeCostKnown(graph, u, v))
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
          checkEdge(graph, edge_iter->first, edge_iter->second, edge_changes);
          unknown_edges.erase(edge_iter);
        }
      }
      algorithm.updateEdges(edge_changes);
      all_path_edges_valid = edge_changes.empty() and unknown_edges.empty();
    }
  } while (result.solved and not all_path_edges_valid);
}

/**
 * LazySP algorithm to plan a path on the given graph.
 * This is a convenience interface for EdgeSelectorTypes that have a default constructor.
 *
 * Template arguments:
 * type G: the type of the graph
 * type EdgeSelectorType: the type of the edge selector to use. Must fulfill the interface of an EdgeSelector
 * type SearchAlgorithmType: the type of the search algorithm to use. Must provide the interface of a SearchAlgorithm
 *
 * @param graph: The graph to search on.
 * @param algorithm: The search algorithm to use.
 * @param result: Will contain the search result.
 */
template <typename G, typename EdgeSelectorType, typename SearchAlgorithmType>
void lazySP(G& graph, SearchAlgorithmType& algorithm, SearchResult& result)
{
  EdgeSelectorType edge_selector;
  lazySP<G, EdgeSelectorType, SearchAlgorithmType>(graph, result, algorithm, edge_selector);
}

/**
 * LazySP algorithm to plan a path on the given graph.
 * This is a convenience interface for algorithm types that have constructor that can be constructed
 * with the graph as sole argument and EdgeSelectorTypes that have a default constructor.
 *
 * Template arguments:
 * type G: the type of the graph
 * type EdgeSelectorType: the type of the edge selector to use. Must fulfill the interface of an EdgeSelector
 * type SearchAlgorithmType: the type of the search algorithm to use. Must provide the interface of a SearchAlgorithm
 *
 * @param graph: The graph to search on.
 * @param result: Will contain the search result.
 */
template <typename G, typename EdgeSelectorType, typename SearchAlgorithmType>
void lazySP(G& graph, SearchResult& result)
{
  SearchAlgorithmType algorithm(graph);
  EdgeSelectorType edge_selector;
  lazySP<G, EdgeSelectorType, SearchAlgorithmType>(graph, algorithm, edge_selector, result);
}

}  // namespace lazysp
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement