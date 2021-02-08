#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

#include <boost/heap/fibonacci_heap.hpp>

#include <hfts_grasp_planner/placement/mp/mgsearch/SearchCommon.h>

/**
 *  This header defines Lazy Weighted A* and required data types.
 */
namespace placement
{
namespace mp
{
namespace mgsearch
{
namespace lwastar
{
struct PQElement
{
  /**
   * Stores information that node v can be reached from node p with at least
   * cost g_value. h_value stores h(v)
   */
  const unsigned int v;  // the node that can be reached
  const unsigned int p;  // the parent
  double g_value;        // g value that v can be reached with if going through p
  double h_value;        // h(v)
  PQElement(unsigned int _v, unsigned int _p, double _g, double _h) : v(_v), p(_p), g_value(_g), h_value(_h)
  {
  }
};

inline double f(const PQElement& el)
{
  return el.g_value + el.h_value;
}

struct PQElementCompare
{
  // for max heaps (should a come after b?)
  bool operator()(const PQElement& a, const PQElement& b) const
  {
    double af = f(a);
    double bf = f(b);
    return af > bf or (af == bf and a.g_value > b.g_value);
  }
};

// used for internal purposes
struct VertexData
{
  /**
   *  Stores whether a node is closed, its parent and its true g value.
   */
  const unsigned int v;
  double g;
  unsigned int p;
  bool closed;
  VertexData(unsigned int _v, double _g, unsigned int _p) : v(_v), g(_g), p(_p)
  {
    closed = false;
  }
};

typedef std::unordered_map<unsigned int, VertexData> VertexDataMap;
typedef std::unordered_map<unsigned int, std::list<PQElement>> WaitingListsMap;

/**
 * Add the given PQ element to the pq or waiting list depending on the heuristic type.
 * If the heuristic is stationary, the element is always added to pq.
 * If the heuristic is non-stationary and new_elemn.h_value is infinite, new_elem is added
 * to waiting_lists as a dependency on the vertex that needs to be explored first
 * for h_value to be computable. If new_elem.h_value is finite, new_elem is added to pq.
 */
template <typename G, typename PQ>
void addToPQ(G& graph, PQ& pq, WaitingListsMap& waiting_lists, const PQElement& new_elem)
{
  if constexpr (G::heuristic_stationary::value)
  {
    pq.push(new_elem);
  }
  else
  {
    // the heuristic may depend on us closing other vertices first. if so add the PQelement to a waiting list
    if constexpr (G::heuristic_vertex_dependency::value)
    {
      if (std::isinf(new_elem.h_value))
      {
        unsigned int dependent_v = graph.getHeuristicDependentVertex(new_elem.v);
        waiting_lists[dependent_v].push_back(new_elem);
      }
      else
      {
        pq.push(new_elem);
      }
    }
    else
    {
      // TODO this function is generally ugly as it is clearly a specialization for FoldedMultiGraspRoadmapGraph and
      // probably nothing else
      throw std::logic_error("Support for graphs with non-stationary heuristic without vertex dependency is not "
                             "implemented for LWA*");
    }
  }
}

/**
 * Adds all PQElements in waiting_lists that are waiting for v to pq.
 * Removes waiting_lists[v] afterwards.
 * Only makes sense for non-stationary heuristic.
 */
template <typename G, typename PQ>
void flushWaitingList(G& graph, PQ& pq, WaitingListsMap& waiting_lists, unsigned int v)
{
  static_assert(not G::heuristic_stationary::value and G::heuristic_vertex_dependency::value);
  auto iter = waiting_lists.find(v);
  if (iter != waiting_lists.end())
  {
    for (auto& pq_elem : iter->second)
    {
      pq_elem.h_value = graph.heuristic(pq_elem.v);
      assert(not std::isinf(pq_elem.h_value));
      pq.push(pq_elem);
    }
    waiting_lists.erase(iter);
  }
}
/**
 * LWA* search algorithm.
 * The template parameter G needs to be of a type implementing the GraspAgnosticGraph interface specified in Graphs.h.
 * The template parameter PQ needs to be a boost::heap // TODO we do not use an addressable PQ here, so check what the
 * most efficient one is and use that
 */
template <typename G, typename PQ = boost::heap::fibonacci_heap<PQElement, boost::heap::compare<PQElementCompare>>>
void lwaStarSearch(G& graph, SearchResult& result)
{
  utils::ScopedProfiler profiler("lwaStarSearch");
  unsigned int v_start = graph.getStartVertex();
  unsigned int v_goal = graph.getGoalVertex();
  // initialize result structure
  result.solved = false;
  result.path.clear();
  result.path_cost = std::numeric_limits<double>::infinity();
  result.goal_cost = std::numeric_limits<double>::infinity();  // TODO obsolete
  result.goal_node = v_start;
  // initialize algorithm data structures
  PQ pq;
  VertexDataMap vertex_data;
  WaitingListsMap waiting_lists;  // waiting list for PQElements for which we don't have stationary h-values yet (in
                                  // case h is adaptive)
  if (graph.checkValidity(v_start))
  {
    vertex_data.emplace(std::make_pair(v_start, VertexData(v_start, 0.0, v_start)));
    pq.push(PQElement(v_start, v_start, 0.0, graph.heuristic(v_start)));
  }
  // add goal vertex
  vertex_data.emplace(std::make_pair(v_goal, VertexData(v_goal, std::numeric_limits<double>::infinity(), v_goal)));
  // main iteration - is skipped if start vertex is invalid
  while (not pq.empty() and not vertex_data.at(v_goal).closed)
  {
    PQElement current_el = pq.top();
    pq.pop();
    // check whether we already extended this node and check its validity; skip if necessary
    if (vertex_data.at(current_el.v).closed or not graph.checkValidity(current_el.v))
    {
      vertex_data.at(current_el.v).closed = true;
      continue;
    }
    // check whether current_el is ready for extension, i.e. whether we have the true g
    double true_edge_cost = current_el.p != current_el.v ? graph.getEdgeCost(current_el.p, current_el.v, false) : 0.0;
    double old_g = current_el.g_value;
    current_el.g_value = vertex_data.at(current_el.p).g + true_edge_cost;
    // we can extend if there is no change, else we should add the element back if the new g is finite
    if (old_g != current_el.g_value)
    {
      if (not std::isinf(current_el.g_value))
      {
        pq.push(current_el);
      }
    }
    else
    {  // we can extend v
      vertex_data.at(current_el.v).closed = true;
      vertex_data.at(current_el.v).p = current_el.p;
      vertex_data.at(current_el.v).g = current_el.g_value;
      if constexpr (not G::heuristic_stationary::value)
      {  // add dependent PQElements to PQ in case we have a non-stationary heuristic type
        // register minimal cost
        graph.registerMinimalCost(current_el.v, current_el.g_value);
        if constexpr (G::heuristic_vertex_dependency::value)
        {
          flushWaitingList(graph, pq, waiting_lists, current_el.v);
        }
      }
      if (current_el.v != v_goal)
      {  // extend v if its not the goal
        auto [siter, send] = graph.getSuccessors(current_el.v, true);
        for (; siter != send; ++siter)
        {
          unsigned int s = *siter;
          // get lower bound of edge cost
          double wvs = graph.getEdgeCost(current_el.v, s, true);
          // TODO is this needed? getSuccessors should skip invalid edges
          // if (std::isinf(wvs))
          // {  // skip edges that are already known to be invalid
          //   continue;
          // }
          // compute the g value s might reach by going through v
          double g_s = current_el.g_value + wvs;
          // get VertexData
          auto iter = vertex_data.find(s);
          if (iter == vertex_data.end())
          {
            // create a VertexData element if it doesn't exist yet.
            bool valid;
            std::tie(iter, valid) = vertex_data.emplace(std::make_pair(s, VertexData(s, g_s, current_el.v)));
            assert(valid);
          }
          else if (iter->second.closed)
          {
            // if its closed, we can skip
            continue;
          }
          // in any case, add a new pq element representing the possibility to reach s from v
          addToPQ(graph, pq, waiting_lists, PQElement(s, current_el.v, g_s, graph.heuristic(s)));
        }
      }
    }
  }
  // extract path
  if (vertex_data.at(v_goal).closed)
  {
    result.solved = true;
    result.path_cost = vertex_data.at(v_goal).g;
    result.goal_node = v_goal;
    result.goal_cost = 0.0;
    extractPath<VertexDataMap>(v_start, vertex_data, result);
  }
}
}  // namespace lwastar
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement