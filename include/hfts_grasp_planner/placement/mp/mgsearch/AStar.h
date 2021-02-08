#pragma once

#include <algorithm>
#include <cmath>

#include <boost/heap/fibonacci_heap.hpp>

#include <hfts_grasp_planner/placement/mp/mgsearch/SearchCommon.h>

/***
 *  Defines and implements templated A* algorithm and required data types.
 */
namespace placement
{
namespace mp
{
namespace mgsearch
{
namespace astar
{
struct PQElement
{
  unsigned int v;
  double g_value;
  double h_value;
  PQElement(unsigned int _v, double _g, double _h) : v(_v), g_value(_g), h_value(_h)
  {
  }
};

inline double f(const PQElement& el)
{
  return el.g_value + el.h_value;
}

struct PQElementCompare
{
  // for max heaps
  bool operator()(const PQElement& a, const PQElement& b) const
  {
    double af = f(a);
    double bf = f(b);
    return af > bf or (af == bf and a.g_value > b.g_value);
  }
};

// used for internal purposes
template <typename PQ>
struct VertexData
{
  const unsigned int v;
  double g;
  unsigned int p;
  typename PQ::handle_type pq_handle;
  bool closed;
  VertexData(unsigned int _v, double _g, unsigned int _p) : v(_v), g(_g), p(_p)
  {
    closed = false;
  }
};

/**
 * A* search algorithm.
 * The template parameter G needs to be of a type implementing the GraspAgnosticGraph interface specified in Graphs.h
 * with a stationary heuristic.
 * The template parameter PQ needs to be a boost::heap
 */
template <typename G, typename PQ = boost::heap::fibonacci_heap<PQElement, boost::heap::compare<PQElementCompare>>>
void aStarSearch(G& graph, SearchResult& result)
{
  static_assert(G::heuristic_stationary::value);
  utils::ScopedProfiler profiler("aStarSearch");
  // get start and goal vertices
  unsigned int v_start = graph.getStartVertex();
  unsigned int v_goal = graph.getGoalVertex();

  // initialize result structure
  result.solved = false;
  result.path.clear();
  result.path_cost = std::numeric_limits<double>::infinity();
  result.goal_cost = std::numeric_limits<double>::infinity();
  result.goal_node = v_start;
  // initialize algorithm data structures
  PQ pq;
  typedef std::unordered_map<unsigned int, VertexData<PQ>> VertexDataMap;
  VertexDataMap vertex_data;
  if (graph.checkValidity(v_start))
  {
    vertex_data.emplace(std::make_pair(v_start, VertexData<PQ>(v_start, 0.0, v_start)));
    vertex_data.at(v_start).pq_handle = pq.push(PQElement(v_start, 0.0, graph.heuristic(v_start)));
  }
  vertex_data.emplace(std::make_pair(v_goal, VertexData<PQ>(v_goal, std::numeric_limits<double>::infinity(), v_goal)));
  vertex_data.at(v_goal).pq_handle = pq.push(PQElement(v_goal, std::numeric_limits<double>::infinity(), 0.0));
  // main iteration - is skipped if start vertex is invalid
  while (not pq.empty() and vertex_data.at(v_goal).g > f(pq.top()))
  {
    PQElement current_el = pq.top();
    pq.pop();
    vertex_data.at(current_el.v).closed = true;
    auto [siter, send] = graph.getSuccessors(current_el.v, false);
    for (; siter != send; ++siter)
    {
      unsigned int s = *siter;
      // check vertex and edge validity
      double wvs = graph.getEdgeCost(current_el.v, s, false);
      if (std::isinf(wvs))
      {
        continue;
      }
      // s is reachable from v. compute the g value it can reach.
      double g_s = current_el.g_value + wvs;
      // create a VertexData element if it doesn't exist yet.
      auto iter = vertex_data.find(s);
      if (iter != vertex_data.end())
      {
        if (iter->second.closed)
          continue;
        // s has been reached from another node before, check whether we can decrease its key
        if (iter->second.g > g_s)
        {
          iter->second.g = g_s;
          iter->second.p = current_el.v;
          (*(iter->second.pq_handle)).g_value = g_s;
          pq.increase(iter->second.pq_handle);  // increase priority
        }
      }
      else
      {
        // s hasn't been reached before, add a new VertexData element and push it to pq
        auto [iter, valid] = vertex_data.emplace(std::make_pair(s, VertexData<PQ>(s, g_s, current_el.v)));
        assert(valid);
        iter->second.pq_handle = pq.push(PQElement(s, g_s, graph.heuristic(s)));
      }
    }
  }
  result.solved = not std::isinf(vertex_data.at(v_goal).g);
  result.path_cost = vertex_data.at(v_goal).g;
  result.goal_cost = 0.0;  // TODO obsolete
  result.goal_node = v_goal;
  // extract path
  if (result.solved)
  {
    extractPath<VertexDataMap>(v_start, vertex_data, result);
  }
}
}  // namespace astar
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement