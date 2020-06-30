#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <boost/heap/fibonacci_heap.hpp>

#include <hfts_grasp_planner/placement/mp/mgsearch/SearchCommon.h>

/**
 *  Defines Lifelong Planning A* and needed data structures
 */
namespace placement
{
namespace mp
{
namespace mgsearch
{
namespace lpastar
{
// Key for priority queue
typedef std::pair<double, double> Key;
bool operator<(const Key& a, const Key& b);
// Class that encapsulates LPA* algorithm and needed data structures
template <typename G>
class LPAStarAlgorithm
{
protected:
  /****** Algorithm specific typedefs and structs *******/
  struct PQElement
  {
    unsigned int v;
    Key key;
    PQElement(unsigned int _v, std::pair<double, double> _key) : v(_v), key(_key)
    {
    }
  };

  struct PQElementCompare
  {
    // Does b have priority over a?
    bool operator()(const PQElement& a, const PQElement& b) const
    {
      return b.key < a.key;
    }
  };

  // Priority queue type used by lpastar
  typedef boost::heap::fibonacci_heap<PQElement, boost::heap::compare<PQElementCompare>> PQ;

  // compute key for priority queue
  inline Key computeKey(double g, double h, double rhs)
  {
    double g_p = std::min(g, rhs);
    return {g_p + h, g_p};
  }

  struct VertexData
  {
    const unsigned int v;  // id
    double g;              // g value
    double h;              // heuristic value
    double rhs;            // right-hand-side value of Bellman equation
    unsigned int p;        // parent id
    bool in_pq;            // true if vertex is in PQ
    typename PQ::handle_type pq_handle;
    VertexData(unsigned int v_, double g_, double h_, double rhs_, unsigned int p_)
      : v(v_), g(g_), h(h_), rhs(rhs_), p(p_), in_pq(false)
    {
    }
  };

  typedef std::unordered_map<unsigned int, VertexData> VertexDataMap;

public:
  /**
   * Create a new instance of the LPAStarAlgorithm.
   * @param graph: The graph to operate on. The reference is stored in a member
   * variable. Hence, an instance of this class should only live as long as the
   * graph.
   */
  LPAStarAlgorithm(const G& graph) : _graph(graph)
  {
    unsigned int v_start = _graph.getStartNode();
    // initialize start state
    if (_graph.checkValidity(v_start))
    {
      VertexData start_node(v_start, 0.0, _graph.heuristic(v_start), 0.0, v_start);
      _vertex_data.emplace(std::make_pair(v_start, start_node));
      _vertex_data.at(v_start).pq_handle =
          _pq.push(PQElement(v_start, computeKey(start_node.g, start_node.h, start_node.rhs)));
    }
  }

  ~LPAStarAlgorithm() = default;

  // Struct to communicate a cost change to the algorithm
  struct EdgeChange
  {
    unsigned int u;  // edge goes from u to v
    unsigned int v;
    double old_cost;  // cost prior to the update
  };

  /**
   * Update algorithm state to reflect edge weight changes.
   * @param edge_changes: the edge changes
   */
  void updateEdges(const std::vector<EdgeChange>& edge_changes)
  {
    utils::ScopedProfiler("LPAStarAlgorithm::updateEdges");
    for (const EdgeChange& ec : edge_changes)
    {
      VertexData& u_data = getVertexData(ec.u);
      VertexData& v_data = getVertexData(ec.v);
      double new_cost = _graph.getEdgeCost(ec.u, ec.v);
      if (ec.old_cost > new_cost)  // did edge get cheaper?
      {
        if (u_data.rhs > v_data.g + new_cost)  // update u if it can now be reached at a lower cost
        {
          v_data.p = ec.u;
          v_data.rhs = u_data.g + new_cost;
          updateVertex(ec.v);
        }
      }
      else if (ec.v != _graph.getStartNode() and v_data.p == ec.u)
      {
        // the edge got more expensive and was used to reach v, so look for a new parent of v
        auto [iter, end] = _graph.getPredecessors(ec.v);
        for (; iter < end; ++iter)
        {
          unsigned int s = **iter;
          VertexData& s_data = getVertexData(s);
          double rhs = s_data.g + _graph.getEdgeCost(s, ec.v);
          if (rhs < v_data.rhs)  // v is cheaper to reach through s
          {
            v_data.rhs = rhs;
            v_data.p = s;
          }
        }
        updateVertex(ec.v);
      }
    }
  }

  /**
   * Compute the shortest path given the current algorithm state.
   * @param result: Struct that will contain the search result
   */
  void computeShortestPath(SearchResult& result)
  {
    utils::ScopedProfiler("LPAStarAlgorithm::computeShortestPath");
    unsigned int v_start = _graph.getStartNode();
    // initialize result structure
    result.solved = false;
    result.path.clear();
    result.path_cost = std::numeric_limits<double>::infinity();
    result.goal_cost = std::numeric_limits<double>::infinity();
    result.goal_node = v_start;
    // main iteration
    while (not _pq.empty())  // TODO figure out terminal conditions taking goal cost and multiple goal states into
                             // account
    {
      // TODO implement
      _pq.pop();
    }
    // extract path
    if (result.solved)
    {
      extractPath<VertexDataMap>(v_start, _vertex_data, result);
    }
  }

protected:
  // VertexData
  PQ _pq;
  VertexDataMap _vertex_data;
  const G& _graph;

  void updateVertex(unsigned int v)
  {
    VertexData& v_data = getVertexData(v);
    if (v_data.g != v_data.rhs and v_data.in_pq)
    {
      (*v_data.pq_handle).key = computeKey(v_data.g, v_data.h, v_data.rhs);
      // TODO it's cheaper to update if we know whether the key increases or decreases
      _pq.update(v_data.pq_handle);
    }
    else if (v_data.g != v_data.rhs and not v_data.in_pq)
    {
      // add v to PQ
      v_data.pq_handle = _pq.push(PQElement(v, computeKey(v_data.g, v_data.h, v_data.rhs)));
      v_data.in_pq = true;
    }
    else if (v_data.g == v_data.rhs and v_data.in_pq)
    {
      // v is no longer inconsistent, so remove it from _pq
      _pq.increase(v_data.pq_handle, Key(0.0, 0.0));
      _pq.pop();
      v_data.in_pq = false;
    }
  }

  VertexData& getVertexData(unsigned int v)
  {
    auto iter = _vertex_data.find(v);
    if (iter == _vertex_data.end())
    {
      VertexData new_data(std::numeric_limits<double>::infinity(), _graph.heuristic(v),
                          std::numeric_limits<double>::infinity(), v);
      iter = _vertex_data.emplace(std::make_pair(v, new_data));
    }
    return *iter;
  }
};

/**
 * LPA* search algorithm.
 * The template parameter G needs to be of a type implementing the
 * GraspAgnosticGraph interface specified in Graphs.h.
 * @param graph: The graph to search a path on.
 * @param result: Struct that will contain the search result
 */
template <typename G>
void lpaStarSearch(const G& graph, SearchResult& result)
{
  utils::ScopedProfiler("lpaStarSearch");
  LPAStarAlgorithm algorithm(graph);
  algorithm.computeShortestPath(result);
}

}  // namespace lpastar
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement