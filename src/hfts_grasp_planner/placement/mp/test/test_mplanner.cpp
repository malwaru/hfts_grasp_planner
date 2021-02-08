#include <algorithm>
#include <boost/functional/hash.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/Algorithms.h>
#include <iostream>
#include <limits>
#include <unordered_map>

/**
 * A simple graph for debugging purposes.
 */
struct DummyGraph
{
  DummyGraph()
  {
    _num_vertices = 10;
    // weights per edges (at least 0.1 for each integer step)
    _edge_weights[{0, 1}] = 0.1;
    _edge_weights[{0, 2}] = 0.3;
    _edge_weights[{0, 3}] = 0.3;
    _edge_weights[{1, 5}] = 0.4;
    _edge_weights[{1, 6}] = 0.5;
    _edge_weights[{2, 4}] = 0.5;
    _edge_weights[{2, 5}] = 0.4;
    _edge_weights[{3, 4}] = 0.1;
    _edge_weights[{3, 6}] = 0.4;
    _edge_weights[{4, 5}] = 0.2;
    _edge_weights[{4, 7}] = 0.3;
    _edge_weights[{5, 6}] = 0.2;
    _edge_weights[{6, 8}] = 0.2;
    _edge_weights[{6, 9}] = 0.4;
    _edge_weights[{7, 8}] = 0.1;
    _edge_weights[{8, 9}] = 0.1;
  }

  ~DummyGraph() = default;
  // GraspAgnostic graph interface
  bool checkValidity(unsigned int v) const
  {
    return true;
  }

  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const
  {
    successors.clear();
    for (unsigned int i = 0; i < _num_vertices; ++i)
    {
      auto iter = _edge_weights.find({std::min(i, v), std::max(i, v)});
      if (iter != _edge_weights.end())
      {
        successors.push_back(i);
      }
    }
  }

  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false)
  {
    getSuccessors(v, predecessors);
  }

  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const
  {
    auto iter = _edge_weights.find({std::min(v1, v2), std::max(v1, v2)});
    if (iter != _edge_weights.end())
    {
      return iter->second;
    }
    return std::numeric_limits<double>::infinity();
  }

  unsigned int getStartVertex() const
  {
    return 0;
  }

  unsigned int getGoalVertex() const
  {
    return _num_vertices - 1;
  }

  bool isGoal(unsigned int v) const
  {
    return v == _num_vertices - 1;
  }

  double getGoalCost(unsigned int v) const
  {
    return 0.0;
  }

  double heuristic(unsigned int v) const
  {
    return (_num_vertices - 1 - v) * 0.1;
  }

  // graph
  typedef std::pair<unsigned int, unsigned int> UIntPair;
  std::unordered_map<UIntPair, double, boost::hash<UIntPair>> _edge_weights;
  unsigned int _num_vertices;
};

int main(int argc, char** argv)
{
  DummyGraph dg;
  placement::mp::mgsearch::SearchResult sr;
  placement::mp::mgsearch::astar::aStarSearch<DummyGraph>(dg, sr);
  // print output path
  std::cout << "Path found: " << std::to_string(sr.solved) << std::endl;
  if (sr.solved)
  {
    std::cout << "Path has cost " << sr.path_cost << " Path is: " << std::endl;
    for (auto v : sr.path)
    {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}