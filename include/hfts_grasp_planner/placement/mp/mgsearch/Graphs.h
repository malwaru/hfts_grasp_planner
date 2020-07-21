#pragma once

#include <boost/functional/hash.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <vector>

namespace placement
{
namespace mp
{
namespace mgsearch
{
/**
 * Type interface for grasp agnostic graphs. Instead of inheriting from this class,
 * simply implement the same interface. All graph search algorithms that use a grasp agnostic graph
 * use this interface.
 */
class GraspAgnosticGraph
{
public:
  struct NeighborIterator
  {
    // Graph-specific forward iterator allowing to iterate over a set of vertices, in particular
    // successors and predecessors
  };
  /**
   * Check the validity of v.
   */
  bool checkValidity(unsigned int v) const;

  /**
   * Return all successor nodes of the node v.
   * @param v - node id to return successor node for
   * @param successors - vector to store successors in
   * @param lazy - if true, no cost evaluation is performed, meaning that a returned successor u may in fact
   *      not be reachable through v, i.e. cost(v, u) could be infinity.
   *      if false, the true cost cost(v, u) is computed first and only us with finite cost are returned.
   */
  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;

  /**
   * Just like above getSuccessors function but returning iterators instead.
   * @param v - node id to return the successors for
   * @param lazy - if true, no cost evaluation is performed, meaning that a returned successor u may in fact
   *      not be reachable through v, i.e. cost(v, u) could be infinity.
   *      if false, the true cost cost(v, u) is computed first and only us with finite cost are returned.
   * @return <begin, end> - begin and end iterators
   */
  std::pair<NeighborIterator, NeighborIterator> getSuccessors(uint v, bool lazy = false) const;

  /**
   * Just like getSuccessors but predecessors. In case of a directed graph, identical to getSuccessors.
   */
  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
  std::pair<NeighborIterator, NeighborIterator> getPredecessors(uint v, bool lazy = false) const;

  /**
   * Get a cost for the edge from v1 to v2. Optionally, a lower bound of the cost.
   * @param v1 - id of first node
   * @param v2 - id of second node
   * @param lazy - if true, return only the best known lower bound cost of the edge, else compute true cost
   */
  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;

  /**
   * Return whether the true cost from v1 to v2 has been computed yet.
   *
   * @param v1: id of the first node
   * @param v2: id of the second node
   * @return true if getEdgeCost(v1, v2, false) has been computed.
   */
  bool trueEdgeCostKnown(unsigned int v1, unsigned int v2) const;

  /**
   * Return the id of the start node.
   */
  unsigned int getStartNode() const;

  /**
   * Return whether the node with the given id is a goal.
   */
  bool isGoal(unsigned int v) const;

  /**
   * Return the additive cost associated with the given goal node.
   * If the given id is not a goal, 0.0 is returned.
   * Note that in general getGoalCost(v) can be greater than heuristic(v), as there may
   * be a better goal with a sufficiently low path cost to be reachable.
   */
  double getGoalCost(uint v) const;

  // Technically not a function of the graph, but the graph might have its own encoding of vertices, so the
  // heuristic needs to be connected to the graph anyways.
  double heuristic(unsigned int v) const;
};

// class GraspAwareGraph {
//     // TODO
// };

/**
 * The SingleGraspRoadmapGraph class implements a view on a MultiGraspRoadmap for a single grasp.
 * that implements the GraspAgnostic graph interface.
 */
class SingleGraspRoadmapGraph
{
public:
  struct NeighborIterator
  {
    ~NeighborIterator() = default;
    NeighborIterator& operator++();
    bool operator==(const NeighborIterator& other) const;
    bool operator!=(const NeighborIterator& other) const;
    uint operator*();
    // iterator traits
    using difference_type = long;
    using value_type = uint;
    using pointer = const uint*;
    using reference = const uint&;
    using iterator_category = std::forward_iterator_tag;

    static NeighborIterator begin(uint v, bool lazy, SingleGraspRoadmapGraph const* parent);
    static NeighborIterator end(uint v, SingleGraspRoadmapGraph const* parent);

  private:
    NeighborIterator(Roadmap::Node::EdgeIterator eiter, Roadmap::Node::EdgeIterator end, bool lazy,
                     SingleGraspRoadmapGraph const* parent);
    SingleGraspRoadmapGraph const* _graph;
    Roadmap::Node::EdgeIterator _iter;
    Roadmap::Node::EdgeIterator _end;
    const bool _lazy;
    void forwardToNextValid();
  };

  friend class NeighborIterator;
  /**
   * Create a new roadmap graph defined by the given roadmap for the given goals.
   * All goals need to correspond to the same grasp (grasp_id).
   * @param roadmap - roadmap to use
   * @param goal_set - goal set - all must belong to the same grasp
   * @param params - Parameters for the goal-path cost trade-off
   * @param grasp_id - the of the grasp
   * @param start_id - the id of the roadmap node that defines the start node
   */
  SingleGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                          ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
                          const ::placement::mp::mgsearch::GoalPathCostParameters& params, unsigned int grasp_id,
                          unsigned int start_id);
  ~SingleGraspRoadmapGraph();
  // GraspAgnostic graph interface
  bool checkValidity(unsigned int v) const;
  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
  std::pair<NeighborIterator, NeighborIterator> getSuccessors(uint v, bool lazy = false) const;
  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
  std::pair<NeighborIterator, NeighborIterator> getPredecessors(uint v, bool lazy = false) const;
  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
  bool trueEdgeCostKnown(unsigned int v1, unsigned int v2) const;
  unsigned int getStartNode() const;
  bool isGoal(unsigned int v) const;
  double getGoalCost(uint v) const;
  double heuristic(unsigned int v) const;

  std::pair<uint, uint> getGraspRoadmapId(uint vid) const;

private:
  ::placement::mp::mgsearch::RoadmapPtr _roadmap;
  ::placement::mp::mgsearch::MultiGraspGoalSetPtr _goal_set;
  ::placement::mp::mgsearch::MultiGoalCostToGo _cost_to_go;
  const unsigned int _grasp_id;
  const unsigned int _start_id;
};

/**
 * The MultiGraspRoadmapGraph class implements a view on a MultiGraspRoadmap for multiple grasps, and
 * implements the GraspAgnostic graph interface.
 * The start vertex of this graph is a special vertex that is not associated with any grasp yet.
 * It is adjacent with cost 0 to #grasps vertices associated with the start configuration - one for each grasp.
 */
class MultiGraspRoadmapGraph
{
public:
  struct NeighborIterator
  {
    ~NeighborIterator() = default;
    NeighborIterator& operator++();
    bool operator==(const NeighborIterator& other) const;
    bool operator!=(const NeighborIterator& other) const;
    uint operator*();
    // iterator traits
    using difference_type = long;
    using value_type = uint;
    using pointer = const uint*;
    using reference = const uint&;
    using iterator_category = std::forward_iterator_tag;

    static NeighborIterator begin(uint v, bool lazy, MultiGraspRoadmapGraph const* graph);
    static NeighborIterator end(uint v, MultiGraspRoadmapGraph const* graph);

  private:
    NeighborIterator(uint v, bool lazy, MultiGraspRoadmapGraph const* parent);
    MultiGraspRoadmapGraph const* _graph;
    uint _v;
    // information about grasps
    std::set<uint>::iterator _grasp_iter;  // for v == 0
    uint _grasp_id;                        // grasp id for any other vertex
    uint _roadmap_id;                      // roadmap node if for any other vertex
    // flag for special case edge back to node 0
    bool _edge_to_0_returned;
    // iterators for roadmap edges
    Roadmap::Node::EdgeIterator _iter;
    Roadmap::Node::EdgeIterator _end;
    bool _lazy;
    void forwardToNextValid();
  };
  /**
   * Create a new MultiGraspRoadmapGraph defined by the given roadmap for the given grasps.
   * @param roadmap - roadmap to use
   * @param goal_set: Set of goals containing goals for the given grasps
   * @param cost_params: Parameters for the goal-path cost tradeoff
   * @param grasp_ids - the ids of the grasps
   * @param start_id - the id of the roadmap node that defines the start node
   */
  MultiGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                         ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
                         const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params,
                         const std::set<unsigned int>& grasp_ids, unsigned int start_id);
  ~MultiGraspRoadmapGraph();
  // GraspAgnostic graph interface
  bool checkValidity(unsigned int v) const;
  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
  std::pair<NeighborIterator, NeighborIterator> getSuccessors(uint v, bool lazy = false) const;
  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
  std::pair<NeighborIterator, NeighborIterator> getPredecessors(uint v, bool lazy = false) const;
  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
  bool trueEdgeCostKnown(unsigned int v1, unsigned int v2) const;
  unsigned int getStartNode() const;
  bool isGoal(unsigned int v) const;
  double getGoalCost(uint v) const;
  double heuristic(unsigned int v) const;

  // roadmap id, grasp id
  std::pair<uint, uint> getGraspRoadmapId(uint vid) const;

private:
  ::placement::mp::mgsearch::RoadmapPtr _roadmap;
  ::placement::mp::mgsearch::MultiGraspGoalSetPtr _goal_set;
  // cost-to-go heuristics for each grasp (grasp id -> heuristic)
  std::unordered_map<unsigned int, ::placement::mp::mgsearch::MultiGoalCostToGoPtr> _individual_cost_to_go;
  // cost-to-go heuristics for all grasps
  ::placement::mp::mgsearch::MultiGoalCostToGo _all_grasps_cost_to_go;
  // grasp ids
  const std::set<unsigned int> _grasp_ids;
  // hash table mapping (grasp_id, roadmap_id) to graph id
  typedef std::pair<unsigned int, unsigned int> GraspNodeIDPair;
  mutable std::unordered_map<GraspNodeIDPair, unsigned int, boost::hash<GraspNodeIDPair>> _roadmap_key_to_graph;
  // hash table mapping graph id to (grasp_id, roadmap_id)
  mutable std::unordered_map<unsigned int, GraspNodeIDPair> _graph_key_to_roadmap;
  unsigned int _roadmap_start_id;

  // grasp id, roadmap id
  std::pair<unsigned int, unsigned int> toRoadmapKey(unsigned int graph_id) const;
  unsigned int toGraphKey(const std::pair<unsigned int, unsigned int>& roadmap_id) const;
  unsigned int toGraphKey(unsigned int grasp_id, unsigned int roadmap_id) const;
  mutable unsigned int _num_graph_nodes;
};
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement