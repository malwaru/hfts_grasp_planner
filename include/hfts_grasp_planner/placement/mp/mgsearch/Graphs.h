#pragma once

#include <boost/functional/hash.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>
#include <vector>
#include <set>

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
  bool checkValidity(unsigned int v);

  /**
   * Return all successor nodes of the node v.
   * @param v - node id to return successor node for
   * @param successors - vector to store successors in
   * @param lazy - if true, no cost evaluation is performed, meaning that a returned successor u may in fact
   *      not be reachable through v, i.e. cost(v, u) could be infinity.
   *      if false, the true cost cost(v, u) is computed first and only us with finite cost are returned.
   */
  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false);

  /**
   * Just like above getSuccessors function but returning iterators instead.
   * @param v - node id to return the successors for
   * @param lazy - if true, no cost evaluation is performed, meaning that a returned successor u may in fact
   *      not be reachable through v, i.e. cost(v, u) could be infinity.
   *      if false, the true cost cost(v, u) is computed first and only us with finite cost are returned.
   * @return <begin, end> - begin and end iterators
   */
  std::pair<NeighborIterator, NeighborIterator> getSuccessors(unsigned int v, bool lazy = false);

  /**
   * Just like getSuccessors but predecessors. In case of a directed graph, identical to getSuccessors.
   */
  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false);
  std::pair<NeighborIterator, NeighborIterator> getPredecessors(unsigned int v, bool lazy = false);

  /**
   * Get a cost for the edge from v1 to v2. Optionally, a lower bound of the cost.
   * @param v1 - id of first node
   * @param v2 - id of second node
   * @param lazy - if true, return only the best known lower bound cost of the edge, else compute true cost
   */
  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false);

  /**
   * Return whether the true cost from v1 to v2 has been computed yet.
   *
   * @param v1: id of the first node
   * @param v2: id of the second node
   * @return true if getEdgeCost(v1, v2, false) has been computed.
   */
  bool trueEdgeCostKnown(unsigned int v1, unsigned int v2) const;

  /**
   * Return the id of the start vertex.
   */
  unsigned int getStartVertex() const;

  /**
   * Return the id of the goal vertex.
   */
  unsigned int getGoalVertex() const;

  // Technically not a function of the graph, but the graph might have its own encoding of vertices, so the
  // heuristic needs to be connected to the graph anyways.
  double heuristic(unsigned int v) const;

  // A flag whether the heuristic is stationary
  typedef std::bool_constant<true> heuristic_stationary;

  /** Additional interface if heuristic is not stationary **/

  /**
   * Query whether the last computed heuristic value for v is still valid.
   * @param v: the vertex id
   * @return true iff heuristic(v) would return the same value as the last time heuristic(v) has been called if ever.
   */
  bool isHeuristicValid(unsigned int v) const;

  /**
   * Inform the graph about the minimal cost to reach v from the start node.
   * This information may be used to improve the heuristic.
   */
  void registerMinimalCost(unsigned int v, double cost);

  // An additional flag indicating that some vertices' heuristic can depend on other vertices being visited
  typedef std::bool_constant<true> heuristic_vertex_dependency;

  /** In case there is a heuristic dependency between vertices, the following additional interface needs to be
   * implemented **/
  /**
   * Return the id of the vertex v0 that needs to be closed so that v has a valid heuristic value.
   * @param v: vertex id
   * @return the node v's heuristic value depends on
   */
  unsigned int getHeuristicDependentVertex(unsigned int v) const;
};

class VertexExpansionLogger
{
public:
  VertexExpansionLogger(RoadmapPtr roadmap);
  ~VertexExpansionLogger();
  void logExpansion(unsigned int rid);
  void logExpansion(unsigned int rid, unsigned int gid);
  void logGoalExpansion();

private:
  RoadmapPtr _roadmap;
};

/**
 * A wrapper around a polymorphic iterator implementation.
 */
struct DynamicNeighborIterator
{
  struct IteratorImplementation
  {
    virtual ~IteratorImplementation() = 0;
    virtual bool equals(const IteratorImplementation* const other) const = 0;
    virtual unsigned int dereference() const = 0;
    virtual void next() = 0;
    virtual std::unique_ptr<IteratorImplementation> copy() const = 0;
    virtual bool isEnd() const = 0;
  };
  DynamicNeighborIterator();
  DynamicNeighborIterator(std::unique_ptr<IteratorImplementation>& impl);
  DynamicNeighborIterator(const DynamicNeighborIterator& other);
  DynamicNeighborIterator(DynamicNeighborIterator&& other);
  ~DynamicNeighborIterator() = default;
  DynamicNeighborIterator& operator++();
  bool operator==(const DynamicNeighborIterator& other) const;
  bool operator!=(const DynamicNeighborIterator& other) const;
  unsigned int operator*();
  // iterator traits
  using difference_type = long;
  using value_type = unsigned int;
  using pointer = const unsigned int*;
  using reference = const unsigned int&;
  using iterator_category = std::forward_iterator_tag;

private:
  std::unique_ptr<IteratorImplementation> _impl;
};

/**
 * The SingleGraspRoadmapGraph class implements a view on a MultiGraspRoadmap for a single grasp.
 * that implements the GraspAgnostic graph interface.
 */
class SingleGraspRoadmapGraph
{
public:
  typedef DynamicNeighborIterator NeighborIterator;
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
  bool checkValidity(unsigned int v);
  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false);
  std::pair<NeighborIterator, NeighborIterator> getSuccessors(unsigned int v, bool lazy = false);
  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false);
  std::pair<NeighborIterator, NeighborIterator> getPredecessors(unsigned int v, bool lazy = false);
  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false);
  bool trueEdgeCostKnown(unsigned int v1, unsigned int v2) const;
  unsigned int getStartVertex() const;
  unsigned int getGoalVertex() const;
  double heuristic(unsigned int v) const;
  typedef std::bool_constant<true> heuristic_stationary;

  std::pair<unsigned int, unsigned int> getGraspRoadmapId(unsigned int vid) const;

private:
  ::placement::mp::mgsearch::RoadmapPtr _roadmap;
  ::placement::mp::mgsearch::MultiGraspGoalSetPtr _goal_set;
  ::placement::mp::mgsearch::MultiGoalCostToGo _cost_to_go;
  const unsigned int _grasp_id;
  const unsigned int _start_rid;                  // roadmap node id of the start vertex
  const static unsigned int START_VERTEX_ID = 0;  // must be 0
  const static unsigned int GOAL_VERTEX_ID = 1;   // must be 1
  unsigned int toVertexId(unsigned int rid) const;
  unsigned int toRoadmapId(unsigned int vid) const;
  VertexExpansionLogger _logger;

  // NeighborIteratorImplementations
  // StandardIterator governs forward and backward adjacency inherited from the roadmap
  template <bool lazy>
  struct StandardIterator : public DynamicNeighborIterator::IteratorImplementation
  {
    StandardIterator(SingleGraspRoadmapGraph const* parent, unsigned int roadmap_id);
    StandardIterator(const StandardIterator<lazy>& other);
    ~StandardIterator() = default;
    // interface
    bool equals(const IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<IteratorImplementation> copy() const override;
    bool isEnd() const override;
    unsigned int getRoadmapId() const;
    SingleGraspRoadmapGraph const* getGraph() const;

  private:
    SingleGraspRoadmapGraph const* _graph;
    const unsigned int _roadmap_id;
    Roadmap::Node::EdgeIterator _iter;
    Roadmap::Node::EdgeIterator _end;
    void forwardToNextValid();
  };

  /**
   * NeighborIterator for vertices adjacent to the goal vertex - use for forward only.
   */
  template <bool lazy>
  struct GoalEntranceIterator : public DynamicNeighborIterator::IteratorImplementation
  {
    GoalEntranceIterator(SingleGraspRoadmapGraph const* parent, unsigned int roadmap_id);
    GoalEntranceIterator(const GoalEntranceIterator<lazy>& other);
    ~GoalEntranceIterator() = default;
    // interface
    bool equals(const IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<IteratorImplementation> copy() const override;
    bool isEnd() const override;

  private:
    StandardIterator<lazy> _standard_iter;
    bool _is_end;
  };

  // NeighborIterator for the goal vertex to iterate over its predecessors.
  struct GoalVertexIterator : public DynamicNeighborIterator::IteratorImplementation
  {
    GoalVertexIterator(SingleGraspRoadmapGraph const* parent);
    ~GoalVertexIterator() = default;
    // interface
    bool equals(const IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<IteratorImplementation> copy() const override;
    bool isEnd() const override;

  private:
    SingleGraspRoadmapGraph const* _graph;
    MultiGraspGoalSet::GoalIterator _iter;
    MultiGraspGoalSet::GoalIterator _end;
  };
};

/**
 * Cost checking types for MultiGraspRoadmapGraph.
 */
enum CostCheckingType
{
  WithGrasp,
  EdgeWithoutGrasp,
  VertexEdgeWithoutGrasp,
};
/**
 * The MultiGraspRoadmapGraph class implements a view on a MultiGraspRoadmap for multiple grasps and
 * implements the GraspAgnostic graph interface.
 * The start vertex of this graph is a special vertex that is not associated with any grasp yet.
 * It is adjacent with cost 0 to #grasps vertices associated with the start configuration - one for each grasp.
 *
 * The graph is template-parameterized by an enum cost_checking_type that allows to control how edge costs and vertex
 * validities are computed when using the grasp agnostic graph interface.
 *
 * If cost_checking_type == WithGrasp, all cost computations and vertex validity checks take the respective grasp into
 * account.
 *
 * If cost_checking_type == EdgeWithoutGrasp, the graph performs edge cost evaluations only for the base (robot) without
 * considering the actual grasp. In this case, the cost of an edge for the actual grasp needs to be explicitly queried
 * using:
 *
 *    double getEdgeCostWithGrasp(unsigned int v1, unsigned int v2);
 *
 * Whether the edge cost for the actual grasp is known can then be queried via:
 *
 *    bool trueEdgeCostWithGraspKnown(unsigned int v1, unsigned int v2) const;
 *
 * In the case cost_checking_type == WithGrasp, the above functions simply return the same as getEdgeCost(v1, v2,
 * lazy=false) and trueEdgeCostKnown(v1, v2)
 *
 * If cost_checking_type == VertexEdgeWithoutGrasp, in addition to edge cost computations, also vertex validity checks
 * are only performed for the robot base without taking the grasp explicitly into account. In this case, the function
 *
 *  bool checkValidityWithGrasp(unsigned int v)
 *
 * needs to be called to explicitly check the validity with the grasp. In the cases cost_checking_type !=
 * VertexEdgeWithoutGrasp, this function simply returns checkValidity(unsigned int v).
 *
 * In any case, all grasp-agnostic functions always return the best known result, i.e. the template argument only
 * governs what is actively computed and not what is retrieved from the cache. For example, if the graph knows already
 * the actual cost of an edge for a given grasp, it of course returns that cost even if cost_checking_type ==
 * EdgeWithoutGrasp or VertexEdgeWithoutGrasp. This allows to lazily evaluate the costs for a specific grasp, while
 * still using the graph in a grasp-agnostic algorithm.
 */
template <CostCheckingType cost_checking_type = WithGrasp>
class MultiGraspRoadmapGraph
{
public:
  typedef DynamicNeighborIterator NeighborIterator;
  /**
   * Create a new MultiGraspRoadmapGraph defined by the given roadmap for the given grasps.
   * @param roadmap - roadmap to use
   * @param goal_set: Set of goals containing goals for the given grasps
   * @param cost_params: Parameters for the goal-path cost tradeoff
   * @param grasp_ids - the ids of the grasps TODO: retrieve from goal_set
   * @param start_id - the id of the roadmap node that defines the start node
   */
  MultiGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                         ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
                         const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params,
                         const std::set<unsigned int>& grasp_ids, unsigned int start_id);
  ~MultiGraspRoadmapGraph();
  // GraspAgnostic graph interface
  bool checkValidity(unsigned int v);
  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false);
  std::pair<NeighborIterator, NeighborIterator> getSuccessors(unsigned int v, bool lazy = false);
  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false);
  std::pair<NeighborIterator, NeighborIterator> getPredecessors(unsigned int v, bool lazy = false);
  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false);
  bool trueEdgeCostKnown(unsigned int v1, unsigned int v2) const;
  unsigned int getStartVertex() const;
  unsigned int getGoalVertex() const;
  double heuristic(unsigned int v) const;
  typedef std::bool_constant<true> heuristic_stationary;
  // additional functions to evaluate grasp-specific costs in case cost_checking_type is not WithGrasp
  double getEdgeCostWithGrasp(unsigned int v1, unsigned int v2);
  bool trueEdgeCostWithGraspKnown(unsigned int v1, unsigned int v2) const;
  // bool checkValidityWithGrasp(unsigned int v);
  bool trueValidityWithGraspKnown(unsigned int v1) const;

  // roadmap id, grasp id
  std::pair<unsigned int, unsigned int> getGraspRoadmapId(unsigned int vid) const;

private:
  ::placement::mp::mgsearch::RoadmapPtr _roadmap;
  ::placement::mp::mgsearch::MultiGraspGoalSetPtr _goal_set;
  // cost-to-go heuristics for all grasps
  ::placement::mp::mgsearch::MultiGoalCostToGo _all_grasps_cost_to_go;
  // cost-to-go heuristics for each grasp (grasp id -> heuristic)
  std::unordered_map<unsigned int, ::placement::mp::mgsearch::MultiGoalCostToGoPtr> _individual_cost_to_go;
  // grasp ids
  const std::set<unsigned int> _grasp_ids;
  // hash table mapping (grasp_id, roadmap_id) to graph id
  typedef std::pair<unsigned int, unsigned int> GraspNodeIDPair;
  mutable std::unordered_map<GraspNodeIDPair, unsigned int, boost::hash<GraspNodeIDPair>> _roadmap_key_to_graph;
  // hash table mapping graph id to (grasp_id, roadmap_id)
  mutable std::unordered_map<unsigned int, GraspNodeIDPair> _graph_key_to_roadmap;
  unsigned int _roadmap_start_id;

  const static unsigned int START_VERTEX_ID;  // must be 0
  const static unsigned int GOAL_VERTEX_ID;   // must be 1

  // grasp id, roadmap id
  std::pair<unsigned int, unsigned int> toRoadmapKey(unsigned int graph_id) const;
  unsigned int toGraphKey(const std::pair<unsigned int, unsigned int>& roadmap_id) const;
  unsigned int toGraphKey(unsigned int grasp_id, unsigned int roadmap_id) const;
  mutable unsigned int _num_graph_nodes;
  // logger
  VertexExpansionLogger _logger;

  // Iterator implementations
  // StandardIterator governs forward and backward adjacency inherited from the roadmap
  template <bool lazy>
  struct StandardIterator : public DynamicNeighborIterator::IteratorImplementation
  {
    StandardIterator(MultiGraspRoadmapGraph const* parent, unsigned int roadmap_id, unsigned int grasp_id);
    StandardIterator(const StandardIterator<lazy>& other);
    ~StandardIterator() = default;
    // interface
    bool equals(const IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<IteratorImplementation> copy() const override;
    bool isEnd() const override;
    unsigned int getRoadmapId() const;
    MultiGraspRoadmapGraph const* getGraph() const;

  private:
    MultiGraspRoadmapGraph const* _graph;
    const unsigned int _roadmap_id;
    const unsigned int _grasp_id;
    Roadmap::Node::EdgeIterator _iter;
    Roadmap::Node::EdgeIterator _end;
    void forwardToNextValid();
  };

  /**
   * NeighborIterator for vertices adjacent to start or goal vertex in addition to normal adjacency.
   */
  template <bool lazy>
  struct StartGoalBridgeIterator : public DynamicNeighborIterator::IteratorImplementation
  {
    // pass id of goal or start vertex as special vertex
    StartGoalBridgeIterator(MultiGraspRoadmapGraph<cost_checking_type> const* parent, unsigned int roadmap_id,
                            unsigned int grasp_id, unsigned int special_vertex);
    StartGoalBridgeIterator(const StartGoalBridgeIterator<lazy>& other);
    ~StartGoalBridgeIterator() = default;
    // interface
    bool equals(const IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<IteratorImplementation> copy() const override;
    bool isEnd() const override;

  private:
    StandardIterator<lazy> _standard_iter;
    bool _is_end;
    unsigned int _special_vertex;
  };

  // NeighborIterator for the goal vertex to iterate over its predecessors.
  struct GoalVertexIterator : public DynamicNeighborIterator::IteratorImplementation
  {
    GoalVertexIterator(MultiGraspRoadmapGraph<cost_checking_type> const* parent);
    ~GoalVertexIterator() = default;
    // interface
    bool equals(const IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<IteratorImplementation> copy() const override;
    bool isEnd() const override;

  private:
    MultiGraspRoadmapGraph<cost_checking_type> const* _graph;
    MultiGraspGoalSet::GoalIterator _iter;
    MultiGraspGoalSet::GoalIterator _end;
  };

  // IteratorImplementation to get successors of the start vertex.
  struct StartVertexIterator : public DynamicNeighborIterator::IteratorImplementation
  {
    StartVertexIterator(MultiGraspRoadmapGraph<cost_checking_type> const* parent);
    ~StartVertexIterator() = default;
    // interface
    bool equals(const IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<IteratorImplementation> copy() const override;
    bool isEnd() const override;

  private:
    MultiGraspRoadmapGraph<cost_checking_type> const* _graph;
    std::set<unsigned int>::const_iterator _iter;
  };

  // struct StandardIterator : public DynamicNeighborIterator::IteratorImplementation
  // {
  //   ~StandardIterator() = 0;
  //   virtual bool equals(const IteratorImplementation* const other) const = 0;
  //   virtual unsigned int dereference() const = 0;
  //   virtual void next() = 0;
  //   virtual std::unique_ptr<IteratorImplementation> copy() const = 0;
  //   virtual bool isEnd() const = 0;

  //   ~NeighborIterator() = default;
  //   NeighborIterator& operator++();
  //   bool operator==(const NeighborIterator& other) const;
  //   bool operator!=(const NeighborIterator& other) const;
  //   unsigned int operator*();
  //   // iterator traits
  //   using difference_type = long;
  //   using value_type = unsigned int;
  //   using pointer = const unsigned int*;
  //   using reference = const unsigned int&;
  //   using iterator_category = std::forward_iterator_tag;

  //   static NeighborIterator begin(unsigned int v, bool lazy, MultiGraspRoadmapGraph const* graph);
  //   static NeighborIterator end(unsigned int v, MultiGraspRoadmapGraph const* graph);

  // private:
  //   NeighborIterator();
  //   NeighborIterator(unsigned int v, bool lazy, MultiGraspRoadmapGraph const* parent);
  //   unsigned int _v;
  //   // information about grasps
  //   std::set<unsigned int>::iterator _grasp_iter;  // for v == 0
  //   unsigned int _grasp_id;                        // grasp id for any other vertex
  //   unsigned int _roadmap_id;                      // roadmap node if for any other vertex
  //   // iterators for roadmap edges
  //   Roadmap::Node::EdgeIterator _iter;
  //   Roadmap::Node::EdgeIterator _end;
  //   bool _is_end;  // flag to indicate the iterator is at the end
  //   bool _lazy;
  //   MultiGraspRoadmapGraph const* _graph;
  //   // flag for special case edge back to node 0
  //   bool _edge_to_0_returned;
  //   void forwardToNextValid();
  // };
};

/**
 * Heuristic types for layers of the FoldedMultiGraspRoadmapGraph other than the base layer.
 */
enum BackwardsHeuristicType
{
  LowerBound,  // simply compute the lower bound c_l(v, s) of the path cost to go to the start, h_i(v) = c_l(v, s)
  // BestKnownDistance,            // use registered shortest path costs to start, i.e. h_i(v) = min(g(v), c_l(v, s))
  SearchAwareBestKnownDistance  // estimate g(v) from the current f-values, i.e. h_i(v) = min(g(v), max(f_c - h_0(v),
                                // c_l(v, s))), where f_c is the current f value and h_0(v) the heuristic of v's
                                // corresponding vertex in the base layer
};
/**
 * The FoldedMultiGraspRoadmapGraph is a special graph view on a MultiGraspRoadmap that aims to facilitate
 * exploiting similarities between the different cost spaces across grasps. Similar to the MultiGraspRoadmapGraph
 * this graph consists of different layers, where each layer represents the roadmap conditioned on a different grasp.
 * In addition, this graph contains a root layer representing the roadmap without any grasp, i.e. only the robot.
 * Let in the following "vertex" refer to a vertex in this graph and "node" refer to a node, i.e. a configuration, in
 * the roadmap. The start vertex of the graph is at the start roadmap node in the root layer. From there, most vertices
 * are only adjacent to other vertices within the same layer. An exception to this are the vertices in the base layer
 * that are at the goal configurations (nodes). Rather than being goal vertices, these vertices instead connect to the
 * other layers of the graph with a special edge that has negative costs. This cost is computed dynamically and
 * represents the lower bound on the cost of moving from the start node to the respective goal node.
 * The actual goal vertices are the vertices at the start node in the layers associated with specific grasps. This
 * means that any valid path first moves through the base layer to a goal node and from there into the layer of the
 * grasp that the goal is associated with and then in this layer back to the start node.
 * The advantage of this structure is that a search algorithm can first explore the base layer, which provides
 * significant information for all grasps, before commiting to a specific grasp.
 *
 * This is achieved by dynamically adjusting the heuristic values depending on what the search algorithm has discovered.
 * For this dynamic adjustment, the graph provides a function registerMinimalCost(v, cost) that a search algorithm
 * should call after the minimal cost for v has been computed. The registration of minimal costs on the base layer, i.e.
 * shortest path costs to move to a node without considering any grasp, allows to provide strong heuristic values for
 * the return paths through the grasp-specific layers.
 *
 * To evaluate the effect of the backwards heuristic, this class is parameterized by the enum BackwardsHeuristicType.
 * It governs how the heuristic for vertices in the grasp-specific layers is computed.
 */
template <BackwardsHeuristicType htype>
class FoldedMultiGraspRoadmapGraph
{
public:
  struct NeighborIterator
  {
    NeighborIterator(const NeighborIterator& other);
    NeighborIterator(NeighborIterator&& other);
    ~NeighborIterator() = default;
    NeighborIterator& operator++();
    bool operator==(const NeighborIterator& other) const;
    bool operator!=(const NeighborIterator& other) const;
    unsigned int operator*();
    // iterator traits
    using difference_type = long;
    using value_type = unsigned int;
    using pointer = const unsigned int*;
    using reference = const unsigned int&;
    using iterator_category = std::forward_iterator_tag;

    static NeighborIterator begin(unsigned int v, bool forward, bool lazy, FoldedMultiGraspRoadmapGraph const* graph);
    static NeighborIterator end(unsigned int v, bool forward, bool lazy, FoldedMultiGraspRoadmapGraph const* graph);

    struct IteratorImplementation
    {
      // virtual bool equals(const std::unique_ptr<IteratorImplementation>& other) const = 0;
      virtual ~IteratorImplementation() = 0;
      virtual bool equals(const IteratorImplementation* const other) const = 0;
      virtual unsigned int dereference() const = 0;
      virtual void next() = 0;
      virtual std::unique_ptr<IteratorImplementation> copy() const = 0;
      virtual void setToEnd() = 0;
    };

  private:
    NeighborIterator(unsigned int v, bool forward, bool lazy, FoldedMultiGraspRoadmapGraph const* parent);
    std::unique_ptr<IteratorImplementation> _impl;
  };

  /**
   * Create a new MultiGraspRoadmapGraph defined by the given roadmap for the given grasps.
   * @param roadmap - roadmap to use
   * @param goal_set: Set of goals containing goals for the given grasps
   * @param cost_params: Parameters for the goal-path cost tradeoff
   * @param start_id - the id of the roadmap node that defines the start node
   */
  FoldedMultiGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                               ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
                               const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params,
                               unsigned int start_id);

  ~FoldedMultiGraspRoadmapGraph() = default;

  // GraspAgnostic graph interface
  bool checkValidity(unsigned int v);

  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false);
  std::pair<NeighborIterator, NeighborIterator> getSuccessors(unsigned int v, bool lazy = false);
  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false);
  std::pair<NeighborIterator, NeighborIterator> getPredecessors(unsigned int v, bool lazy = false);

  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false);
  bool trueEdgeCostKnown(unsigned int v1, unsigned int v2) const;
  unsigned int getStartVertex() const;
  unsigned int getGoalVertex() const
  {
    // TODO implement me
    throw std::runtime_error("Not implemented");
    return 0;
  }

  bool isGoal(unsigned int v) const;
  double getGoalCost(unsigned int v) const;
  double heuristic(unsigned int v) const;

  // cost aware interface
  void registerMinimalCost(unsigned int v, double cost);
  bool isHeuristicValid(unsigned int v) const;
  typedef std::is_same<std::integral_constant<BackwardsHeuristicType, htype>,
                       std::integral_constant<BackwardsHeuristicType, BackwardsHeuristicType::LowerBound>>
      heuristic_stationary;
  typedef std::bool_constant<true> heuristic_vertex_dependency;
  // return the id of the vertex v0 that needs to be closed so that v has a valid heuristic value
  unsigned int getHeuristicDependentVertex(unsigned int v) const;

  /**
   * Return the roadmap id and grasp id associated with the given vertex.
   * @return ((rid, gid), gid_valid): The flag gid_valid is true if vid is a vertex associated with
   *  a specific grasp. If it is not, i.e. on the base layer, gid_valid is false.
   */
  std::pair<std::pair<unsigned int, unsigned int>, bool> getGraspRoadmapId(unsigned int vid) const;

private:
  const ::placement::mp::mgsearch::RoadmapPtr _roadmap;
  const ::placement::mp::mgsearch::MultiGraspGoalSetPtr _goal_set;
  const ::placement::mp::mgsearch::MultiGoalCostToGo _cost_to_go;
  const ::placement::mp::mgsearch::PathCostFn _lower_bound;
  // const BackwardsHeuristicType _htype;
  const unsigned int _start_rid;  // roadmap id of the start node

  struct VertexInformation
  {
    VertexInformation(unsigned int rid, unsigned int lid) : roadmap_id(rid), layer_id(lid)
    {
    }
    VertexInformation() : roadmap_id(0), layer_id(0)
    {
    }
    VertexInformation(const VertexInformation&) = default;
    unsigned int roadmap_id;
    unsigned int layer_id;  // grasp_id + 1 (0 = base layer = no grasp)
  };

  // array of vertex information, indexed by vertex id
  mutable std::vector<VertexInformation> _vertex_info;
  // map from (roadmap_id, layer_id) -> vertex_id.
  typedef std::pair<unsigned int, unsigned int> GraspNodeIDPair;
  mutable std::unordered_map<GraspNodeIDPair, unsigned int, boost::hash<GraspNodeIDPair>> _vertex_ids;
  // maps vertex id to registered path cost
  std::unordered_map<unsigned int, double> _registered_costs;
  // logger
  VertexExpansionLogger _logger;

  // convenience function to add/retrieve vertex id from _vertex_ids
  unsigned int getVertexId(unsigned int roadmap_id, unsigned int layer_id) const;

  double baseLayerHeuristic(unsigned int v) const;
  /**** Heuristic implementations for grasp-specific layers  ****/
  // simple lower bound heuristic
  double getLowerBound(unsigned int v) const;

  double graspLayerHeuristicImplementation(
      unsigned int v,
      const std::integral_constant<BackwardsHeuristicType, BackwardsHeuristicType::LowerBound>& htype_flag) const;

  double graspLayerHeuristicImplementation(
      unsigned int v,
      const std::integral_constant<BackwardsHeuristicType, BackwardsHeuristicType::SearchAwareBestKnownDistance>&
          htype_flag) const;
  // Iterator implementations
  /**
   * Neighbor iterator for vertices that are only adjacent to other vertices in the same layer.
   * Template arguments:
   *  lazy = lazy cost evaluation
   *  base = base layer
   */
  template <bool lazy, bool base>
  class InLayerIterator : public NeighborIterator::IteratorImplementation
  {
  public:
    InLayerIterator(unsigned int v, FoldedMultiGraspRoadmapGraph<htype> const* parent)
      : _v(v)
      , _graph(parent)
      , _layer_id(_graph->_vertex_info.at(v).layer_id)
      , _roadmap_id(_graph->_vertex_info.at(v).roadmap_id)
    {
      auto node = _graph->_roadmap->getNode(_roadmap_id);
      assert(node);
      std::tie(_iter, _end) = node->getEdgesIterators();
    }

    // InLayerIterator(const InLayerIterator<lazy, base>& other)
    //   : _v(other._v)
    //   , _graph(other._graph)
    //   , _layer_id(other._layer_id)
    //   , _roadmap_id(other._roadmap_id)
    //   , _iter(other._iter)
    //   , _end(other._end)
    // {
    // }

    ~InLayerIterator() = default;

    bool equals(const typename NeighborIterator::IteratorImplementation* const other) const override
    {
      auto other_casted = dynamic_cast<const InLayerIterator<lazy, base>*>(other);
      if (!other_casted)
        return false;
      return other_casted->_graph == _graph && other_casted->_v == _v && other_casted->_iter == _iter;
    }

    unsigned int dereference() const override
    {
      return _graph->getVertexId(_iter->first, _layer_id);
    }

    void next() override
    {
      if (_iter != _end)
      {
        ++_iter;
      }
      forwardToNextValid();
    }

    std::unique_ptr<typename NeighborIterator::IteratorImplementation> copy() const override
    {
      return std::make_unique<InLayerIterator<lazy, base>>(*this);
    }

    void setToEnd() override
    {
      _iter = _end;
    }

  private:
    // const information
    const unsigned int _v;
    FoldedMultiGraspRoadmapGraph const* const _graph;
    const unsigned int _layer_id;
    const unsigned int _roadmap_id;
    // iterators for roadmap edges
    Roadmap::Node::EdgeIterator _iter;
    Roadmap::Node::EdgeIterator _end;

    void forwardToNextValid()
    {
      std::integral_constant<bool, lazy> lazy_flag;
      std::integral_constant<bool, base> base_flag;
      while (_iter != _end and !checkEdgeValidity(lazy_flag, base_flag))
      {
        _iter++;
      }
    }

    bool checkEdgeValidity(const std::true_type& lazy_flag, const std::true_type& base_flag)
    {
      return not std::isinf(_iter->second->base_cost);
    }

    bool checkEdgeValidity(const std::false_type& non_lazy, const std::true_type& base_flag)
    {
      return _graph->_roadmap->computeCost(_iter->second).first;
    }

    bool checkEdgeValidity(const std::true_type& lazy_flag, const std::false_type& non_base)
    {
      double cost = _iter->second->getBestKnownCost(_layer_id - 1);
      return not std::isinf(cost);
    }

    bool checkEdgeValidity(const std::false_type& non_lazy, const std::false_type& non_base)
    {
      return _graph->_roadmap->computeCost(_iter->second, _layer_id - 1).first;
    }
  };
  template <bool lazy, bool base>
  friend class InLayerIterator;

  /**
   * Neighbor iterator for vertices that bridge between layers.
   * Template arguments:
   *  lazy = lazy cost evaluation
   *  base = base layer
   */
  template <bool lazy, bool base>
  class LayerBridgeIterator : public NeighborIterator::IteratorImplementation
  {
  public:
    LayerBridgeIterator(unsigned int v, FoldedMultiGraspRoadmapGraph const* parent)
      : _v(v)
      , _graph(parent)
      , _layer_id(_graph->_vertex_info.at(v).layer_id)
      , _roadmap_id(_graph->_vertex_info.at(v).roadmap_id)
      , _in_layer_iter(v, parent)
      , _in_layer_end(v, parent)
      , _current_goal_idx(0)
      , _num_remaining_bridges(1)
    {
      _in_layer_end.setToEnd();
      if constexpr (base)
      {
        _goal_ids = _graph->_goal_set->getGoalIds(_roadmap_id);
        _num_remaining_bridges = _goal_ids.size();
      }
    }

    LayerBridgeIterator(const LayerBridgeIterator& other) = default;

    ~LayerBridgeIterator() = default;

    bool equals(const typename NeighborIterator::IteratorImplementation* const other) const override
    {
      auto other_casted = dynamic_cast<const LayerBridgeIterator<lazy, base>*>(other);
      if (!other_casted)
        return false;
      return other_casted->_graph == _graph && other_casted->_v == _v &&
             other_casted->_in_layer_iter.equals(&_in_layer_iter) &&
             _current_goal_idx == other_casted->_current_goal_idx &&
             _num_remaining_bridges == other_casted->_num_remaining_bridges;
    }

    unsigned int dereference() const override
    {
      if (_in_layer_iter.equals(&_in_layer_end))
      {
        if constexpr (base)
        {
          assert(_current_goal_idx != _goal_ids.size());
          unsigned int new_layer = _graph->_goal_set->getGoal(_goal_ids.at(_current_goal_idx)).grasp_id + 1;
          return _graph->getVertexId(_roadmap_id, new_layer);
        }
        else
        {
          assert(_num_remaining_bridges > 0);
          return _graph->getVertexId(_roadmap_id, 0);
        }
      }
      return _in_layer_iter.dereference();
    }

    void next() override
    {
      if (!_in_layer_iter.equals(&_in_layer_end))
      {
        _in_layer_iter.next();
      }
      else
      {
        if (base && _current_goal_idx != _goal_ids.size())
        {
          _current_goal_idx++;
        }
        _num_remaining_bridges = _num_remaining_bridges > 0 ? _num_remaining_bridges - 1 : 0;
      }
    }

    std::unique_ptr<typename NeighborIterator::IteratorImplementation> copy() const override
    {
      return std::make_unique<LayerBridgeIterator<lazy, base>>(*this);
    }

    void setToEnd() override
    {
      _in_layer_iter.setToEnd();
      if constexpr (base)
      {
        _current_goal_idx = _goal_ids.size();
      }
      _num_remaining_bridges = 0;
    }

  private:
    const unsigned int _v;
    const FoldedMultiGraspRoadmapGraph* const _graph;
    const unsigned int _layer_id;
    const unsigned int _roadmap_id;
    InLayerIterator<lazy, base> _in_layer_iter;
    InLayerIterator<lazy, base> _in_layer_end;
    std::vector<unsigned int> _goal_ids;
    size_t _current_goal_idx;
    unsigned int _num_remaining_bridges;
  };
  template <bool lazy, bool base>
  friend class LayerBridgeIterator;
};
// include implementation of FoldedMultiGraspRoadmapGraph
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs-impl.h>

}  // namespace mgsearch
}  // namespace mp
}  // namespace placement