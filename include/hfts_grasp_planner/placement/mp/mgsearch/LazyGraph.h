#pragma once
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>
#include <vector>
#include <iterator>

namespace placement
{
namespace mp
{
namespace mgsearch
{
/**
 * The LazyLayeredMultiGraspRoadmapGraph is a special type of MultiGraspRoadmapGraph that aims to combine the
 * benefits of MultiGraspRoadmapGraph and the FoldedMultiGraspRoadmapGraph. Similar to the FoldedMultiGraspRoadmapGraph,
 * the LazyLayeredMultiGraspRoadmapGraph has initially only a single layer that corresponds to all grasps and
 * contains all goals.
 * Whenever an edge on the base layer is found to differ for a specific grasp g, a new layer is inserted into the graph
 * that represents the roadmap conditioned on g. This is done by simply adding an edge between the virtual start vertex
 * and the start node conditioned on g. For the base layer, in turn, the heuristic function is adapted to no longer
 * include goals for g and the goal costs of goals belonging to g are set to infinity for that layer. This way, the
 * graph lazily grows from a single layer to MultiGraspRoadmapGraph as more different grasps are actually evaluated.
 * Once the base layer corresponds only to a single grasp, the graph is equivalent to the MultiGraspRoadmapGraph.
 *
 * The graph is designed to be used by a grasp-aware specialization of LazySP that uses a grasp-agnostic LPA*.
 * For this, the graph implements the GraspAgnostic interface and additional grasp-aware functions.
 * Whenever LPA* finds a new solution on the grasp-agnostic interface, the specialized LazySP algorithm
 * identifies what grasp is required by the goal of that solution and starts evaluating
 * the true edge costs w.r.t this grasp. When a cost is different, the graph adjusts its internal structure as described
 * and informs LazySP about the new edge weights and goal cost increases.
 *
 * Similar to the MultiGraspRoadmapGraph, the type of cost evaluation performed
 * when using the grasp-agnostic interface is template-parameterized by the enum CostCheckingType.
 * Note that as long as the base layer represents multiple grasps, the cost evaluation on the base layer is always
 * VertexEdgeWithoutGrasp.
 */
template <CostCheckingType cost_checking_type = EdgeWithoutGrasp>
class LazyLayeredMultiGraspRoadmapGraph
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

    static NeighborIterator begin(unsigned int v, bool forward, bool lazy,
                                  LazyLayeredMultiGraspRoadmapGraph const* graph);
    static NeighborIterator end(unsigned int v, bool forward, bool lazy,
                                LazyLayeredMultiGraspRoadmapGraph const* graph);

    struct IteratorImplementation
    {
      virtual ~IteratorImplementation() = 0;
      virtual bool equals(const IteratorImplementation* const other) const = 0;
      virtual unsigned int dereference() const = 0;
      virtual void next() = 0;
      virtual std::unique_ptr<IteratorImplementation> copy() const = 0;
      virtual void setToEnd() = 0;
    };

  private:
    NeighborIterator(unsigned int v, bool forward, bool lazy, LazyLayeredMultiGraspRoadmapGraph const* parent);
    std::unique_ptr<IteratorImplementation> _impl;
  };

  /**
   * Create a new LazyLayeredMultiGraspRoadmapGraph defined by the given roadmap for the given grasps.
   * @param roadmap - roadmap to use
   * @param goal_set: Set of goals containing goals for the given grasps
   * @param cost_params: Parameters for the goal-path cost tradeoff
   * @param grasp_ids - the ids of the grasps TODO: retrieve from goal_set
   * @param start_id - the id of the roadmap node that defines the start node
   */
  LazyLayeredMultiGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                                    ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
                                    const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params,
                                    const std::set<unsigned int>& grasp_ids, unsigned int start_id);
  ~LazyLayeredMultiGraspRoadmapGraph();

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
    throw std::runtime_error("Not implemented");
    return 0;
  }
  bool isGoal(unsigned int v) const;
  double getGoalCost(unsigned int v) const;
  double heuristic(unsigned int v) const;

  // non-stationary heuristic interface
  typedef std::bool_constant<false> heuristic_stationary;
  // Inform the graph that vertex v has been closed and it no longer needs to monitor the correctness of v's heuristic
  // value. We do not care about the actual cost.
  void registerMinimalCost(unsigned int v, double cost);
  // Return whether the heuristic value for v is valid.
  bool isHeuristicValid(unsigned int v) const;
  // we have no vertex dependency for the heuristic
  typedef std::bool_constant<false> heuristic_vertex_dependency;

  // Grasp-aware interface
  /**
   * Return/Compute the cost of the edge connecting v1 and v2 for grasp gid.
   * If v1 and v2 belong to the layer that is only associated with gid, this function simply computes the true cost
   * of the edge <v1,v2> taking the grasp into account.
   * If v1 and v2 belong the the base layer that is associated with many grasps (including gid), this function in
   * addition may result in a split of the layer if cost(v1, v2, gid) != cost(v1, v2). This means that this function
   * has severe influence on the future behaviour of this graph. Apart from changes to edge weights, the goal costs of
   * some vertices may change, too. This implies that some vertices may no longer be goals and isGoal returns false
   * for vertices for which it formerly returned true. In addition, if eventually the base layer only corresponds to a
   * single grasp, the return values of trueEdgeCostKnown, getEdgeCost for vertices on that layer also change. You can
   * query information about changes that occurred during splits using the functions getNewEdges and
   * getInvalidatedGoals.
   * @param v1 - the first vertex
   * @param v2 - the second vertex
   * @param gid - the id of the grasp
   * @return edge cost
   */
  double getGraspSpecificEdgeCost(unsigned int v1, unsigned int v2, unsigned int gid);

  /**
   * Return whether the grasp specific cost for grasp gid of edge (v1, v2) is known.
   * If v1 and v2 to not belong to the same layer or gid is not associated with layer of v1 and v2, an TODO: assertion
   * error will be thrown.
   * @param v1: the first vertex
   * @param v2: the second vertex
   * @param gid: the grasp id
   * @return whether the cost(v1, v2, gid) is known.
   */
  bool isGraspSpecificEdgeCostKnown(unsigned int v1, unsigned int v2, unsigned int gid) const;

  /**
   * Return whether the grasp-specific validity of v is known for grasp gid.
   * If v is not on a layer containing gid, an TODO assertion error will be thrown.
   * @param v: the vertex id
   * @param gid: the grasp id
   * @return true if grasp-specific validity is known, else false
   */
  bool isGraspSpecificValidityKnown(unsigned int v, unsigned int gid) const;

  /**
   * Return edge changes that were made due to grasp-specific edge cost computations, i.e. layer splits.
   * These are typically new edges to grasp-specific layers, but may also be an edge removal to the base layer if
   * there is a grasp-specific layer for each grasp.
   * @param edge_changes vector of edge changes <v1, v2, increase>
   * @param clear_cache - if true, clear the change cache so that next time this function is called only
   *    new edge changes are returned
   * @return true if there are any new edge changes
   */
  bool getHiddenEdgeChanges(std::vector<EdgeChange>& edge_changes, bool clear_cache);

  /**
   * Return the ids of vertices for which goal costs have changed and may no longer be goals due to layer splits.
   * @param changed_goals will contain the vertices for which the goal cost has changed.
   * @param clear_cache: if true, clear the internal cache so that subsequent calls will not return the same goals
   * again.
   * @return true if there are any changed goals
   */
  bool getGoalChanges(std::vector<unsigned int>& goal_changes, bool clear_cache);

  // roadmap id, grasp id
  // std::pair<unsigned int, unsigned int> getGraspRoadmapId(unsigned int vid) const;
  /**
   * Return the best goal associated with v and the cost associated with it.
   * If there is no goal associated with v, the goal is uninitialized and the goal cost is infinite.
   * @param v: the graph vertex id
   * @return: the best goal and the corresponding goal cost
   */
  std::pair<MultiGraspMP::Goal, double> getBestGoal(unsigned int v) const;

  /**
   * Return the grasp and roadmap node associated with vertex vid.
   * If the vertex vid is on the base layer, and thus associated with multiple grasps
   * TODO: gid = 0 is returned. TODO a logic_error is thrown?
   * @param vid: the vertex id
   * @return {roadmap_id, grasp_id}
   */
  std::pair<unsigned int, unsigned int> getGraspRoadmapId(unsigned int vid) const;

private:
  // roadmap, costs, etc
  ::placement::mp::mgsearch::RoadmapPtr _roadmap;
  const ::placement::mp::mgsearch::GoalPathCostParameters _cost_params;
  const unsigned int _roadmap_start_id;  // roadmap node id of the start node
  double _start_h;                       // heuristic value of virtual start vertex
  std::pair<double, double> _goal_quality_range;

  struct LayerInformation
  {  // stores information for each layer
    ::placement::mp::mgsearch::MultiGoalCostToGoPtr cost_to_go;
    ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set;
    std::set<unsigned int> grasps;
    unsigned int start_vertex_id;  // vertex id of the start vertex of this layer

    LayerInformation() : cost_to_go(nullptr), goal_set(nullptr), start_vertex_id(0)
    {
    }

    LayerInformation(::placement::mp::mgsearch::MultiGoalCostToGoPtr cost_to_go_,
                     ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set_, const std::set<unsigned int>& grasps_,
                     unsigned int start_vertex_id_)
      : cost_to_go(cost_to_go_), goal_set(goal_set_), grasps(grasps_), start_vertex_id(start_vertex_id_)
    {
    }
  };
  std::vector<LayerInformation> _layers;
  std::unordered_map<unsigned int, unsigned int> _grasp_id_to_layer_id;
  // store for base layer vertices which grasp was responsible for their last computed heuristic value
  mutable std::unordered_map<unsigned int, unsigned int> _grasp_for_heuristic_value;
  // edge change cache
  std::vector<EdgeChange> _hidden_edge_changes;
  // goal change cache
  std::vector<unsigned int> _goal_changes;
  // id helper
  typedef std::pair<unsigned int, unsigned int> LayerNodeIDPair;  // {layer_id, roadmap_node_id}
  LayerNodeIDPair toLayerRoadmapKey(unsigned int graph_id) const;
  unsigned int toGraphKey(unsigned int layer_id, unsigned int roadmap_id) const;
  // counter of total number of vertices
  mutable unsigned int _num_graph_vertices;
  // hash table mapping (layer_id, roadmap_id) to graph id
  mutable std::unordered_map<LayerNodeIDPair, unsigned int, boost::hash<LayerNodeIDPair>> _layer_roadmap_key_to_graph;
  // hash table mapping graph id to (layer_id, roadmap_id)
  mutable std::unordered_map<unsigned int, LayerNodeIDPair> _graph_key_to_layer_roadmap;
  // logger
  VertexExpansionLogger _logger;
  // Iterator implementations
  template <bool forward>
  class StartVertexIterator : public NeighborIterator::IteratorImplementation
  {
  public:
    StartVertexIterator(LazyLayeredMultiGraspRoadmapGraph const* graph);
    ~StartVertexIterator();
    bool equals(const typename NeighborIterator::IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<typename NeighborIterator::IteratorImplementation> copy() const override;
    void setToEnd() override;

  private:
    const LazyLayeredMultiGraspRoadmapGraph* const _graph;
    typename std::vector<LayerInformation>::const_iterator _layer_iter;
  };
  template <bool forward>
  friend class StartVertexIterator;

  template <bool lazy, bool forward, bool base>
  class InLayerVertexIterator : public NeighborIterator::IteratorImplementation
  {
  public:
    InLayerVertexIterator(unsigned int layer_id, unsigned int roadmap_id,
                          LazyLayeredMultiGraspRoadmapGraph const* graph);
    ~InLayerVertexIterator();
    bool equals(const typename NeighborIterator::IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<typename NeighborIterator::IteratorImplementation> copy() const override;
    void setToEnd() override;

  private:
    const LazyLayeredMultiGraspRoadmapGraph* const _graph;
    const unsigned int _layer_id;
    const unsigned int _roadmap_id;
    const unsigned int _grasp_id;
    bool _edge_to_start_returned;  // bit to track whether we returned edge to the start vertex (if forward = false and
                                   // we are at a layer start vertex)
    Roadmap::Node::EdgeIterator _iter;
    Roadmap::Node::EdgeIterator _end;
    void forwardToNextValid();
  };
  template <bool lazy, bool forward, bool base>
  friend class InLayerVertexIterator;

  /**
   * Special iterator implementation for the case that a roadmap node does no longer exist.
   * It's always equal to its end, i.e. should never be dereferenced nor increased (calling next()).
   */
  class InvalidVertexIterator : public NeighborIterator::IteratorImplementation
  {
  public:
    InvalidVertexIterator();
    ~InvalidVertexIterator();
    bool equals(const typename NeighborIterator::IteratorImplementation* const other) const override;
    unsigned int dereference() const override;
    void next() override;
    std::unique_ptr<typename NeighborIterator::IteratorImplementation> copy() const override;
    void setToEnd() override;
  };
};

#include <hfts_grasp_planner/placement/mp/mgsearch/LazyGraph-impl.h>

#if 0
/**
 * The LazyMultiGraspRoadmapGraph class implements a view on a MultiGraspRoadmap for multiple grasps, and
 * implements the GraspAgnostic graph interface. In contrast to the MultiGraspRoadmapGraph, this view is constructed
 * lazily. The base assumption is that the grasp is irrelevant and the graph assumes that any vertex can be reached with
 * every grasp. Only when graph edges are explicitly queried to be evaluated for a specific grasp g, the different costs
 * for the grasp are taken into account. If it turns out that the tested edge (u, v) is invalid or has higher cost than
 * anticipated, the graph splits the edge and adds two new nodes v' and v'', where v' is now specifically associated
 * with the grasp g, while v'' remains associated with all remaining grasps. As a consequence, any edge (v', w) will
 * lead to other nodes w only associated with g, whereas all edges (v'', w) will lead to nodes w associated with the
 * remaining grasps excluding g. The node v itself will be removed from the graph and all edges (x, v), (v, x) will
 * lazily be set to infinity.
 *
 * The start vertex is always associated with all grasps.
 *
 * This graph is intended to be used with LPA* and LazySP.
 */
class LazyMultiGraspRoadmapGraph
{
public:
  struct NeighborIterator
  {
    NeighborIterator(const NeighborIterator&);
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

    static NeighborIterator begin(unsigned int v, bool forward, bool lazy, LazyMultiGraspRoadmapGraph const* graph);
    static NeighborIterator end(unsigned int v, bool forward, bool lazy, LazyMultiGraspRoadmapGraph const* graph);

    struct IteratorImplementation
    {
      virtual bool equals(const std::unique_ptr<IteratorImplementation>& other) const = 0;
      virtual unsigned int dereference() const = 0;
      virtual void next() = 0;
      virtual std::unique_ptr<IteratorImplementation> copy() const = 0;
      virtual void setToEnd() = 0;
    };

  private:
    NeighborIterator(unsigned int v, bool forward, bool lazy, LazyMultiGraspRoadmapGraph const* parent);
    std::unique_ptr<IteratorImplementation> _impl;
  };

  /**
   * Create a new LazyMultiGraspRoadmapGraph defined by the given roadmap for the given grasps.
   * @param roadmap - roadmap to use
   * @param goal_set: Set of goals containing goals for the given grasps
   * @param cost_params: Parameters for the goal-path cost tradeoff
   * @param start_id - the id of the roadmap node that defines the start node
   */
  LazyMultiGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                             ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
                             const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params,
                             unsigned int start_id);
  ~LazyMultiGraspRoadmapGraph();
  // GraspAgnostic graph interface
  bool checkValidity(unsigned int v) const;
  void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
  std::pair<NeighborIterator, NeighborIterator> getSuccessors(unsigned int v, bool lazy = false) const;
  void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
  std::pair<NeighborIterator, NeighborIterator> getPredecessors(unsigned int v, bool lazy = false) const;
  double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
  bool trueEdgeCostKnown(unsigned int v1, unsigned int v2) const;
  unsigned int getStartNode() const;
  bool isGoal(unsigned int v) const;
  double getGoalCost(unsigned int v) const;
  double heuristic(unsigned int v) const;
  // Grasp aware interface

  /**
   * Check the edge (v1, v2) for grasp gid and split the edge if needed.
   * If the grasp has no effect on the edge (i.e. it has the same cost as was already assumed),
   * nothing is changed. If the edge has a larger cost for grasp gid than originally anticipated,
   * v2 is split in v2' and v2'' as described in the class description and the ids of the new
   * nodes <v2', v2''> are returned. Future evaluations to functions of the GraspAgnostic interface
   * involving the ids v1 and v2 will reflect the state in the updated graph.
   * @param v1 - first vertex id
   * @param v2 - second vertex id
   * @param gid - grasp id
   * @return  If a split occured <v2', v2''>, else <v2, v2>.
   */
  std::pair<unsigned int, unsigned int> checkEdgeSplit(unsigned int v1, unsigned int v2, unsigned int gid);

  // roadmap id, grasp id
  // std::pair<unsigned int, unsigned int> getGraspRoadmapId(unsigned int vid) const;

  /**TODO current (non-finished) approach to storing adjacency:
   * 1. This graph is a directed graph as we can move from vertices belonging to a grasp group with many grasps
   *    to vertices belonging to a grasp group with only a single or the other remaining grasps. The opposite
   *    direction is invalid.
   * 2. A vertex is bidirectionally adjacent to vertices within its group according to the roadmap's adjacency.
   * 3. The idea is to store all cross-group adjacencies in the VertexInformation struct, while all within-group
   *    adjacencies are implicit from the roadmap.
   */

  // Stores information for a group of grasps
  struct GraspGroup
  {
    ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set;
    std::set<unsigned int> grasp_set;
    ::placement::mp::mgsearch::MultiGoalCostToGoPtr cost_to_go;
    // map roadmap id to the corresponding vertex in this group
    std::unordered_map<unsigned int, unsigned int> roadmap_id_to_vertex;
  };

  // Stores which grasp group a vertex belongs to and what roadmap node it maps to
  struct VertexInformation
  {
    unsigned int grasp_group;
    unsigned int roadmap_node_id;
    // neighboring vertices that belong to different grasp groups
    typedef std::unordered_set<unsigned int> NonGroupNeighborSet;
    NonGroupNeighborSet non_group_neighbors;
  };

private:
  ::placement::mp::mgsearch::RoadmapPtr _roadmap;
  ::placement::mp::mgsearch::GoalPathCostParameters _cost_params;
  std::pair<double, double> _quality_range;  // min, max quality of goals
  // set of grasp groups
  mutable std::vector<GraspGroup> _grasp_groups;  // mutable to allow lazy extension in getVertexId and getGraspGroupId
  // map from a grasp group string representation to its group index
  mutable std::unordered_map<std::string, unsigned int> _grasp_group_ids;

  // vertex information; 0 refers to start state
  mutable std::vector<VertexInformation> _vertex_information;  // mutable to allow lazy extension in getVertexId

  // Retrieve the vertex id given roadmap id and grasp group id (adds if not existing yet)
  unsigned int getVertexId(unsigned int rid, unsigned int grasp_group_id) const;
  // Rertieve the grasp group id given a set of grasps (adds if not existing yet)
  unsigned int getGraspGroupId(const std::set<unsigned int>& grasp_set) const;

  // get a string representation of the given grasp set
  std::string getStringRepresentation(const std::set<unsigned int>& grasp_set) const;
  // get a string representation of a single-grasp grasp set
  std::string getStringRepresentation(unsigned int gid) const;

  /************************ Iterator implementations ***************************/
  /**
   * ForwardSingleGraspIterator defines the forward adjacency (successors) for vertices that
   * are associated with a specific grasp. The class is templated on a bool that determines whether
   * the iterator queries edge costs for validity checks lazily or not.
   */
  template <bool lazy>
  class ForwardSingleGraspIterator : public NeighborIterator::IteratorImplementation
  {
  public:
    ForwardSingleGraspIterator(unsigned int v, LazyMultiGraspRoadmapGraph* const graph)
      : _(v)
      , _graph(graph)
      , _rid(_graph->_vertex_information.at(v).roadmap_node_id)
      , _grasp_group_id(_graph->_vertex_information.at(v).grasp_group)
      , _grasp_id(*(_graph->_grasp_groups.at(_grasp_group_id).grasp_set.begin()))
    {
      std::tie(_edge_iter, _end_edge_iter) = _graph->_roadmap->getNode(_rid)->getEdgesIterators();
      forwardToNextValid();
    }

    ~ForwardSingleGraspIterator() = default;

    bool equals(const std::unique_ptr<IteratorImplementation>& other) const override
    {
      auto other_casted = dynamic_cast<ForwardSingleGraspIterator<lazy>*>(other.get());
      if (!other_casted)
        return false;
      return other_casted->_graph == _graph && other_casted->_v == _v && other_casted->_edge_iter == _edge_iter;
    }

    unsigned int dereference() const override
    {
      return _graph->getVertexId(_edge_iter->first, _grasp_group_id);
    }

    void next() override
    {
      if (_edge_iter != _end_edge_iter)
      {
        ++_edge_iter;
      }
      forwardToNextValid();
    }

    // forward to next valid neighbor
    void forwardToNextValid()
    {
      std::integral_constant<bool, lazy> lazy_flag;
      while (_edge_iter != _end_edge_iter and !isEdgeValid(lazy_flag))
        ++_edge_iter;
    }

    std::unique_ptr<IteratorImplementation> copy() const override
    {
      std::unique<ForwardSingleGraspIterator<lazy>> the_copy =
          std::make_unique<ForwardSingleGraspIterator<lazy>>(_v, _graph);
      the_copy->_edge_iter = _edge_iter;
      return the_copy;
    }

    void setToEnd() override
    {
      _edge_iter = _end_edge_iter;
    }

    bool isEdgeValid(std::true_type lazy) const
    {
      return not std::isinf(_edge_iter->second->getBestKnownCost(_grasp_id));
    }

    bool isEdgeValid(std::false_type not_lazy) const
    {
      return _graph->_roadmap->computeCost(_edge_iter->second, _grasp_id).second;
    }

  private:
    const unsigned int _v;
    const LazyMultiGraspRoadmapGraph const* _graph;
    const unsigned int _grasp_id;
    const unsigned int _rid;
    const unsigned int _grasp_group_id;
    Roadmap::Node::EdgeIterator _edge_iter;
    Roadmap::Node::EdgeIterator _end_edge_iter;
  };

  /**
   * BackwardSingleGraspIterator defines the backward adjacency (predecessors) for vertices that
   * are associated with a specific grasp. The class is templated on a bool that determines whether
   * the iterator queries edge costs for validity checks lazily or not.
   */
  template <bool lazy>
  class BackwardSingleGraspIterator : public NeighborIterator::IteratorImplementation
  {
  public:
    BackwardSingleGraspIterator(unsigned int v, LazyMultiGraspRoadmapGraph* const graph)
      : _(v), _graph(graph), _forward_iter(v, graph), _end_iter(v, graph)
    {
      forwardToNextValid();
      _end_iter.setToEnd();
      _adjacent_non_group_iter = _graph->_vertex_information.at(_v).non_group_neighbors.begin();
    }

    ~BackwardSingleGraspIterator() = default;

    bool equals(const std::unique_ptr<IteratorImplementation>& other) const override
    {
      auto other_casted = dynamic_cast<BackwardSingleGraspIterator<lazy>*>(other.get());
      if (!other_casted)
        return false;
      return other_casted->_graph == _graph && other_casted->_v == _v && _forward_iter.equals(other->forward_iter));
    }

    unsigned int dereference() const override
    {
      if (_forward_iter != _end_iter)
      {
        return _forward_iter.dereference();
      }
      return *_adjacent_non_group_iter;
    }

    void next() override
    {
      if (_forward_iter != _end_iter)
      {
        _forward_iter.next();
      }
      else
      {
        _adjacent_non_group_iter++;
      }
    }

    std::unique_ptr<IteratorImplementation> copy() const override
    {
      std::unique<BackwardSingleGraspIterator<lazy>> the_copy =
          std::make_unique<BackwardSingleGraspIterator<lazy>>(_v, _graph);
      the_copy->_forward_iter = _foward_iter;
      the_copy->_adjacent_non_group_iter = _adjacent_non_group_iter;
      return the_copy;
    }

    void setToEnd() override
    {
      _forward_iter = _end_iter;
      _adjacent_non_group_iter = _graph->_vertex_information.at(_v).non_group_neighbors.end();
    }

  private:
    const unsigned int _v;
    const LazyMultiGraspRoadmapGraph const* _graph;
    ForwardSingleGraspIterator _forward_iter;
    ForwardSingleGraspIterator _end_iter;
    VertexInformation::NonGroupNeighborSet::iterator _adjacent_non_group_iter;
  };

  /**
   * ForwardMultiGraspIterator defines the forward adjacency for vertices that are associated with multiple grasps.
   * The class is templated by a bool that determines whether the iterator queries edge costs for validity
   * checks lazily or not.
   */
  template <bool lazy>
  class ForwardMultiGraspIterator : public NeighborIterator::IteratorImplementation
  {
  public:
    ForwardMultiGraspIterator(unsigned int v, LazyMultiGraspRoadmapGraph* const graph)
      : _(v)
      , _graph(graph) _rid(_graph->_vertex_information.at(v).roadmap_node_id)
      , _grasp_group_id(_graph->_vertex_information.at(v).grasp_group)
    {
      std::tie(_edge_iter, _end_edge_iter) = _graph->_roadmap->getNode(_rid)->getEdgesIterators();
      _adjacent_non_group_iter = _graph->_vertex_information.at(v).non_group_neighbors.begin();
      forwardToNextValid();
    }

    ~ForwardMultiGraspIterator() = default;

    bool equals(const std::unique_ptr<IteratorImplementation>& other) const override
    {
      auto other_casted = dynamic_cast<ForwardMultiGraspIterator<lazy>*>(other.get());
      if (!other_casted)
        return false;
      // TODO update
      return other_casted->_graph == _graph && other_casted->_v == _v && other_casted->_next_vertex == _next_vertex;
    }

    unsigned int dereference() const override
    {
      return _next_vertex;
    }

    // forward _next_vertex until it points to a valid vertex or to _v
    void forwardToNextValid()
    {
      std::integral_constant<bool, lazy> lazy_flag();
      if (_edge_iter != _end_edge_iter)
      {  // skip invalid edges and those for which there is a split
        bool valid = isEdgeValid(lazy_flag);
        while (!valid and _edge_iter != _end_edge_iter)
        {
          ++_edge_iter;
          valid = isEdgeValid(lazy_flag);
        }
        if (valid and _edge_iter != _end_edge_iter)
        {
          _next_vertex = _graph->getVertexId(_edge_iter->first, _grasp_group_id);
          return;
        }
      }
      // iterate through split edges
      while (_adjacent_non_group_iter != _graph->_vertex_information.at(_v).non_group_neighbors.end())
      {
        // TODO
        // _adjacent_non_group_iter
      }
    }

    std::unique_ptr<IteratorImplementation> copy() const override
    {
      std::unique<ForwardMultiGraspIterator<lazy>> the_copy =
          std::make_unique<ForwardMultiGraspIterator<lazy>>(_v, _graph);
      the_copy->_edge_iter = _edge_iter;
      the_copy->_next_vertex = _next_vertex;
      the_copy->_adjacent_non_group_iter = _adjacent_non_group_iter;
      return the_copy;
    }

    bool isEdgeValid(std::true_type lazy) const
    {
      // TODO return false if edge has a conditional cost on grasp from our group
      return not std::isinf(_edge_iter->second->getBestKnownCost(_grasp_id));
    }

    bool isEdgeValid(std::false_type not_lazy) const
    {
      // TODO return false if edge has a conditional cost on grasp from our group
      return _graph->_roadmap->computeCost(_edge_iter->second, _grasp_id).second;
    }

  private:
    const unsigned int _v;
    const LazyMultiGraspRoadmapGraph const* _graph;
    const unsigned int _rid;
    const unsigned int _grasp_group_id;
    unsigned int _next_vertex;
    Roadmap::Node::EdgeIterator _edge_iter;
    Roadmap::Node::EdgeIterator _end_edge_iter;
    VertexInformation::NonGroupNeighborSet::iterator _adjacent_non_group_iter;
  };
};
#endif
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement