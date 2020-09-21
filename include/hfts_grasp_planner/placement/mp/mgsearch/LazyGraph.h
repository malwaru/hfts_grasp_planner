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
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement