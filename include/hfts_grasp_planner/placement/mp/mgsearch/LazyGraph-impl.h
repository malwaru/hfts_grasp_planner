#pragma once

#if 0
// shouldn't ever be compiled but helps linter to understand the code
#include <hfts_grasp_planner/placement/mp/mgsearch/LazyGraph.h>
using namespace placement::mp::mgsearch;
#endif

// NeighborIterator
template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::NeighborIterator(
    LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator&& other)
  : _impl(std::move(other._impl))
{
}

template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::NeighborIterator(
    const LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator& other)
{
  _impl = other._impl->copy();
}

template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::NeighborIterator(
    unsigned int v, bool forward, bool lazy, LazyLayeredMultiGraspRoadmapGraph<cost_checking_type> const* parent)
{
  if (v == 0)
  {
    if (forward)
    {
      _impl = std::make_unique<StartVertexIterator<true>>(parent);
    }
    else
    {
      _impl = std::make_unique<StartVertexIterator<false>>(parent);
    }
  }
  else
  {
    // TODO
  }
  // unsigned int rid = parent->_vertex_info.at(v).roadmap_id;
  // unsigned int layer_id = parent->_vertex_info.at(v).layer_id;
  // bool goal_bridge = parent->_goal_set->canBeGoal(rid);
  // if (layer_id == 0)
  // {  // base layer
  //   if (not goal_bridge || not forward)
  //   {
  //     if (lazy)
  //     {
  //       _impl = std::make_unique<InLayerIterator<true, true>>(v, parent);
  //     }
  //     else
  //     {
  //       _impl = std::make_unique<InLayerIterator<false, true>>(v, parent);
  //     }
  //   }
  //   else
  //   {  // bridge and forward
  //     assert(goal_bridge and forward);
  //     if (lazy)
  //     {
  //       _impl = std::make_unique<LayerBridgeIterator<true, true>>(v, parent);
  //     }
  //     else
  //     {
  //       _impl = std::make_unique<LayerBridgeIterator<false, true>>(v, parent);
  //     }
  //   }
  // }
  // else
  // {  // grasp-specific layer
  //   if (forward || not goal_bridge)
  //   {
  //     if (lazy)
  //     {
  //       _impl = std::make_unique<InLayerIterator<true, false>>(v, parent);
  //     }
  //     else
  //     {
  //       _impl = std::make_unique<InLayerIterator<false, false>>(v, parent);
  //     }
  //   }
  //   else
  //   {
  //     if (lazy)
  //     {
  //       _impl = std::make_unique<LayerBridgeIterator<true, false>>(v, parent);
  //     }
  //     else
  //     {
  //       _impl = std::make_unique<LayerBridgeIterator<false, false>>(v, parent);
  //     }
  //   }
  // }
}

template <CostCheckingType cost_checking_type>
typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator&
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::operator++()
{
  _impl->next();
  return *this;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::operator==(
    const NeighborIterator& other) const
{
  return _impl->equals(other._impl.get());
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::operator!=(
    const NeighborIterator& other) const
{
  return not operator==(other);
}

template <CostCheckingType cost_checking_type>
unsigned int LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::operator*()
{
  return _impl->dereference();
}

template <CostCheckingType cost_checking_type>
typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::begin(
    unsigned int v, bool forward, bool lazy, LazyLayeredMultiGraspRoadmapGraph<cost_checking_type> const* graph)
{
  auto [layer_id, rid] = graph->toLayerRoadmapKey(v);
  auto node = graph->_roadmap->getNode(rid);
  if (!node)
  {
    throw std::logic_error("Creating neighbor iterator on non-existing node");
  }
  graph->_roadmap->updateAdjacency(node);
  return NeighborIterator(v, forward, lazy, graph);
}

template <CostCheckingType cost_checking_type>
typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::end(
    unsigned int v, bool forward, bool lazy, LazyLayeredMultiGraspRoadmapGraph<cost_checking_type> const* graph)
{
  NeighborIterator iter(v, forward, lazy, graph);
  iter._impl->setToEnd();
  return iter;
}

/*****************************************  IteratorImplementation *********************************************/
template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<
    cost_checking_type>::NeighborIterator::IteratorImplementation::~IteratorImplementation() = default;

/*****************************************  StartVertexIterator ************************************************/
template <CostCheckingType ctype>
template <bool forward>
LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator<forward>::StartVertexIterator(
    LazyLayeredMultiGraspRoadmapGraph<ctype>* graph)
  : _layer_iter(_graph->_layers.begin())
{
}

template <CostCheckingType ctype>
template <bool forward>
LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator<forward>::~StartVertexIterator() = default;

template <CostCheckingType ctype>
template <bool forward>
bool LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator<forward>::equals(
    const typename NeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<StartVertexIterator<forward>*>(other);
  return other_casted != nullptr and other_casted->_layer_iter == _layer_iter;
}

template <CostCheckingType ctype>
template <bool forward>
unsigned int LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator<forward>::dereference() const
{
  if constexpr (forward)
  {
    return _graph->layers.at(_layer_iter).start_vertex_id;
  }
  else
  {
    return 0;
  }
}

template <CostCheckingType ctype>
template <bool forward>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator<forward>::next()
{
  if constexpr (forward)
  {
    ++_layer_iter;
  }
}

template <CostCheckingType ctype>
template <bool forward>
std::unique_ptr<typename LazyLayeredMultiGraspRoadmapGraph<ctype>::NeighborIterator::IteratorImplementation>
LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator<forward>::copy() const
{
  auto the_copy = std::make_unique<StartVertexIterator<forward>>(_graph);
  the_copy->_layer_iter = _layer_iter;
  return the_copy;
}

template <CostCheckingType ctype>
template <bool forward>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator<forward>::setToEnd()
{
  if constexpr (forward)
  {
    _layer_iter = _graph->_layers.end();
  }
}

/*****************************************  InLayerVertexIterator ************************************************/
template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::InLayerVertexIterator(
    unsigned int layer_id, unsigned int roadmap_id, LazyLayeredMultiGraspRoadmapGraph<ctype>* graph)
  : _graph(graph)
  , _layer_id(layer_id)
  , _roadmap_id(roadmap_id)
  , _grasp_id(*_graph->_layers.at(_layer_id).grasps.begin())
  , _edge_to_start_returned(false)
{
  auto node = _graph->_roadmap->getNode(_roadmap_id);
  assert(node);
  std::tie(_iter, _end) = node->getEdgesIterators();
  forwardToNextValid();
}

template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::~InLayerVertexIterator() =
    default;

template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
bool LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::equals(
    const typename NeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<InLayerVertexIterator<lazy, forward, base>*>(other);
  return other_casted != nullptr and other_casted->_layer_id == _layer_id and
         other_casted->_roadmap_id == _roadmap_id and other_casted->_iter == _iter and
         _edge_to_start_returned == other_casted->_edge_to_start_returned;
}

template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
unsigned int LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::dereference() const
{
  if constexpr (forward)
  {
    return _graph->toGraphKey(_layer_id, _iter->first);
  }
  else
  {  // backwards iterator
    if (_iter != _end)
    {
      return _graph->toGraphKey(_layer_id, _iter->first);
    }
    assert(_graph->_layers.at(_layer_id).start_vertex_id == _graph->toGraphId(_layer_id, _roadmap_id));
    assert(!_edge_to_start_returned);
    // capture special case of start state
    return 0;
  }
}

template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::next()
{
  forwardToNextValid();
  if constexpr (not forward)
  {
    if (_iter == _end and _graph->_layers.at(_layer_id).start_vertex_id == _graph->toGraphId(_layer_id, _roadmap_id))
    {
      _edge_to_start_returned = true;
    }
  }
}

template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
std::unique_ptr<typename LazyLayeredMultiGraspRoadmapGraph<ctype>::NeighborIterator::IteratorImplementation>
LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::copy() const
{
  auto the_copy = std::make_unique<InLayerVertexIterator<lazy, forward, base>>(_graph);
  the_copy->_iter = _iter;
  if constexpr (not forward)
  {
    the_copy->_edge_to_start_returned = _edge_to_start_returned;
  }
  return the_copy;
}

template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::setToEnd()
{
  _iter = _end;
  if constexpr (not forward)
  {
    if (_graph->_layers.at(_layer_id).start_vertex_id == _graph->toGraphId(_layer_id, _roadmap_id))
    {  // for start vertices we require also to return 0 as neighbor
      _edge_to_start_returned = true;
    }
  }
}

template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::forwardToNextValid()
{
  // increase _iter until it points to a valid edge or _end
  for (; _iter != _end; ++_iter)
  {
    if constexpr (base)
    {  // on base layer we do not check for any grasp-specific costs
      if constexpr (lazy)
      {  // edge exists -> its not known to be invalid yet)
        return;
      }
      else
      {  // else we need to compute the base cost and skip this edge if its invalid
        auto [cost, valid] = _graph->_roadmap->computeCost(_iter->second);
        if (valid)
          return;
      }
    }
    else
    {  // query grasp-specific costs depending on ctype
      if constexpr (lazy)
      {
        if (!std::isinf(_iter->second->getBestKnownCost(_grasp_id)))
        {
          return;
        }
      }
      else
      {  // compute cost according to cost checking type
        if constexpr (ctype == WithGrasp)
        {
          auto [cost, valid] = _graph->_roadmap->computeCost(_iter->second, _grasp_id);
          if (valid)
            return;
        }
        else
        {
          auto [cost, valid] = _graph->_roadmap->computeCost(_iter->second);
          if (valid)
          {  // check whether we know that the node is invalid for _grasp_id
            auto node_b = _roadmap->getNode(_iter->first);
            bool cond_valid;
            if (node_b->getConditionalValidity(_grasp_id, cond_valid))
              valid = cond_valid;
          }
          if (valid)
            return;
        }
      }
    }
  }
}

/*********************************** LazyLayeredMultiGraspRoadmapGraph*******************************************/
template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::LazyLayeredMultiGraspRoadmapGraph(
    ::placement::mp::mgsearch::RoadmapPtr roadmap, ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
    const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params, const std::set<unsigned int>& grasp_ids,
    unsigned int start_id)
  : _roadmap(roadmap)
  , _cost_params(cost_params)
  , _roadmap_start_id(start_id)
  , _num_graph_vertices(1)  // 1 virtual vertex that connects all layers
{
  // create base layer
  _layers.emplace_back(std::make_shared<MultiGoalCostToGo>(goal_set, _cost_params), goal_set, grasp_ids,
                       toGraphKey(0, _roadmap_start_id));
  // init grasp to layer mapping
  for (auto gid : grasp_ids)
  {
    _grasp_id_to_layer_id[gid] = 0;
  }
  _start_h = _layers[0].cost_to_go->costToGo(_roadmap->getNode(_roadmap_start_id)->config);
  _goal_quality_range = _layers[0].goal_set->getGoalQualityRange();
}

template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::~LazyLayeredMultiGraspRoadmapGraph() = default;

// GraspAgnostic graph interface
template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::checkValidity(unsigned int v)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::checkValidity");
#endif
  if (v == 0)
  {
    return true;
  }
  // get layer and roadmap id for the given vertex
  auto [layer_id, roadmap_id] = toLayerRoadmapKey(v);
  auto node = _roadmap->getNode(roadmap_id);
  if (!node)
    return false;
  if (_layers.at(layer_id).grasps.size() > 1)
  {  // we are on the base layer and it still represents multiple grasps
    assert(layer_id == 0);
    return _roadmap->isValid(node);
  }
  else
  {  // the layer corresponds to a single grasp
    unsigned int grasp_id = *(_layers.at(layer_id).grasps.begin());
    if constexpr (cost_checking_type == VertexEdgeWithoutGrasp)
    {  // only case when we do not check validity with grasp
      bool grasp_validity;
      if (node->getConditionalValidity(grasp_id, grasp_validity))
      {
        return grasp_validity;
      }
      return _roadmap->isValid(node);
    }
    else
    {
      return _roadmap->isValid(node, grasp_id);
    }
  }
}

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getSuccessors(unsigned int v,
                                                                          std::vector<unsigned int>& successors,
                                                                          bool lazy)
{
}

template <CostCheckingType cost_checking_type>
std::pair<typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator,
          typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getSuccessors(unsigned int v, bool lazy)
{
}

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getPredecessors(unsigned int v,
                                                                            std::vector<unsigned int>& predecessors,
                                                                            bool lazy)
{
}

template <CostCheckingType cost_checking_type>
std::pair<typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator,
          typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getPredecessors(unsigned int v, bool lazy)
{
}

template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getEdgeCost");
#endif
  // catch special case of start node
  if (v1 == 0 || v2 == 0)
  {
    return 0.0;  // TODO could return costs for obtaining a grasp
  }
  // default case, asking the roadmap
  auto [lid_1, rnid_1] = toLayerRoadmapKey(v1);
  auto [lid_2, rnid_2] = toLayerRoadmapKey(v2);
  assert(lid_1 == lid_2);
  LayerInformation& layer = _layers.at(lid_1);
  auto node_v1 = _roadmap->getNode(rnid_1);
  // ensure v1's edges are up-to-date
  _roadmap->updateAdjacency(node_v1);
  auto edge = node_v1->getEdge(rnid_2);
  if (!edge)
    return INFINITY;
  if (layer.grasps.size() > 1)
  {  // base layer for many grasps
    if (lazy)
    {
      return edge->base_cost;
    }
    if (!checkValidity(v1))
      return INFINITY;
    if (!checkValidity(v2))
      return INFINITY;
    return _roadmap->computeCost(edge).second;
  }
  else
  {  // grasp-specific layer
    unsigned int grasp_id = *layer.grasps.begin();
    if (lazy)
    {
      return edge->getBestKnownCost(grasp_id);
    }
    if (!checkValidity(v1))
      return INFINITY;
    if (!checkValidity(v2))
      return INFINITY;
    if constexpr (cost_checking_type != WithGrasp)
    {
      _roadmap->computeCost(edge);
      return edge->getBestKnownCost(grasp_id);  // we might actually know a better cost estimate then the base
    }
    else
    {
      return _roadmap->computeCost(edge, grasp_id).second;
    }
  }
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::trueEdgeCostKnown");
#endif
  // catch special case of start node
  if (v1 == 0 || v2 == 0)
  {
    return true;
  }
  // default case, asking the roadmap
  auto [lid_1, rnid_1] = toLayerRoadmapKey(v1);
  auto [lid_2, rnid_2] = toLayerRoadmapKey(v2);
  assert(lid_1 == lid_2);
  auto node_v1 = _roadmap->getNode(rnid_1);
  if (!node_v1)
    return true;  // case the node got deleted already
  // ensure v1's edges are up-to-date
  _roadmap->updateAdjacency(node_v1);
  auto edge = node_v1->getEdge(rnid_2);
  if (!edge)
    return true;
  if constexpr (cost_checking_type != WithGrasp)
  {  // we do NOT evaluate edges with grasps in the grasp-agnostic interface
    return edge->base_evaluated;
  }
  else
  {  // we DO evaluate edges with grasps in the grasp-agnostic interface
    if (_layers.at(lid_1).grasps.size() > 1)
    {  // base layer with multiple grasps -> alsways no grasp
      return edge->base_evaluated;
    }
    else
    {  // grasp-specific layer
      unsigned int grasp_id = *_layers.at(lid_1).grasps.begin();
      return edge->conditional_costs.find(grasp_id) != edge->conditional_costs.end();
    }
  }
}

template <CostCheckingType cost_checking_type>
unsigned int LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getStartNode() const
{
  return 0;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::isGoal(unsigned int v) const
{
  if (v == 0)
    return false;
  auto [layer_id, rid_id] = toLayerRoadmapKey(v);
  // the layer's goal set only contains goals for the grasps associated with it
  return _layers.at(layer_id).goal_set->canBeGoal(rid_id);
}

template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGoalCost(unsigned int v) const
{
  auto [goal, cost] = getBestGoal(v);
  return cost;
}

template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::heuristic(unsigned int v) const
{
  if (v == 0)
    return _start_h;
  auto [lid, rid] = toLayerRoadmapKey(v);
  auto node = _roadmap->getNode(rid);
  if (!node)
    return INFINITY;
  auto [h_value, goal] = _layers.at(lid).cost_to_go->nearestGoal(node->config);
  if (_layers.at(lid).grasps.size() > 1)
  {  // store which grasp is repsonsible
    _grasp_for_heuristic_value[v] = goal.grasp_id;
  }
  return h_value;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::isHeuristicValid(unsigned int v) const
{
  auto iter = _grasp_for_heuristic_value.find(v);
  if (iter != _grasp_for_heuristic_value.end())
  {  // verify that the grasp is still part of the same layer
    auto [lid, rid] = toLayerRoadmapKey(v);
    return lid == _grasp_id_to_layer_id[iter->second];
  }
  return true;
}

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::vertexClosed(unsigned int v)
{
  _grasp_for_heuristic_value.erase(v);
}

// grasp-aware interface
template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGraspSpecificEdgeCost(unsigned int v1, unsigned int v2,
                                                                                       unsigned int gid)
{
  if (v1 == 0 or v2 == 0)
    return 0.0;
  auto [lid1, rid1] = toLayerRoadmapKey(v1);
  auto [lid2, rid2] = toLayerRoadmapKey(v2);
  if (lid1 != lid2)
    return INFINITY;
  auto& layer = _layers.at(lid1);
  assert(_grasp_id_to_layer_id[gid] == lid1);
  // get node
  auto node = _roadmap->getNode(rid1);
  if (!node)
    return INFINITY;
  bool split_graph = layer.grasps.size() > 1;  // we only need to split layer with multiple grasps
  double return_val = INFINITY;
  if (_roadmap->isValid(node, gid))
  {
    auto edge = node->getEdge(rid2);
    if (!edge)
      return INFINITY;
    bool edge_valid = false;
    std::tie(return_val, edge_valid) = _roadmap->computeCost(edge, gid);
    split_graph &= return_val != edge->base_cost;
  }
  if (split_graph)
  {
    assert(layer.grasps.size() > 1);
    // add new layer for gid
    std::set<unsigned int> single_grasp_set({gid});
    auto sub_goal_set = layer.goal_set->createSubset(single_grasp_set);
    _layers.emplace_back(std::make_shared<MultiGoalCostToGo>(sub_goal_set, _cost_params, _goal_quality_range),
                         sub_goal_set, single_grasp_set, toGraphKey(_layers.size(), _roadmap_start_id));
    _grasp_id_to_layer_id[gid] = _layers.size() - 1;
    // remove gid from old layer
    layer.goal_set->removeGoals(sub_goal_set->begin(), sub_goal_set->end());
    layer.cost_to_go->removeGoals(sub_goal_set->begin(), sub_goal_set->end());
    // add changes to change caches
    _new_edges.push_back({0, _layers.back().start_vertex_id, false});  // new edge
    for (auto iter = sub_goal_set->begin(); iter != sub_goal_set->end(); ++iter)
    {  // flag old goals as changed
      _goal_changes.push_back(toGraphKey(lid1, sub_goal_set->getRoadmapId(iter->id)));
    }
  }
  return return_val;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getNewEdges(std::vector<EdgeChange>& edge_changes,
                                                                        bool clear_cache)
{
  edge_changes.clear();
  edge_changes.insert(edge_changes.end(), _new_edges.begin(), _new_edges.end());
  if (clear_cache)
  {
    _new_edges.clear();
  }
  return edge_changes.size() > 0;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGoalChanges(std::vector<unsigned int>& goal_changes,
                                                                           bool clear_cache)
{
  goal_changes.clear();
  goal_changes.insert(goal_changes.end(), _goal_changes.begin(), _goal_changes.end());
  if (clear_cache)
  {
    _goal_changes.clear();
  }
  return goal_changes.size() > 0;
}

template <CostCheckingType cost_checking_type>
std::pair<MultiGraspMP::Goal, double> LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getBestGoal(unsigned int v)
{
  double min_cost = std::numeric_limits<double>::infinity();
  MultiGraspMP::Goal best_goal;
  if (v != 0)
  {  // virtual start is never a goal
    auto [layer_id, rid] = toLayerRoadmapKey(v);
    // the layer's goal set only contains goals for the grasps associated with it
    LayerInformation& layer_info = _layers.at(layer_id);
    auto goal_ids = layer_info.goal_set->getGoalIds(rid);
    for (auto gid : goal_ids)
    {
      auto goal = layer_info.goal_set->getGoal(gid);
      double cost = layer_info.cost_to_go->qualityToGoalCost(goal.quality);
      if (cost < min_cost)
      {
        min_cost = cost;
        best_goal = goal;
      }
    }
  }
  return {best_goal, min_cost};
}

template <CostCheckingType cost_checking_type>
std::pair<unsigned int, unsigned int>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::toLayerRoadmapKey(unsigned int graph_id) const
{
  return _graph_key_to_layer_roadmap.at(graph_id);
}

template <CostCheckingType cost_checking_type>
unsigned int LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::toGraphKey(unsigned int layer_id,
                                                                               unsigned int roadmap_id) const
{
  auto iter = _layer_roadmap_key_to_graph.find({layer_id, roadmap_id});
  if (iter == _layer_roadmap_key_to_graph.end())
  {
    // we do not have a graph node for this layer and roadmap node yet, so add a new one
    unsigned int new_id = ++_num_graph_vertices;
    _layer_roadmap_key_to_graph[{layer_id, roadmap_id}] = new_id;
    _graph_key_to_layer_roadmap[new_id] = {layer_id, roadmap_id};
    return new_id;
  }
  return iter->second;
}
