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
    auto [layer_id, roadmap_id] = parent->toLayerRoadmapKey(v);
    auto node = parent->_roadmap->getNode(roadmap_id);
    if (!node)
    {
      _impl = std::make_unique<InvalidVertexIterator>();
    }
    else
    {
      parent->_roadmap->updateAdjacency(node);
      if (forward)
      {
        if (lazy)
        {
          if (layer_id == 0)
            _impl = std::make_unique<InLayerVertexIterator<true, true, true>>(layer_id, roadmap_id, parent);
          else
            _impl = std::make_unique<InLayerVertexIterator<true, true, false>>(layer_id, roadmap_id, parent);
        }
        else
        {
          if (layer_id == 0)
            _impl = std::make_unique<InLayerVertexIterator<false, true, true>>(layer_id, roadmap_id, parent);
          else
            _impl = std::make_unique<InLayerVertexIterator<false, true, false>>(layer_id, roadmap_id, parent);
        }
      }
      else
      {
        if (lazy)
        {
          if (layer_id == 0)
            _impl = std::make_unique<InLayerVertexIterator<true, false, true>>(layer_id, roadmap_id, parent);
          else
            _impl = std::make_unique<InLayerVertexIterator<true, false, false>>(layer_id, roadmap_id, parent);
        }
        else
        {
          if (layer_id == 0)
            _impl = std::make_unique<InLayerVertexIterator<false, false, true>>(layer_id, roadmap_id, parent);
          else
            _impl = std::make_unique<InLayerVertexIterator<false, false, false>>(layer_id, roadmap_id, parent);
        }
      }
    }
  }
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
    LazyLayeredMultiGraspRoadmapGraph<ctype> const* graph)
  : _graph(graph), _layer_iter(_graph->_layers.cbegin())
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
  auto other_casted = dynamic_cast<StartVertexIterator<forward> const*>(other);
  return other_casted != nullptr and other_casted->_layer_iter == _layer_iter;
}

template <CostCheckingType ctype>
template <bool forward>
unsigned int LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator<forward>::dereference() const
{
  if constexpr (forward)
  {
    return (*_layer_iter).start_vertex_id;
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
    unsigned int layer_id, unsigned int roadmap_id, LazyLayeredMultiGraspRoadmapGraph<ctype> const* graph)
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
  auto other_casted = dynamic_cast<InLayerVertexIterator<lazy, forward, base> const*>(other);
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
    assert(_graph->_layers.at(_layer_id).start_vertex_id == _graph->toGraphKey(_layer_id, _roadmap_id));
    assert(!_edge_to_start_returned);
    // capture special case of start state
    return 0;
  }
}

template <CostCheckingType ctype>
template <bool lazy, bool forward, bool base>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, forward, base>::next()
{
  ++_iter;
  forwardToNextValid();
  if constexpr (not forward)
  {
    if (_iter == _end and _graph->_layers.at(_layer_id).start_vertex_id == _graph->toGraphKey(_layer_id, _roadmap_id))
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
  auto the_copy = std::make_unique<InLayerVertexIterator<lazy, forward, base>>(_layer_id, _roadmap_id, _graph);
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
    if (_graph->_layers.at(_layer_id).start_vertex_id == _graph->toGraphKey(_layer_id, _roadmap_id))
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
        auto [valid, cost] = _graph->_roadmap->computeCost(_iter->second);
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
          auto [valid, cost] = _graph->_roadmap->computeCost(_iter->second, _grasp_id);
          if (valid)
            return;
        }
        else
        {
          auto [valid, cost] = _graph->_roadmap->computeCost(_iter->second);
          if (valid)
          {  // check whether we know that the node is invalid for _grasp_id
            auto node_b = _graph->_roadmap->getNode(_iter->first);
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
/*********************************** InvalidVertexIterator *******************************************/
template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::InvalidVertexIterator::InvalidVertexIterator() = default;

template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::InvalidVertexIterator::~InvalidVertexIterator() = default;

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::InvalidVertexIterator::equals(
    const typename NeighborIterator::IteratorImplementation* const other) const
{
  return dynamic_cast<InvalidVertexIterator const*>(other) != nullptr;
}

template <CostCheckingType cost_checking_type>
unsigned int LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::InvalidVertexIterator::dereference() const
{
  throw std::runtime_error("Dereferencing InvalidVertexIterator");
}

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::InvalidVertexIterator::next()
{
  throw std::runtime_error("Calling next() on InvalidVertexIterator");
}

template <CostCheckingType cost_checking_type>
std::unique_ptr<
    typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::IteratorImplementation>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::InvalidVertexIterator::copy() const
{
  return std::make_unique<InvalidVertexIterator>();
}

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::InvalidVertexIterator::setToEnd()
{
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
  , _logger(_roadmap)
{
  // copy goal set
  auto my_goal_set = goal_set->createSubset(grasp_ids);
  // create base layer
  _layers.emplace_back(std::make_shared<MultiGoalCostToGo>(my_goal_set, _cost_params), my_goal_set, grasp_ids,
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
  if (layer_id == 0)
  {  // we are on the base layer
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
  auto [begin, end] = getSuccessors(v, lazy);
  for (; begin != end; ++begin)
  {
    successors.push_back(*begin);
  }
}

template <CostCheckingType cost_checking_type>
std::pair<typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator,
          typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getSuccessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v != 0)
  {
    auto [layer_id, rid] = toLayerRoadmapKey(v);
    if (layer_id == 0)
    {
      _logger.logExpansion(rid);
    }
    else
    {
      _logger.logExpansion(rid, *_layers.at(layer_id).grasps.begin());
    }
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getSuccessors");
#endif
  return {NeighborIterator::begin(v, true, lazy, this), NeighborIterator::end(v, true, lazy, this)};
}

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getPredecessors(unsigned int v,
                                                                            std::vector<unsigned int>& predecessors,
                                                                            bool lazy)
{
  auto [begin, end] = getPredecessors(v, lazy);
  for (; begin != end; ++begin)
  {
    predecessors.push_back(*begin);
  }
}

template <CostCheckingType cost_checking_type>
std::pair<typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator,
          typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getPredecessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v != 0)
  {
    auto [layer_id, rid] = toLayerRoadmapKey(v);
    if (layer_id == 0)
    {
      _logger.logExpansion(rid);
    }
    else
    {
      _logger.logExpansion(rid, *_layers.at(layer_id).grasps.begin());
    }
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getPredecessors");
#endif
  return {NeighborIterator::begin(v, false, lazy, this), NeighborIterator::end(v, false, lazy, this)};
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
  const LayerInformation& layer = _layers.at(lid_1);
  auto node_v1 = _roadmap->getNode(rnid_1);
  // ensure v1's edges are up-to-date
  _roadmap->updateAdjacency(node_v1);
  auto edge = node_v1->getEdge(rnid_2);
  if (!edge)
    return INFINITY;
  if (lid_1 == 0)
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
    if (lid_1 == 0)
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
unsigned int LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getStartVertex() const
{
  return 0;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::isGoal(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::isGoal");
#endif
  if (v == 0)
    return false;
  auto [layer_id, rid_id] = toLayerRoadmapKey(v);
  // the layer's goal set only contains goals for the grasps associated with it
  return _layers.at(layer_id).goal_set->canBeGoal(rid_id);
}

template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGoalCost(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getGoalCost");
#endif
  auto [goal, cost] = getBestGoal(v);
  return cost;
}

template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::heuristic(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::heuristic");
#endif
  if (v == 0)
    return _start_h;
  auto [lid, rid] = toLayerRoadmapKey(v);
  // capture special case that base layer is empty
  if (lid == 0 and _layers.at(lid).grasps.empty())
    return INFINITY;
  auto node = _roadmap->getNode(rid);
  if (!node)
    return INFINITY;
  auto [h_value, goal] = _layers.at(lid).cost_to_go->nearestGoal(node->config);
  if (lid == 0)
  {  // store which grasp is repsonsible
    _grasp_for_heuristic_value[v] = goal.grasp_id;
  }
  return h_value;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::isHeuristicValid(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::isHeuristicValid");
#endif
  auto iter = _grasp_for_heuristic_value.find(v);
  if (iter != _grasp_for_heuristic_value.end())
  {  // verify that the grasp is still part of the same layer
    auto [lid, rid] = toLayerRoadmapKey(v);
    return lid == _grasp_id_to_layer_id.at(iter->second);
  }
  return true;
}

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::registerMinimalCost(unsigned int v, double cost)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::registerMinimalCost");
#endif
  // TODO is deleting this really beneficial or just unneeded overhead?
  _grasp_for_heuristic_value.erase(v);
}

// grasp-aware interface
template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGraspSpecificEdgeCost(unsigned int v1, unsigned int v2,
                                                                                       unsigned int gid)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getGraspSpecificEdgeCost");
#endif
  if (v1 == 0 or v2 == 0)
    return 0.0;
  auto [lid1, rid1] = toLayerRoadmapKey(v1);
  auto [lid2, rid2] = toLayerRoadmapKey(v2);
  if (lid1 != lid2)
    return INFINITY;
  assert(_grasp_id_to_layer_id[gid] == lid1);
  // get node
  auto node = _roadmap->getNode(rid1);
  if (!node)
    return INFINITY;
  bool split_graph = lid1 == 0;  // we only need to split layer with multiple grasps
  double return_val = INFINITY;
  if (_roadmap->isValid(node, gid))
  {
    auto edge = node->getEdge(rid2);
    if (!edge)
      return INFINITY;
    bool edge_valid = false;
    std::tie(edge_valid, return_val) = _roadmap->computeCost(edge, gid);
    split_graph &= return_val != edge->base_cost;
  }
  if (split_graph)
  {
    assert(not _layers.at(0).grasps.empty());
    // add new layer for gid
    std::set<unsigned int> single_grasp_set({gid});
    auto sub_goal_set = _layers.at(lid1).goal_set->createSubset(single_grasp_set);
    _layers.emplace_back(std::make_shared<MultiGoalCostToGo>(sub_goal_set, _cost_params, _goal_quality_range),
                         sub_goal_set, single_grasp_set, toGraphKey(_layers.size(), _roadmap_start_id));
    _grasp_id_to_layer_id[gid] = _layers.size() - 1;
    // remove gid from old layer
    _layers.at(lid1).goal_set->removeGoals(sub_goal_set->begin(), sub_goal_set->end());
    _layers.at(lid1).cost_to_go->removeGoals(sub_goal_set->begin(), sub_goal_set->end());
    _layers.at(lid1).grasps.erase(gid);
    // add changes to change caches
    _hidden_edge_changes.push_back({0, _layers.back().start_vertex_id, false});  // new edge
    for (auto iter = sub_goal_set->begin(); iter != sub_goal_set->end(); ++iter)
    {  // flag old goals as changed
      _goal_changes.push_back(toGraphKey(lid1, sub_goal_set->getRoadmapId(iter->id)));
    }
    if (_layers.at(0).grasps.empty())
    {  // if there is no grasp left for the base layer, invalidate entrance edge
      _hidden_edge_changes.push_back({0, _layers.at(0).start_vertex_id, true});
    }
  }
  return return_val;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::isGraspSpecificEdgeCostKnown(unsigned int v1,
                                                                                         unsigned int v2,
                                                                                         unsigned int gid) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::isGraspSpecificEdgeCostKnown");
#endif
  if (v1 == 0 || v2 == 0)
    return true;
  auto [lid1, rid1] = toLayerRoadmapKey(v1);
  auto [lid2, rid2] = toLayerRoadmapKey(v2);
  // if (lid1 != lid2) throw std::logic_error("The vertices " + std::to_string(v1) + ", " + std::to_string(v2) )
  assert(lid1 == lid2);
  assert(_grasp_id_to_layer_id.at(gid) == lid1);
  auto node = _roadmap->getNode(rid1);
  if (!node)
    return true;
  auto edge = node->getEdge(rid2);
  if (!edge)
    return true;
  return edge->conditional_costs.find(gid) != edge->conditional_costs.end();
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::isGraspSpecificValidityKnown(unsigned int v,
                                                                                         unsigned int gid) const
{
  if (v == 0)
    return true;
  auto [lid, rid] = toLayerRoadmapKey(v);
  assert(_grasp_id_to_layer_id.at(gid) == lid);
  auto node = _roadmap->getNode(rid);
  if (!node)
    return true;
  bool validity;
  return node->getConditionalValidity(gid, validity);
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getHiddenEdgeChanges(std::vector<EdgeChange>& edge_changes,
                                                                                 bool clear_cache)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getNewEdges");
#endif
  if (_hidden_edge_changes.empty())
    return false;
  edge_changes.insert(edge_changes.end(), _hidden_edge_changes.begin(), _hidden_edge_changes.end());
  if (clear_cache)
  {
    _hidden_edge_changes.clear();
  }
  return true;
}

template <CostCheckingType cost_checking_type>
bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGoalChanges(std::vector<unsigned int>& goal_changes,
                                                                           bool clear_cache)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getGoalChanges");
#endif
  if (_goal_changes.empty())
    return false;
  goal_changes.insert(goal_changes.end(), _goal_changes.begin(), _goal_changes.end());
  if (clear_cache)
  {
    _goal_changes.clear();
  }
  return true;
}

template <CostCheckingType cost_checking_type>
std::pair<MultiGraspMP::Goal, double>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getBestGoal(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getBestGoal");
#endif
  double min_cost = std::numeric_limits<double>::infinity();
  MultiGraspMP::Goal best_goal;
  if (v != 0)
  {  // virtual start is never a goal
    auto [layer_id, rid] = toLayerRoadmapKey(v);
    // the layer's goal set only contains goals for the grasps associated with it
    const LayerInformation& layer_info = _layers.at(layer_id);
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
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGraspRoadmapId(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getGraspRoadmapId");
#endif
  if (v == 0)
  {
    return {_roadmap_start_id, 0};  // TODO what to return?
  }
  auto [layer_id, rid] = toLayerRoadmapKey(v);
  if (layer_id == 0)
  {
    return {rid, 0};  // TODO what to return?
  }
  return {rid, *_layers.at(layer_id).grasps.begin()};
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
    unsigned int new_id = _num_graph_vertices++;
    _layer_roadmap_key_to_graph[{layer_id, roadmap_id}] = new_id;
    _graph_key_to_layer_roadmap[new_id] = {layer_id, roadmap_id};
    return new_id;
  }
  return iter->second;
}
