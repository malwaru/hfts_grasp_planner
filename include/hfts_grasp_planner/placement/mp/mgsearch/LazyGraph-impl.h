#pragma once

#if 0
// shouldn't ever be compiled but helps linter to understand the code
#include <hfts_grasp_planner/placement/mp/mgsearch/LazyGraph.h>
using namespace placement::mp::mgsearch;
#endif

// MultiGraspRoadmapGraph constants
template <CostCheckingType ctype>
const unsigned int LazyLayeredMultiGraspRoadmapGraph<ctype>::START_VERTEX_ID(0);

template <CostCheckingType ctype>
const unsigned int LazyLayeredMultiGraspRoadmapGraph<ctype>::GOAL_VERTEX_ID(1);

// template <CostCheckingType cost_checking_type>
// LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::NeighborIterator(
//     unsigned int v, bool forward, bool lazy, LazyLayeredMultiGraspRoadmapGraph<cost_checking_type> const* parent)
// {
//   if (v == 0)
//   {
//     if (forward)
//     {
//       _impl = std::make_unique<StartVertexIterator<true>>(parent);
//     }
//     else
//     {
//       _impl = std::make_unique<StartVertexIterator<false>>(parent);
//     }
//   }
//   else
//   {
//     auto [layer_id, roadmap_id] = parent->toLayerRoadmapKey(v);
//     auto node = parent->_roadmap->getNode(roadmap_id);
//     if (!node)
//     {
//       _impl = std::make_unique<InvalidVertexIterator>();
//     }
//     else
//     {
//       parent->_roadmap->updateAdjacency(node);
//       if (forward)
//       {
//         if (lazy)
//         {
//           if (layer_id == 0)
//             _impl = std::make_unique<InLayerVertexIterator<true, true, true>>(layer_id, roadmap_id, parent);
//           else
//             _impl = std::make_unique<InLayerVertexIterator<true, true, false>>(layer_id, roadmap_id, parent);
//         }
//         else
//         {
//           if (layer_id == 0)
//             _impl = std::make_unique<InLayerVertexIterator<false, true, true>>(layer_id, roadmap_id, parent);
//           else
//             _impl = std::make_unique<InLayerVertexIterator<false, true, false>>(layer_id, roadmap_id, parent);
//         }
//       }
//       else
//       {
//         if (lazy)
//         {
//           if (layer_id == 0)
//             _impl = std::make_unique<InLayerVertexIterator<true, false, true>>(layer_id, roadmap_id, parent);
//           else
//             _impl = std::make_unique<InLayerVertexIterator<true, false, false>>(layer_id, roadmap_id, parent);
//         }
//         else
//         {
//           if (layer_id == 0)
//             _impl = std::make_unique<InLayerVertexIterator<false, false, true>>(layer_id, roadmap_id, parent);
//           else
//             _impl = std::make_unique<InLayerVertexIterator<false, false, false>>(layer_id, roadmap_id, parent);
//         }
//       }
//     }
//   }
// }

/*****************************************  StartVertexIterator ************************************************/
template <CostCheckingType ctype>
LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator::StartVertexIterator(
    LazyLayeredMultiGraspRoadmapGraph<ctype> const* graph)
  : _graph(graph), _layer_iter(_graph->_layers.cbegin())
{
}

template <CostCheckingType ctype>
LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator::~StartVertexIterator() = default;

template <CostCheckingType ctype>
bool LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator::equals(
    const DynamicNeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<StartVertexIterator const*>(other);
  return other_casted != nullptr and other_casted->_layer_iter == _layer_iter;
}

template <CostCheckingType ctype>
unsigned int LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator::dereference() const
{
  return (*_layer_iter).start_vertex_id;
}

template <CostCheckingType ctype>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator::next()
{
  ++_layer_iter;
}

template <CostCheckingType ctype>
std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator::copy() const
{
  auto the_copy = std::make_unique<StartVertexIterator>(_graph);
  the_copy->_layer_iter = _layer_iter;
  return the_copy;
}

template <CostCheckingType ctype>
bool LazyLayeredMultiGraspRoadmapGraph<ctype>::StartVertexIterator::isEnd() const
{
  return _layer_iter == _graph->_layers.end();
}

/*****************************************  GoalVertexIterator ************************************************/
template <CostCheckingType ctype>
LazyLayeredMultiGraspRoadmapGraph<ctype>::GoalVertexIterator::GoalVertexIterator(
    LazyLayeredMultiGraspRoadmapGraph<ctype> const* graph)
  : _graph(graph)
  , _layer_iter(_graph->_layers.cbegin())
  , _goal_iter(_layer_iter->goal_set->begin())
  , _goal_iter_end(_layer_iter->goal_set->end())
{
  forwardToValidLayer();
}

template <CostCheckingType ctype>
LazyLayeredMultiGraspRoadmapGraph<ctype>::GoalVertexIterator::~GoalVertexIterator() = default;

template <CostCheckingType ctype>
bool LazyLayeredMultiGraspRoadmapGraph<ctype>::GoalVertexIterator::equals(
    const DynamicNeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<GoalVertexIterator const*>(other);
  return other_casted != nullptr and other_casted->_layer_iter == _layer_iter and
         other_casted->_goal_iter == _goal_iter;
}

template <CostCheckingType ctype>
unsigned int LazyLayeredMultiGraspRoadmapGraph<ctype>::GoalVertexIterator::dereference() const
{
  unsigned int rid = _layer_iter->goal_set->getRoadmapId(_goal_iter->id);
  return _graph->toGraphKey(_layer_iter->layer_id, rid);
}

template <CostCheckingType ctype>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::GoalVertexIterator::next()
{
  if (_goal_iter != _goal_iter_end)
  {
    ++_goal_iter;
  }
  if (_goal_iter == _goal_iter_end)
  {
    ++_layer_iter;
    forwardToValidLayer();
  }
}

template <CostCheckingType ctype>
std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
LazyLayeredMultiGraspRoadmapGraph<ctype>::GoalVertexIterator::copy() const
{
  auto the_copy = std::make_unique<GoalVertexIterator>(_graph);
  the_copy->_layer_iter = _layer_iter;
  the_copy->_goal_iter = _goal_iter;
  the_copy->_goal_iter_end = _goal_iter_end;
  return the_copy;
}

template <CostCheckingType ctype>
bool LazyLayeredMultiGraspRoadmapGraph<ctype>::GoalVertexIterator::isEnd() const
{
  return _layer_iter == _graph->_layers.end();
}

template <CostCheckingType ctype>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::GoalVertexIterator::forwardToValidLayer()
{
  bool valid = false;
  while (_layer_iter != _graph->_layers.cend() and !valid)
  {
    _goal_iter = _layer_iter->goal_set->begin();
    _goal_iter_end = _layer_iter->goal_set->end();
    valid = _layer_iter->goal_set->getNumGoals() > 0;
    if (!valid)
      ++_layer_iter;
  }
}

/*****************************************  InLayerVertexIterator ************************************************/
template <CostCheckingType ctype>
template <bool lazy, bool base>
LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, base>::InLayerVertexIterator(
    unsigned int layer_id, unsigned int roadmap_id, LazyLayeredMultiGraspRoadmapGraph<ctype> const* graph)
  : _graph(graph)
  , _layer_id(layer_id)
  , _roadmap_id(roadmap_id)
  , _grasp_id(*_graph->_layers.at(_layer_id).grasps.begin())
{
  auto node = _graph->_roadmap->getNode(_roadmap_id);
  assert(node);
  std::tie(_iter, _end) = node->getEdgesIterators();
  forwardToNextValid();
}

template <CostCheckingType ctype>
template <bool lazy, bool base>
LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, base>::~InLayerVertexIterator() = default;

template <CostCheckingType ctype>
template <bool lazy, bool base>
bool LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, base>::equals(
    const typename NeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<InLayerVertexIterator<lazy, base> const*>(other);
  return other_casted != nullptr and other_casted->_layer_id == _layer_id and
         other_casted->_roadmap_id == _roadmap_id and other_casted->_iter == _iter;
}

template <CostCheckingType ctype>
template <bool lazy, bool base>
unsigned int LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, base>::dereference() const
{
  assert(_iter != _end);
  return _graph->toGraphKey(_layer_id, _iter->first);
}

template <CostCheckingType ctype>
template <bool lazy, bool base>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, base>::next()
{
  ++_iter;
  forwardToNextValid();
}

template <CostCheckingType ctype>
template <bool lazy, bool base>
std::unique_ptr<typename LazyLayeredMultiGraspRoadmapGraph<ctype>::NeighborIterator::IteratorImplementation>
LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, base>::copy() const
{
  auto the_copy = std::make_unique<InLayerVertexIterator<lazy, base>>(_layer_id, _roadmap_id, _graph);
  the_copy->_iter = _iter;
  return the_copy;
}

template <CostCheckingType ctype>
template <bool lazy, bool base>
bool LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, base>::isEnd() const
{
  return _iter == _end;
}

template <CostCheckingType ctype>
template <bool lazy, bool base>
void LazyLayeredMultiGraspRoadmapGraph<ctype>::InLayerVertexIterator<lazy, base>::forwardToNextValid()
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

/*********************************** LazyLayeredMultiGraspRoadmapGraph*******************************************/
template <CostCheckingType cost_checking_type>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::LazyLayeredMultiGraspRoadmapGraph(
    ::placement::mp::mgsearch::RoadmapPtr roadmap, ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
    const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params, const std::set<unsigned int>& grasp_ids,
    unsigned int start_id)
  : _roadmap(roadmap)
  , _cost_params(cost_params)
  , _roadmap_start_id(start_id)
  , _num_graph_vertices(2)  // start and goal vertex
  , _logger(_roadmap)
{
  // copy goal set
  auto my_goal_set = goal_set->createSubset(grasp_ids);
  // create base layer
  _layers.emplace_back(std::make_shared<MultiGoalCostToGo>(my_goal_set, _cost_params), my_goal_set, grasp_ids,
                       toGraphKey(0, _roadmap_start_id), 0);
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
  if (v == START_VERTEX_ID || v == GOAL_VERTEX_ID)
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
  successors.insert(successors.begin(), begin, end);
}

template <CostCheckingType cost_checking_type>
std::pair<typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator,
          typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getSuccessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v != START_VERTEX_ID and v != GOAL_VERTEX_ID)
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
  else if (v == GOAL_VERTEX_ID)
  {
    _logger.logGoalExpansion();
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getSuccessors");
#endif
  std::unique_ptr<DynamicNeighborIterator::IteratorImplementation> impl;
  if (v == START_VERTEX_ID)
  {
    impl = std::make_unique<StartVertexIterator>(this);
  }
  else if (v != GOAL_VERTEX_ID)
  {
    auto [layer_id, roadmap_id] = toLayerRoadmapKey(v);
    auto node = _roadmap->getNode(roadmap_id);
    if (node)
    {
      _roadmap->updateAdjacency(node);
      bool goal_bridge = _layers.at(layer_id).goal_set->canBeGoal(roadmap_id);
      if (layer_id == 0)
      {
        if (lazy)
        {
          if (goal_bridge)
            impl = std::make_unique<OneMoreIterator<InLayerVertexIterator<true, true>>>(
                InLayerVertexIterator<true, true>(layer_id, roadmap_id, this), GOAL_VERTEX_ID);
          else
            impl = std::make_unique<InLayerVertexIterator<true, true>>(layer_id, roadmap_id, this);
        }
        else
        {
          if (goal_bridge)
            impl = std::make_unique<OneMoreIterator<InLayerVertexIterator<false, true>>>(
                InLayerVertexIterator<false, true>(layer_id, roadmap_id, this), GOAL_VERTEX_ID);
          else
            impl = std::make_unique<InLayerVertexIterator<false, true>>(layer_id, roadmap_id, this);
        }
      }
      else
      {
        if (lazy)
        {
          if (goal_bridge)
            impl = std::make_unique<OneMoreIterator<InLayerVertexIterator<true, false>>>(
                InLayerVertexIterator<true, false>(layer_id, roadmap_id, this), GOAL_VERTEX_ID);
          else
            impl = std::make_unique<InLayerVertexIterator<true, false>>(layer_id, roadmap_id, this);
        }
        else
        {
          if (goal_bridge)
            impl = std::make_unique<OneMoreIterator<InLayerVertexIterator<false, false>>>(
                InLayerVertexIterator<false, false>(layer_id, roadmap_id, this), GOAL_VERTEX_ID);
          else
            impl = std::make_unique<InLayerVertexIterator<false, false>>(layer_id, roadmap_id, this);
        }
      }
    }
  }
  return {NeighborIterator(impl), NeighborIterator()};
}

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getPredecessors(unsigned int v,
                                                                            std::vector<unsigned int>& predecessors,
                                                                            bool lazy)
{
  auto [begin, end] = getPredecessors(v, lazy);
  predecessors.insert(predecessors.begin(), begin, end);
}

template <CostCheckingType cost_checking_type>
std::pair<typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator,
          typename LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getPredecessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v != START_VERTEX_ID and v != GOAL_VERTEX_ID)
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
  else if (v == GOAL_VERTEX_ID)
  {
    _logger.logGoalExpansion();
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getPredecessors");
#endif
  std::unique_ptr<DynamicNeighborIterator::IteratorImplementation> impl;
  if (v == GOAL_VERTEX_ID)
  {
    impl = std::make_unique<GoalVertexIterator>(this);
  }
  else if (v != START_VERTEX_ID)
  {
    auto [layer_id, roadmap_id] = toLayerRoadmapKey(v);
    auto node = _roadmap->getNode(roadmap_id);
    if (node)
    {
      _roadmap->updateAdjacency(node);
      bool start_bridge = _layers.at(layer_id).start_vertex_id == v;
      if (layer_id == 0)
      {
        if (lazy)
        {
          if (start_bridge)
            impl = std::make_unique<OneMoreIterator<InLayerVertexIterator<true, true>>>(
                InLayerVertexIterator<true, true>(layer_id, roadmap_id, this), START_VERTEX_ID);
          else
            impl = std::make_unique<InLayerVertexIterator<true, true>>(layer_id, roadmap_id, this);
        }
        else
        {
          if (start_bridge)
            impl = std::make_unique<OneMoreIterator<InLayerVertexIterator<false, true>>>(
                InLayerVertexIterator<false, true>(layer_id, roadmap_id, this), START_VERTEX_ID);
          else
            impl = std::make_unique<InLayerVertexIterator<false, true>>(layer_id, roadmap_id, this);
        }
      }
      else
      {
        if (lazy)
        {
          if (start_bridge)
            impl = std::make_unique<OneMoreIterator<InLayerVertexIterator<true, false>>>(
                InLayerVertexIterator<true, false>(layer_id, roadmap_id, this), START_VERTEX_ID);
          else
            impl = std::make_unique<InLayerVertexIterator<true, false>>(layer_id, roadmap_id, this);
        }
        else
        {
          if (start_bridge)
            impl = std::make_unique<OneMoreIterator<InLayerVertexIterator<false, false>>>(
                InLayerVertexIterator<false, false>(layer_id, roadmap_id, this), START_VERTEX_ID);
          else
            impl = std::make_unique<InLayerVertexIterator<false, false>>(layer_id, roadmap_id, this);
        }
      }
    }
  }
  return {NeighborIterator(impl), NeighborIterator()};
}

template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getEdgeCost");
#endif
  // catch special case of start node
  if (v1 == START_VERTEX_ID || v2 == START_VERTEX_ID)
  {
    return 0.0;  // TODO could return costs for obtaining a grasp
  }
  else if (v1 == GOAL_VERTEX_ID || v2 == GOAL_VERTEX_ID)
  {
    unsigned v_not_goal = v1 == GOAL_VERTEX_ID ? v2 : v1;
    auto [goal, cost] = getBestGoal(v_not_goal);
    return cost;
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
  if (v1 == START_VERTEX_ID || v2 == START_VERTEX_ID || v1 == GOAL_VERTEX_ID || v2 == GOAL_VERTEX_ID)
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
  return START_VERTEX_ID;
}

template <CostCheckingType cost_checking_type>
unsigned int LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGoalVertex() const
{
  return GOAL_VERTEX_ID;
}

// template <CostCheckingType cost_checking_type>
// bool LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::isGoal(unsigned int v) const
// {
// #ifdef ENABLE_GRAPH_PROFILING
//   utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::isGoal");
// #endif
//   if (v == 0)
//     return false;
//   auto [layer_id, rid_id] = toLayerRoadmapKey(v);
//   // the layer's goal set only contains goals for the grasps associated with it
//   return _layers.at(layer_id).goal_set->canBeGoal(rid_id);
// }

// template <CostCheckingType cost_checking_type>
// double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getGoalCost(unsigned int v) const
// {
// #ifdef ENABLE_GRAPH_PROFILING
//   utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getGoalCost");
// #endif
//   auto [goal, cost] = getBestGoal(v);
//   return cost;
// }

template <CostCheckingType cost_checking_type>
double LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::heuristic(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::heuristic");
#endif
  if (v == START_VERTEX_ID)
    return _start_h;
  if (v == GOAL_VERTEX_ID)
    return 0.0;
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
  if (v1 == START_VERTEX_ID or v2 == START_VERTEX_ID)
    return 0.0;
  if (v1 == GOAL_VERTEX_ID or v2 == GOAL_VERTEX_ID)
  {
    unsigned int not_goal_v = v1 == GOAL_VERTEX_ID ? v2 : v1;
    auto [goal, cost] = getBestGoal(not_goal_v);
    return cost;
  }
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
    splitGraph(gid);
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
  if (v1 == START_VERTEX_ID || v2 == START_VERTEX_ID || v1 == GOAL_VERTEX_ID || v2 == GOAL_VERTEX_ID)
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
  if (v == START_VERTEX_ID or v == GOAL_VERTEX_ID)
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
std::pair<MultiGraspMP::Goal, double>
LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::getBestGoal(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("LazyLayeredMultiGraspRoadmapGraph::getBestGoal");
#endif
  double min_cost = std::numeric_limits<double>::infinity();
  MultiGraspMP::Goal best_goal;
  if (v != START_VERTEX_ID && v != GOAL_VERTEX_ID)
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
  if (v == START_VERTEX_ID || v == GOAL_VERTEX_ID)
  {
    throw std::logic_error("Can not provide grasp and roadmap id for start or goal vertex");
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

template <CostCheckingType cost_checking_type>
void LazyLayeredMultiGraspRoadmapGraph<cost_checking_type>::splitGraph(unsigned int grasp_id)
{
  assert(not _layers.at(0).grasps.empty());
  assert(_layers.at(0).grasps.find(grasp_id) != _layers.at(0).grasps.end());
  // add new layer for the given grasp
  std::set<unsigned int> single_grasp_set({grasp_id});
  auto sub_goal_set = _layers.at(0).goal_set->createSubset(single_grasp_set);
  _layers.emplace_back(std::make_shared<MultiGoalCostToGo>(sub_goal_set, _cost_params, _goal_quality_range),
                       sub_goal_set, single_grasp_set, toGraphKey(_layers.size(), _roadmap_start_id), _layers.size());
  _grasp_id_to_layer_id[grasp_id] = _layers.size() - 1;
  // remove gid from old layer
  _layers.at(0).goal_set->removeGoals(sub_goal_set->begin(), sub_goal_set->end());
  _layers.at(0).cost_to_go->removeGoals(sub_goal_set->begin(), sub_goal_set->end());
  _layers.at(0).grasps.erase(grasp_id);
  // add changes to change caches
  _hidden_edge_changes.push_back({START_VERTEX_ID, _layers.back().start_vertex_id, false});  // new edge
  for (auto iter = sub_goal_set->begin(); iter != sub_goal_set->end(); ++iter)
  {  // flag edges to goal vertex if the removed grasp-specific goals were responsible for their cost
    unsigned int old_goal_entrance_id = toGraphKey(0, sub_goal_set->getRoadmapId(iter->id));
    auto [goal, new_cost] = getBestGoal(old_goal_entrance_id);
    double old_cost = _layers.at(0).cost_to_go->qualityToGoalCost(iter->quality);
    if (new_cost > old_cost)
      _hidden_edge_changes.push_back({old_goal_entrance_id, GOAL_VERTEX_ID, true});
  }
  if (_layers.at(0).grasps.size() == 1)
  {  // if there is only a single grasp left, split that one also off
    splitGraph(*(_layers.at(0).grasps.begin()));
  }
  else if (_layers.at(0).grasps.empty())
  {  // if there is no grasp left for the base layer, invalidate entrance edge
    _hidden_edge_changes.push_back({START_VERTEX_ID, _layers.at(0).start_vertex_id, true});
  }
}
