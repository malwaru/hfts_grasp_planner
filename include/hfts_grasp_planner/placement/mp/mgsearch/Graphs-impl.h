#pragma once
/************************************* OneMoreIterator ********************************/
template <typename T>
OneMoreIterator<T>::OneMoreIterator(const T& wrapped_iter, unsigned int additional_vertex)
  : _wrapped_iter(wrapped_iter), _is_end(false), _additional_vertex(additional_vertex)
{
}

template <typename T>
OneMoreIterator<T>::OneMoreIterator(const OneMoreIterator<T>& other)
  : _wrapped_iter(other._wrapped_iter), _is_end(other._is_end), _additional_vertex(other._additional_vertex)
{
}

template <typename T>
bool OneMoreIterator<T>::equals(const DynamicNeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<const OneMoreIterator<T>*>(other);
  if (!other_casted)
    return false;
  return _wrapped_iter.equals(&(other_casted->_wrapped_iter)) and _is_end == other_casted->_is_end;
}

template <typename T>
unsigned int OneMoreIterator<T>::dereference() const
{
  if (_wrapped_iter.isEnd())
  {
    return _additional_vertex;
  }
  return _wrapped_iter.dereference();
}

template <typename T>
void OneMoreIterator<T>::next()
{
  if (!_wrapped_iter.isEnd())
  {
    _wrapped_iter.next();
  }
  else
  {
    _is_end = true;
  }
}

template <typename T>
std::unique_ptr<DynamicNeighborIterator::IteratorImplementation> OneMoreIterator<T>::copy() const
{
  return std::make_unique<OneMoreIterator<T>>(*this);
}

template <typename T>
bool OneMoreIterator<T>::isEnd() const
{
  return _is_end;
}

/************************************* SingleGraspRoadmapGraph::StandardIterator ********************************/
template <bool lazy>
SingleGraspRoadmapGraph::StandardIterator<lazy>::StandardIterator(SingleGraspRoadmapGraph const* parent,
                                                                  unsigned int roadmap_id)
  : _graph(parent), _roadmap_id(roadmap_id)
{
  auto node = parent->_roadmap->getNode(roadmap_id);
  assert(node);
  auto [begin, end] = node->getEdgesIterators();
  _iter = begin;
  _end = end;
  forwardToNextValid();
}

template <bool lazy>
SingleGraspRoadmapGraph::StandardIterator<lazy>::StandardIterator(const StandardIterator<lazy>& other)
  : _graph(other._graph), _roadmap_id(other._roadmap_id), _iter(other._iter), _end(other._end)
{
}

template <bool lazy>
bool SingleGraspRoadmapGraph::StandardIterator<lazy>::equals(
    const DynamicNeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<const StandardIterator<lazy>*>(other);
  if (!other_casted)
    return false;
  return _iter == other_casted->_iter;
}

template <bool lazy>
unsigned int SingleGraspRoadmapGraph::StandardIterator<lazy>::dereference() const
{
  return _graph->toVertexId(_iter->first);
}

template <bool lazy>
void SingleGraspRoadmapGraph::StandardIterator<lazy>::next()
{
  ++_iter;
  forwardToNextValid();
}

template <bool lazy>
std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
SingleGraspRoadmapGraph::StandardIterator<lazy>::copy() const
{
  return std::make_unique<StandardIterator<lazy>>(*this);
}

template <bool lazy>
bool SingleGraspRoadmapGraph::StandardIterator<lazy>::isEnd() const
{
  return _iter == _end;
}

template <bool lazy>
unsigned int SingleGraspRoadmapGraph::StandardIterator<lazy>::getRoadmapId() const
{
  return _roadmap_id;
}

template <bool lazy>
SingleGraspRoadmapGraph const* SingleGraspRoadmapGraph::StandardIterator<lazy>::getGraph() const
{
  return _graph;
}

template <bool lazy>
void SingleGraspRoadmapGraph::StandardIterator<lazy>::forwardToNextValid()
{
  bool valid = false;
  // for (; !valid and _iter != _end; ++_iter)
  while (!valid and _iter != _end)
  {
    if constexpr (lazy)
    {
      double cost = _iter->second->getBestKnownCost(_graph->_grasp_id);
      valid = not std::isinf(cost);
    }
    else
    {  // non lazy edge evaluations
      if (_graph->_roadmap->isValid(_iter->first, _graph->_grasp_id))
      {
        auto [lvalid, cost] = _graph->_roadmap->computeCost(_iter->second, _graph->_grasp_id);
        valid = lvalid;
      }
    }
    if (!valid)
    {  // only increase _iter if it's pointing to an invalid neighbor
      ++_iter;
    }
  }
}

/************************************* SingleGraspRoadmapGraph::GoalEntranceIterator ********************************/
// template <bool lazy>
// SingleGraspRoadmapGraph::GoalEntranceIterator<lazy>::GoalEntranceIterator(SingleGraspRoadmapGraph const* parent,
//                                                                           unsigned int roadmap_id)
//   : _standard_iter(parent, roadmap_id), _is_end(false)
// {
// }

// template <bool lazy>
// SingleGraspRoadmapGraph::GoalEntranceIterator<lazy>::GoalEntranceIterator(const GoalEntranceIterator<lazy>& other)
//   : _standard_iter(other._standard_iter), _is_end(other._is_end)
// {
// }

// template <bool lazy>
// bool SingleGraspRoadmapGraph::GoalEntranceIterator<lazy>::equals(
//     const DynamicNeighborIterator::IteratorImplementation* const other) const
// {
//   auto other_casted = dynamic_cast<const GoalEntranceIterator<lazy>*>(other);
//   if (!other_casted)
//     return false;
//   return _standard_iter.equals(&(other_casted->_standard_iter)) and _is_end == other_casted->_is_end;
// }

// template <bool lazy>
// unsigned int SingleGraspRoadmapGraph::GoalEntranceIterator<lazy>::dereference() const
// {
//   if (_standard_iter.isEnd())
//   {
//     return GOAL_VERTEX_ID;
//   }
//   return _standard_iter.dereference();
// }

// template <bool lazy>
// void SingleGraspRoadmapGraph::GoalEntranceIterator<lazy>::next()
// {
//   if (!_standard_iter.isEnd())
//   {
//     _standard_iter.next();
//   }
//   else
//   {
//     _is_end = true;
//   }
// }

// template <bool lazy>
// std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
// SingleGraspRoadmapGraph::GoalEntranceIterator<lazy>::copy() const
// {
//   return std::make_unique<GoalEntranceIterator<lazy>>(*this);
// }

// template <bool lazy>
// bool SingleGraspRoadmapGraph::GoalEntranceIterator<lazy>::isEnd() const
// {
//   return _standard_iter.isEnd() and _is_end;
// }

// MultiGraspRoadmapGraph constants
template <CostCheckingType ctype>
const unsigned int MultiGraspRoadmapGraph<ctype>::START_VERTEX_ID(0);

template <CostCheckingType ctype>
const unsigned int MultiGraspRoadmapGraph<ctype>::GOAL_VERTEX_ID(1);

/************************************* MultiGraspRoadmapGraph::StandardIterator ********************************/
template <CostCheckingType cost_checking_type>
template <bool lazy>
MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::StandardIterator(
    MultiGraspRoadmapGraph<cost_checking_type> const* parent, unsigned int rid, unsigned gid)
  : _graph(parent), _roadmap_id(rid), _grasp_id(gid)
{
  auto node = parent->_roadmap->getNode(rid);
  assert(node);
  auto [begin, end] = node->getEdgesIterators();
  _iter = begin;
  _end = end;
  forwardToNextValid();
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::StandardIterator(
    const StandardIterator<lazy>& other)
  : _graph(other._graph)
  , _roadmap_id(other._roadmap_id)
  , _grasp_id(other._grasp_id)
  , _iter(other._iter)
  , _end(other._end)
{
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
bool MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::equals(
    const DynamicNeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<const StandardIterator<lazy>*>(other);
  if (!other_casted)
    return false;
  return _iter == other_casted->_iter;
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
unsigned int MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::dereference() const
{
  return _graph->toGraphKey(_grasp_id, _iter->first);
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
void MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::next()
{
  ++_iter;
  forwardToNextValid();
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::copy() const
{
  return std::make_unique<StandardIterator<lazy>>(*this);
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
bool MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::isEnd() const
{
  return _iter == _end;
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
unsigned int MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::getRoadmapId() const
{
  return _roadmap_id;
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
MultiGraspRoadmapGraph<cost_checking_type> const*
MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::getGraph() const
{
  return _graph;
}

template <CostCheckingType cost_checking_type>
template <bool lazy>
void MultiGraspRoadmapGraph<cost_checking_type>::StandardIterator<lazy>::forwardToNextValid()
{
  bool valid = false;
  // for (; !valid and _iter != _end; ++_iter)
  while (!valid and _iter != _end)
  {
    if constexpr (lazy)
    {
      double cost = _iter->second->getBestKnownCost(_grasp_id);
      valid = not std::isinf(cost);
    }
    else
    {  // non lazy edge evaluations
      if constexpr (cost_checking_type != WithGrasp)
      {  // compute for robot only
        if (_graph->_roadmap->isValid(_iter->first))
        {
          _graph->_roadmap->computeCost(_iter->second);
          valid = not std::isinf(_iter->second->getBestKnownCost(_grasp_id));  // in case we know a better cost estimate
        }
      }
      else
      {
        if (_graph->_roadmap->isValid(_iter->first, _grasp_id))
        {
          auto [lvalid, cost] = _graph->_roadmap->computeCost(_iter->second, _grasp_id);
          valid = lvalid;
        }
      }
    }
    if (!valid)
    {  // only increase _iter if it's pointing to an invalid neighbor
      ++_iter;
    }
  }
}

/************************************* MultiGraspRoadmapGraph::StartGoalBridgeIterator ********************************/
// template <CostCheckingType ctype>
// template <bool lazy>
// MultiGraspRoadmapGraph<ctype>::StartGoalBridgeIterator<lazy>::StartGoalBridgeIterator(
//     MultiGraspRoadmapGraph<ctype> const* parent, unsigned int roadmap_id, unsigned int grasp_id,
//     unsigned int special_vertex)
//   : _standard_iter(parent, roadmap_id, grasp_id), _is_end(false), _special_vertex(special_vertex)
// {
// }

// template <CostCheckingType ctype>
// template <bool lazy>
// MultiGraspRoadmapGraph<ctype>::StartGoalBridgeIterator<lazy>::StartGoalBridgeIterator(
//     const StartGoalBridgeIterator<lazy>& other)
//   : _standard_iter(other._standard_iter), _is_end(other._is_end), _special_vertex(other._special_vertex)
// {
// }

// template <CostCheckingType ctype>
// template <bool lazy>
// bool MultiGraspRoadmapGraph<ctype>::StartGoalBridgeIterator<lazy>::equals(
//     const DynamicNeighborIterator::IteratorImplementation* const other) const
// {
//   auto other_casted = dynamic_cast<const StartGoalBridgeIterator<lazy>*>(other);
//   if (!other_casted)
//     return false;
//   return _standard_iter.equals(&(other_casted->_standard_iter)) and _is_end == other_casted->_is_end;
// }

// template <CostCheckingType ctype>
// template <bool lazy>
// unsigned int MultiGraspRoadmapGraph<ctype>::StartGoalBridgeIterator<lazy>::dereference() const
// {
//   if (_standard_iter.isEnd())
//   {
//     return _special_vertex;
//   }
//   return _standard_iter.dereference();
// }

// template <CostCheckingType ctype>
// template <bool lazy>
// void MultiGraspRoadmapGraph<ctype>::StartGoalBridgeIterator<lazy>::next()
// {
//   if (!_standard_iter.isEnd())
//   {
//     _standard_iter.next();
//   }
//   else
//   {
//     _is_end = true;
//   }
// }

// template <CostCheckingType ctype>
// template <bool lazy>
// std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
// MultiGraspRoadmapGraph<ctype>::StartGoalBridgeIterator<lazy>::copy() const
// {
//   return std::make_unique<StartGoalBridgeIterator<lazy>>(*this);
// }

// template <CostCheckingType ctype>
// template <bool lazy>
// bool MultiGraspRoadmapGraph<ctype>::StartGoalBridgeIterator<lazy>::isEnd() const
// {
//   return _standard_iter.isEnd() and _is_end;
// }

/************************************* MultiGraspRoadmapGraph::GoalVertexIterator ********************************/
template <CostCheckingType ctype>
MultiGraspRoadmapGraph<ctype>::GoalVertexIterator::GoalVertexIterator(MultiGraspRoadmapGraph<ctype> const* parent)
  : _graph(parent), _iter(_graph->_goal_set->begin()), _end(_graph->_goal_set->end())
{
}

template <CostCheckingType ctype>
bool MultiGraspRoadmapGraph<ctype>::GoalVertexIterator::equals(
    const DynamicNeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<const GoalVertexIterator*>(other);
  if (!other_casted)
    return false;
  return _iter == other_casted->_iter;
}

template <CostCheckingType ctype>
unsigned int MultiGraspRoadmapGraph<ctype>::GoalVertexIterator::dereference() const
{
  assert(_iter != _end);
  unsigned int rid = _graph->_goal_set->getRoadmapId(_iter->id);
  return _graph->toGraphKey(_iter->grasp_id, rid);
}

template <CostCheckingType ctype>
void MultiGraspRoadmapGraph<ctype>::GoalVertexIterator::next()
{
  ++_iter;
}

template <CostCheckingType ctype>
std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
MultiGraspRoadmapGraph<ctype>::GoalVertexIterator::copy() const
{
  auto new_copy = std::make_unique<GoalVertexIterator>(_graph);
  new_copy->_iter = _iter;
  return new_copy;
}

template <CostCheckingType ctype>
bool MultiGraspRoadmapGraph<ctype>::GoalVertexIterator::isEnd() const
{
  return _iter == _end;
}

/************************************* MultiGraspRoadmapGraph::StartVertexIterator ********************************/
template <CostCheckingType ctype>
MultiGraspRoadmapGraph<ctype>::StartVertexIterator::StartVertexIterator(MultiGraspRoadmapGraph<ctype> const* parent)
  : _graph(parent), _iter(_graph->_grasp_ids.begin())
{
}

template <CostCheckingType ctype>
bool MultiGraspRoadmapGraph<ctype>::StartVertexIterator::equals(
    const DynamicNeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<const StartVertexIterator*>(other);
  if (!other_casted)
    return false;
  return _iter == other_casted->_iter;
}

template <CostCheckingType ctype>
unsigned int MultiGraspRoadmapGraph<ctype>::StartVertexIterator::dereference() const
{
  assert(_iter != _graph->_grasp_ids.end());
  return _graph->toGraphKey(*_iter, _graph->_roadmap_start_id);
}

template <CostCheckingType ctype>
void MultiGraspRoadmapGraph<ctype>::StartVertexIterator::next()
{
  ++_iter;
}

template <CostCheckingType ctype>
std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
MultiGraspRoadmapGraph<ctype>::StartVertexIterator::copy() const
{
  auto new_copy = std::make_unique<StartVertexIterator>(_graph);
  new_copy->_iter = _iter;
  return new_copy;
}

template <CostCheckingType ctype>
bool MultiGraspRoadmapGraph<ctype>::StartVertexIterator::isEnd() const
{
  return _iter == _graph->_grasp_ids.end();
}

/************************************* MultiGraspRoadmapGraph::NeighborIterator ********************************/
// template <CostCheckingType cost_checking_type>
// MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::NeighborIterator()
//   : _v(0), _is_end(true), _edge_to_0_returned(true)
// {
// }

// template <CostCheckingType cost_checking_type>
// MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::NeighborIterator(
//     unsigned int v, bool lazy, MultiGraspRoadmapGraph<cost_checking_type> const* parent)
//   : _v(v), _is_end(false), _lazy(lazy), _graph(parent), _edge_to_0_returned(false)
// {
//   if (_v == 0)
//   {  // initialization for special case v == 0
//     // neighbors are vertices that share the same roadmap node but with different grasp
//     _grasp_iter = _graph->_grasp_ids.begin();
//     _edge_to_0_returned = true;
//   }
//   else
//   {  // initialization for the standard case v != 0
//     std::tie(_grasp_id, _roadmap_id) = _graph->toRoadmapKey(_v);
//     auto node = _graph->_roadmap->getNode(_roadmap_id);
//     assert(node);
//     // neighbors are vertices associated with the adjacent roadmap nodes
//     std::tie(_iter, _end) = node->getEdgesIterators();
//     // and in case the roadmap node is the start node, we also have an adjacency to node 0
//     _edge_to_0_returned = _roadmap_id != _graph->_roadmap_start_id;
//     // if we do not have the special edge, we need to ensure _iter points to a valid neighbor
//     if (_edge_to_0_returned)
//     {
//       forwardToNextValid();
//     }
//   }
// }

// template <CostCheckingType cost_checking_type>
// typename MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator&
// MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::operator++()
// {
//   assert(!_is_end);
//   if (_v == 0)
//   {  // increase iterator for special case v == 0
//     ++_grasp_iter;
//     _is_end = _grasp_iter == _graph->_grasp_ids.end();
//   }
//   else
//   {  // increase iterator for default case
//     // treat the special edge back to vertex 0 for start vertices
//     if (not _edge_to_0_returned)
//     {
//       _edge_to_0_returned = true;
//     }
//     else
//     {  // increase edge iterator
//       ++_iter;
//     }
//     forwardToNextValid();
//   }
//   return (*this);
// }

// template <CostCheckingType cost_checking_type>
// bool MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::operator==(
//     const typename MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator& other) const
// {
//   if (_is_end and other._is_end)  // always equal if both are end pointers
//     return true;
//   // otherwise equal if nodes are the same and state is the same
//   return _v == other._v and other._grasp_iter == _grasp_iter and _edge_to_0_returned == other._edge_to_0_returned and
//          other._iter == _iter;
// }

// template <CostCheckingType cost_checking_type>
// bool MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::operator!=(
//     const MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator& other) const
// {
//   return not operator==(other);
// }

// template <CostCheckingType cost_checking_type>
// unsigned int MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::operator*()
// {
//   if (_v == 0)
//   {
//     unsigned int gid = *_grasp_iter;
//     return _graph->toGraphKey(gid, _graph->_roadmap_start_id);
//   }
//   // else
//   if (not _edge_to_0_returned)
//   {
//     return 0;
//   }
//   // else
//   unsigned int rid = _iter->first;
//   return _graph->toGraphKey(_grasp_id, rid);
// }

// template <CostCheckingType cost_checking_type>
// void MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::forwardToNextValid()
// {
//   assert(_edge_to_0_returned);
//   assert(_v != 0);
//   bool valid = false;
//   while (!valid and _iter != _end)
//   {
//     if (_lazy)
//     {
//       double cost = _iter->second->getBestKnownCost(_grasp_id);
//       valid = not std::isinf(cost);
//     }
//     else
//     {
//       if constexpr (cost_checking_type != WithGrasp)
//       {
//         if (_graph->_roadmap->isValid(_iter->first))
//         {
//           _graph->_roadmap->computeCost(_iter->second);
//           valid = not std::isinf(_iter->second->getBestKnownCost(_grasp_id));  // in case we know a better cost
//           estimate
//         }
//       }
//       else
//       {
//         if (_graph->_roadmap->isValid(_iter->first, _grasp_id))
//         {
//           auto [lvalid, cost] = _graph->_roadmap->computeCost(_iter->second, _grasp_id);
//           valid = lvalid;
//         }
//       }
//     }
//     if (not valid)
//     {
//       ++_iter;
//     }
//   }
//   _is_end = _iter == _end;
// }

// template <CostCheckingType cost_checking_type>
// typename MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator
// MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::begin(
//     unsigned int v, bool lazy, MultiGraspRoadmapGraph<cost_checking_type> const* graph)
// {
//   if (v != 0)
//   {
//     // update roadmap
//     auto [gid, rid] = graph->toRoadmapKey(v);
//     auto node = graph->_roadmap->getNode(rid);
//     if (!node)
//       return NeighborIterator();
//     graph->_roadmap->updateAdjacency(node);
//   }
//   return NeighborIterator(v, lazy, graph);
// }

// template <CostCheckingType cost_checking_type>
// typename MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator
// MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator::end(
//     unsigned int v, MultiGraspRoadmapGraph<cost_checking_type> const* graph)
// {
//   // NeighborIterator end_iter(v, true, graph);
//   // end_iter._grasp_iter = graph->_grasp_ids.end();  // covers end condition for v = 0
//   // end_iter._iter = end_iter._end;                  // covers end condition for v != 0
//   // end_iter._edge_to_0_returned = true;             // covers end condition for v != 0
//   // return end_iter;
//   return NeighborIterator();
// }

/************************************* MultiGraspRoadmapGraph ********************************/
template <CostCheckingType cost_checking_type>
MultiGraspRoadmapGraph<cost_checking_type>::MultiGraspRoadmapGraph(
    RoadmapPtr roadmap, MultiGraspGoalSetPtr goal_set,
    const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params, const std::set<unsigned int>& grasp_ids,
    unsigned int start_id)
  : _roadmap(roadmap)
  , _goal_set(goal_set)
  , _all_grasps_cost_to_go(goal_set, cost_params)
  , _grasp_ids(grasp_ids)
  , _roadmap_start_id(start_id)
  , _num_graph_nodes(1)
  , _logger(_roadmap)
{
  auto quality_range = goal_set->getGoalQualityRange();
  for (unsigned int gid : grasp_ids)
  {
    // TODO MultiGoalCostToGo has some issue that prevents emplace to work properly. Haven't figured out what yet
    // _individual_cost_to_go.emplace(gid, MultiGoalCostToGo(goal_set->createSubset({gid}), cost_params,
    // quality_range));
    _individual_cost_to_go[gid] =
        std::make_shared<MultiGoalCostToGo>(goal_set->createSubset({gid}), cost_params, quality_range);
  }
}

template <CostCheckingType cost_checking_type>
MultiGraspRoadmapGraph<cost_checking_type>::~MultiGraspRoadmapGraph() = default;

template <CostCheckingType cost_checking_type>
bool MultiGraspRoadmapGraph<cost_checking_type>::checkValidity(unsigned int v)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::checkValidity");
#endif
  if (v == START_VERTEX_ID)
  {  // special case for start node which is not associated with any grasp
    return _roadmap->isValid(_roadmap->getNode(_roadmap_start_id));
  }
  else if (v == GOAL_VERTEX_ID)
    return true;
  auto [grasp_id, roadmap_id] = toRoadmapKey(v);
  auto node = _roadmap->getNode(roadmap_id);
  if (!node)
    return false;
  if constexpr (cost_checking_type == VertexEdgeWithoutGrasp)
  {
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

template <CostCheckingType cost_checking_type>
void MultiGraspRoadmapGraph<cost_checking_type>::getSuccessors(unsigned int v, std::vector<unsigned int>& successors,
                                                               bool lazy)
{
  successors.clear();
  auto [begin, end] = getSuccessors(v, lazy);
  successors.insert(successors.begin(), begin, end);
}

template <CostCheckingType cost_checking_type>
std::pair<typename MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator,
          typename MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator>
MultiGraspRoadmapGraph<cost_checking_type>::getSuccessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v == START_VERTEX_ID)
  {
    _logger.logExpansion(_roadmap_start_id);
  }
  else if (v == GOAL_VERTEX_ID)
  {
    _logger.logGoalExpansion();
  }
  else
  {
    auto [rid, gid] = getGraspRoadmapId(v);
    _logger.logExpansion(rid, gid);
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::getSuccessors");
#endif
  std::unique_ptr<DynamicNeighborIterator::IteratorImplementation> impl(nullptr);
  if (v == START_VERTEX_ID)
  {
    impl = std::make_unique<StartVertexIterator>(this);
  }
  else if (v != GOAL_VERTEX_ID)
  {
    auto [gid, rid] = toRoadmapKey(v);
    auto node = _roadmap->getNode(rid);
    if (node)
    {
      _roadmap->updateAdjacency(node);
      bool is_goal = _goal_set->isGoal(rid, gid);
      if (is_goal)
      {
        if (lazy)
        {
          impl = std::make_unique<OneMoreIterator<StandardIterator<true>>>(StandardIterator<true>(this, rid, gid),
                                                                           GOAL_VERTEX_ID);
        }
        else
        {
          impl = std::make_unique<OneMoreIterator<StandardIterator<false>>>(StandardIterator<false>(this, rid, gid),
                                                                            GOAL_VERTEX_ID);
        }
      }
      else
      {  // standard
        if (lazy)
        {
          impl = std::make_unique<StandardIterator<true>>(this, rid, gid);
        }
        else
        {
          impl = std::make_unique<StandardIterator<false>>(this, rid, gid);
        }
      }
    }
  }
  return {NeighborIterator(impl), NeighborIterator()};
}

template <CostCheckingType cost_checking_type>
void MultiGraspRoadmapGraph<cost_checking_type>::getPredecessors(unsigned int v,
                                                                 std::vector<unsigned int>& predecessors, bool lazy)
{
  predecessors.clear();
  auto [begin, end] = getPredecessors(v, lazy);
  predecessors.insert(predecessors.begin(), begin, end);
}

template <CostCheckingType cost_checking_type>
std::pair<typename MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator,
          typename MultiGraspRoadmapGraph<cost_checking_type>::NeighborIterator>
MultiGraspRoadmapGraph<cost_checking_type>::getPredecessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v == START_VERTEX_ID)
  {
    _logger.logExpansion(_roadmap_start_id);
  }
  else if (v == GOAL_VERTEX_ID)
  {
    _logger.logGoalExpansion();
  }
  else
  {
    auto [rid, gid] = getGraspRoadmapId(v);
    _logger.logExpansion(rid, gid);
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::getPredecessors");
#endif
  std::unique_ptr<DynamicNeighborIterator::IteratorImplementation> impl(nullptr);
  if (v == GOAL_VERTEX_ID)
  {
    impl = std::make_unique<GoalVertexIterator>(this);
  }
  else if (v != START_VERTEX_ID)
  {
    auto [gid, rid] = toRoadmapKey(v);
    auto node = _roadmap->getNode(rid);
    if (node)
    {
      _roadmap->updateAdjacency(node);
      if (rid == _roadmap_start_id)
      {
        if (lazy)
        {
          impl = std::make_unique<OneMoreIterator<StandardIterator<true>>>(StandardIterator<true>(this, rid, gid),
                                                                           START_VERTEX_ID);
        }
        else
        {
          impl = std::make_unique<OneMoreIterator<StandardIterator<false>>>(StandardIterator<false>(this, rid, gid),
                                                                            START_VERTEX_ID);
        }
      }
      else
      {  // standard
        if (lazy)
        {
          impl = std::make_unique<StandardIterator<true>>(this, rid, gid);
        }
        else
        {
          impl = std::make_unique<StandardIterator<false>>(this, rid, gid);
        }
      }
    }
  }
  return {NeighborIterator(impl), NeighborIterator()};
}

template <CostCheckingType cost_checking_type>
double MultiGraspRoadmapGraph<cost_checking_type>::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::getEdgeCost");
#endif
  // catch special case of start node
  if (v1 == START_VERTEX_ID || v2 == START_VERTEX_ID)
  {
    unsigned int non_start = v1 == START_VERTEX_ID ? v2 : v1;
    auto [grasp_id, rid] = toRoadmapKey(non_start);
    if (rid != _roadmap_start_id or !checkValidity(non_start))
      return INFINITY;
    return 0.0;  // TODO could return here costs for obtaining a grasp
  }
  else if (v1 == GOAL_VERTEX_ID || v2 == GOAL_VERTEX_ID)
  {
    unsigned int actual_v = v1 == GOAL_VERTEX_ID ? v2 : v1;
    auto [gid, rid] = toRoadmapKey(actual_v);
    auto [goal_id, is_goal] = _goal_set->getGoalId(rid, gid);
    if (not is_goal)  // no validity check needed as goals are checked in advance
    {
      return std::numeric_limits<double>::infinity();
    }
    return _all_grasps_cost_to_go.qualityToGoalCost(_goal_set->getGoal(goal_id).quality);
  }
  // default case, asking the roadmap
  auto [grasp_1, rnid_1] = toRoadmapKey(v1);
  auto [grasp_2, rnid_2] = toRoadmapKey(v2);
  assert(grasp_1 == grasp_2);
  auto node_v1 = _roadmap->getNode(rnid_1);
  // ensure v1's edges are up-to-date
  _roadmap->updateAdjacency(node_v1);
  auto edge = node_v1->getEdge(rnid_2);
  if (!edge)
    return INFINITY;
  if (lazy)
  {
    return edge->getBestKnownCost(grasp_1);
  }
  if (!checkValidity(v1))
    return INFINITY;
  if (!checkValidity(v2))
    return INFINITY;
  if constexpr (cost_checking_type != WithGrasp)
  {
    _roadmap->computeCost(edge);
    return edge->getBestKnownCost(grasp_1);  // we might actually know a better cost estimate then the base
  }
  else
  {
    return _roadmap->computeCost(edge, grasp_1).second;
  }
}

template <CostCheckingType cost_checking_type>
bool MultiGraspRoadmapGraph<cost_checking_type>::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::trueEdgeCostKnown");
#endif
  // catch special case of start and goal vertex
  if (v1 == START_VERTEX_ID || v2 == START_VERTEX_ID || v1 == GOAL_VERTEX_ID || v2 == GOAL_VERTEX_ID)
  {
    return true;
  }
  // default case, asking the roadmap
  auto [grasp_1, rnid_1] = toRoadmapKey(v1);
  auto [grasp_2, rnid_2] = toRoadmapKey(v2);
  assert(grasp_1 == grasp_2);
  auto node_v1 = _roadmap->getNode(rnid_1);
  if (!node_v1)
    return true;  // case the node got deleted already
  // ensure v1's edges are up-to-date
  _roadmap->updateAdjacency(node_v1);
  auto edge = node_v1->getEdge(rnid_2);
  if (!edge)
    return true;
  if constexpr (cost_checking_type != WithGrasp)
  {
    return edge->base_evaluated;
  }
  else
  {
    return edge->conditional_costs.find(grasp_1) != edge->conditional_costs.end();
  }
}

template <CostCheckingType cost_checking_type>
unsigned int MultiGraspRoadmapGraph<cost_checking_type>::getStartVertex() const
{
  return START_VERTEX_ID;
}

template <CostCheckingType cost_checking_type>
unsigned int MultiGraspRoadmapGraph<cost_checking_type>::getGoalVertex() const
{
  return GOAL_VERTEX_ID;
}

// template <CostCheckingType cost_checking_type>
// bool MultiGraspRoadmapGraph<cost_checking_type>::isGoal(unsigned int v) const
// {
// #ifdef ENABLE_GRAPH_PROFILING
//   utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::isGoal");
// #endif
//   if (v == 0)
//     return false;
//   auto [grasp_id, rnid] = toRoadmapKey(v);
//   return _goal_set->isGoal(rnid, grasp_id);
// }

// template <CostCheckingType cost_checking_type>
// double MultiGraspRoadmapGraph<cost_checking_type>::getGoalCost(unsigned int v) const
// {
// #ifdef ENABLE_GRAPH_PROFILING
//   utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::getGoalCost");
// #endif
//   if (v == 0)
//   {  // can not be a goal
//     return std::numeric_limits<double>::infinity();
//   }
//   auto [grasp_id, rnid] = toRoadmapKey(v);
//   auto node = _roadmap->getNode(rnid);
//   assert(node);
//   auto [goal_id, is_goal] = _goal_set->getGoalId(node->uid, grasp_id);
//   if (not is_goal)
//   {
//     return std::numeric_limits<double>::infinity();
//   }
//   return _all_grasps_cost_to_go.qualityToGoalCost(_goal_set->getGoal(goal_id).quality);
// }

template <CostCheckingType cost_checking_type>
double MultiGraspRoadmapGraph<cost_checking_type>::heuristic(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::heuristic");
#endif
  if (v == START_VERTEX_ID)
  {
    auto node = _roadmap->getNode(_roadmap_start_id);
    return _all_grasps_cost_to_go.costToGo(node->config);
  }
  else if (v == GOAL_VERTEX_ID)
  {
    return 0.0;
  }
  auto [grasp_id, rnid] = toRoadmapKey(v);
  auto node = _roadmap->getNode(rnid);
  if (!node)
    return INFINITY;
  return _individual_cost_to_go.at(grasp_id)->costToGo(node->config);
}

template <CostCheckingType cost_checking_type>
double MultiGraspRoadmapGraph<cost_checking_type>::getEdgeCostWithGrasp(unsigned int v1, unsigned int v2)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::getEdgeCostWithGrasp");
#endif
  if constexpr (cost_checking_type != WithGrasp)
  {
    // catch special case of start node
    if (v1 == START_VERTEX_ID || v2 == START_VERTEX_ID)
    {
      unsigned int non_start = std::max(v1, v2);
      auto [grasp_id, rnid] = toRoadmapKey(non_start);
      if (!_roadmap->isValid(rnid, grasp_id))
        return INFINITY;
      return 0.0;  // TODO could return here costs for obtaining a grasp
    }
    else if (v1 == GOAL_VERTEX_ID || v2 == GOAL_VERTEX_ID)
    {
      unsigned int actual_v = v1 == GOAL_VERTEX_ID ? v2 : v1;
      auto [gid, rid] = toRoadmapKey(actual_v);
      auto [goal_id, is_goal] = _goal_set->getGoalId(rid, gid);
      if (not is_goal)
      {
        return std::numeric_limits<double>::infinity();
      }
      return _all_grasps_cost_to_go.qualityToGoalCost(_goal_set->getGoal(goal_id).quality);
    }
    // default case, asking the roadmap
    auto [grasp_1, rnid_1] = toRoadmapKey(v1);
    auto [grasp_2, rnid_2] = toRoadmapKey(v2);
    assert(grasp_1 == grasp_2);
    auto node_v1 = _roadmap->getNode(rnid_1);
    // ensure v1's edges are up-to-date
    _roadmap->updateAdjacency(node_v1);
    auto edge = node_v1->getEdge(rnid_2);
    if (!edge)
      return INFINITY;
    if (!_roadmap->isValid(rnid_1, grasp_1))
      return INFINITY;
    if (!_roadmap->isValid(rnid_2, grasp_1))
      return INFINITY;
    return _roadmap->computeCost(edge, grasp_1).second;
  }
  else
  {
    return getEdgeCost(v1, v2, false);
  }
}

template <CostCheckingType cost_checking_type>
bool MultiGraspRoadmapGraph<cost_checking_type>::trueEdgeCostWithGraspKnown(unsigned int v1, unsigned int v2) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::trueEdgeCostWithGraspKnown");
#endif
  if constexpr (cost_checking_type != WithGrasp)
  {
    // catch special case of start node
    if (v1 == START_VERTEX_ID || v2 == START_VERTEX_ID)
    {  // we know the edge cost if we know validity of both nodes (0 is always valid)
      return trueValidityWithGraspKnown(v1 == START_VERTEX_ID ? v2 : v1);
    }
    else if (v1 == GOAL_VERTEX_ID || v2 == GOAL_VERTEX_ID)
      return true;
    // default case, asking the roadmap
    auto [grasp_1, rnid_1] = toRoadmapKey(v1);
    auto [grasp_2, rnid_2] = toRoadmapKey(v2);
    assert(grasp_1 == grasp_2);
    auto node_v1 = _roadmap->getNode(rnid_1);
    if (!node_v1)
      return true;  // case the node got deleted already
    // ensure v1's edges are up-to-date
    _roadmap->updateAdjacency(node_v1);
    auto edge = node_v1->getEdge(rnid_2);
    if (!edge)
      return true;
    return edge->conditional_costs.find(grasp_1) != edge->conditional_costs.end();
  }
  else
  {
    return trueEdgeCostKnown(v1, v2);
  }
}

template <CostCheckingType cost_checking_type>
bool MultiGraspRoadmapGraph<cost_checking_type>::trueValidityWithGraspKnown(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("MultiGraspRoadmapGraph::trueValidityWithGraspKnown");
#endif
  if (v == START_VERTEX_ID || v == GOAL_VERTEX_ID)
    return true;
  // get roadmap graph
  auto [grasp_id, rnid] = toRoadmapKey(v);
  auto node = _roadmap->getNode(rnid);
  if (!node)
    return true;
  bool validity;
  return node->getConditionalValidity(grasp_id, validity);
}

template <CostCheckingType cost_checking_type>
std::pair<unsigned int, unsigned int>
MultiGraspRoadmapGraph<cost_checking_type>::getGraspRoadmapId(unsigned int vid) const
{
  assert(vid != GOAL_VERTEX_ID);
  if (vid == START_VERTEX_ID)
  {
    return {_roadmap_start_id, 0};  // TODO what makes sense to return here for the grasp?
  }
  auto [grasp_id, rid] = toRoadmapKey(vid);
  return {rid, grasp_id};
}

template <CostCheckingType cost_checking_type>
std::pair<unsigned int, unsigned int>
MultiGraspRoadmapGraph<cost_checking_type>::toRoadmapKey(unsigned int graph_id) const
{
  assert(graph_id != GOAL_VERTEX_ID);
  auto iter = _graph_key_to_roadmap.find(graph_id);
  assert(iter != _graph_key_to_roadmap.end());
  return iter->second;
}

template <CostCheckingType cost_checking_type>
unsigned int
MultiGraspRoadmapGraph<cost_checking_type>::toGraphKey(const std::pair<unsigned int, unsigned int>& roadmap_id) const
{
  auto iter = _roadmap_key_to_graph.find(roadmap_id);
  if (iter == _roadmap_key_to_graph.end())
  {
    // we do not have a graph node for this grasp and roadmap node yet, so add a new one
    unsigned int new_id = ++_num_graph_nodes;
    _roadmap_key_to_graph[roadmap_id] = new_id;
    _graph_key_to_roadmap[new_id] = roadmap_id;
    return new_id;
  }
  return iter->second;
}

template <CostCheckingType cost_checking_type>
unsigned int MultiGraspRoadmapGraph<cost_checking_type>::toGraphKey(unsigned int grasp_id, unsigned int node_id) const
{
  return toGraphKey({grasp_id, node_id});
}

/**************************** FoldedMultiGraspRoadmapGraph **************************************/
template <BackwardsHeuristicType htype>
FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::NeighborIterator(
    FoldedMultiGraspRoadmapGraph::NeighborIterator&& other)
  : _impl(std::move(other._impl))
{
}

template <BackwardsHeuristicType htype>
FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::NeighborIterator(
    const FoldedMultiGraspRoadmapGraph::NeighborIterator& other)
{
  _impl = other._impl->copy();
}

template <BackwardsHeuristicType htype>
FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::NeighborIterator(
    unsigned int v, bool forward, bool lazy, FoldedMultiGraspRoadmapGraph<htype> const* parent)
{
  unsigned int rid = parent->_vertex_info.at(v).roadmap_id;
  unsigned int layer_id = parent->_vertex_info.at(v).layer_id;
  bool goal_bridge = parent->_goal_set->canBeGoal(rid);
  if (layer_id == 0)
  {  // base layer
    if (not goal_bridge || not forward)
    {
      if (lazy)
      {
        _impl = std::make_unique<InLayerIterator<true, true>>(v, parent);
      }
      else
      {
        _impl = std::make_unique<InLayerIterator<false, true>>(v, parent);
      }
    }
    else
    {  // bridge and forward
      assert(goal_bridge and forward);
      if (lazy)
      {
        _impl = std::make_unique<LayerBridgeIterator<true, true>>(v, parent);
      }
      else
      {
        _impl = std::make_unique<LayerBridgeIterator<false, true>>(v, parent);
      }
    }
  }
  else
  {  // grasp-specific layer
    if (forward || not goal_bridge)
    {
      if (lazy)
      {
        _impl = std::make_unique<InLayerIterator<true, false>>(v, parent);
      }
      else
      {
        _impl = std::make_unique<InLayerIterator<false, false>>(v, parent);
      }
    }
    else
    {
      if (lazy)
      {
        _impl = std::make_unique<LayerBridgeIterator<true, false>>(v, parent);
      }
      else
      {
        _impl = std::make_unique<LayerBridgeIterator<false, false>>(v, parent);
      }
    }
  }
}

template <BackwardsHeuristicType htype>
typename FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator&
FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::operator++()
{
  _impl->next();
  return *this;
}

template <BackwardsHeuristicType htype>
bool FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::operator==(const NeighborIterator& other) const
{
  return _impl->equals(other._impl.get());
}

template <BackwardsHeuristicType htype>
bool FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::operator!=(const NeighborIterator& other) const
{
  return not operator==(other);
}

template <BackwardsHeuristicType htype>
unsigned int FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::operator*()
{
  return _impl->dereference();
}

template <BackwardsHeuristicType htype>
typename FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator
FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::begin(unsigned int v, bool forward, bool lazy,
                                                             FoldedMultiGraspRoadmapGraph<htype> const* graph)
{
  auto [rid_gid_pair, valid_gid] = graph->getGraspRoadmapId(v);
  auto node = graph->_roadmap->getNode(rid_gid_pair.first);
  if (!node)
  {
    throw std::logic_error("Creating neighbor iterator on non-existing node");
  }
  graph->_roadmap->updateAdjacency(node);
  return NeighborIterator(v, forward, lazy, graph);
}

template <BackwardsHeuristicType htype>
typename FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator
FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::end(unsigned int v, bool forward, bool lazy,
                                                           FoldedMultiGraspRoadmapGraph<htype> const* graph)
{
  NeighborIterator iter(v, forward, lazy, graph);
  iter._impl->setToEnd();
  return iter;
}

template <BackwardsHeuristicType htype>
FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::IteratorImplementation::~IteratorImplementation() = default;

template <BackwardsHeuristicType htype>
FoldedMultiGraspRoadmapGraph<htype>::FoldedMultiGraspRoadmapGraph(
    ::placement::mp::mgsearch::RoadmapPtr roadmap, ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
    const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params, unsigned int start_id)
  : _roadmap(roadmap)
  , _goal_set(goal_set)
  , _cost_to_go(goal_set, cost_params)
  , _lower_bound(cost_params.path_cost)
  , _start_rid(start_id)
  , _logger(_roadmap)
{
  _vertex_info.emplace_back(_start_rid, 0);
  _vertex_ids[{_start_rid, 0}] = 0;
}

// // GraspAgnostic graph interface
template <BackwardsHeuristicType htype>
bool FoldedMultiGraspRoadmapGraph<htype>::checkValidity(unsigned int v)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("FoldedMultiGraspRoadmapGraph::checkValidity");
#endif
  auto node = _roadmap->getNode(_vertex_info.at(v).roadmap_id);
  if (!node)
    return false;
  if (_vertex_info.at(v).layer_id > 0)
  {  // check validity with grasp
    return _roadmap->isValid(node, _vertex_info.at(v).layer_id - 1);
  }
  // without grasp
  return _roadmap->isValid(node);
}

template <BackwardsHeuristicType htype>
void FoldedMultiGraspRoadmapGraph<htype>::getSuccessors(unsigned int v, std::vector<unsigned int>& successors,
                                                        bool lazy)
{
  successors.clear();
  auto [begin, end] = getSuccessors(v, lazy);
  successors.insert(successors.begin(), begin, end);
}

template <BackwardsHeuristicType htype>
std::pair<typename FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator,
          typename FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator>
FoldedMultiGraspRoadmapGraph<htype>::getSuccessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  auto [rid_gid_pair, gid_valid] = getGraspRoadmapId(v);
  if (gid_valid)
  {
    _logger.logExpansion(rid_gid_pair.first, rid_gid_pair.second);
  }
  else
  {
    _logger.logExpansion(rid_gid_pair.first);
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("FoldedMultiGraspRoadmapGraph::getSuccessors");
#endif
  auto begin = FoldedMultiGraspRoadmapGraph::NeighborIterator::begin(v, true, lazy, this);
  auto end = FoldedMultiGraspRoadmapGraph::NeighborIterator::end(v, true, lazy, this);
  return {std::move(begin), std::move(end)};
}

template <BackwardsHeuristicType htype>
void FoldedMultiGraspRoadmapGraph<htype>::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors,
                                                          bool lazy)
{
  predecessors.clear();
  auto [begin, end] = getPredecessors(v, lazy);
  predecessors.insert(predecessors.begin(), begin, end);
}

template <BackwardsHeuristicType htype>
std::pair<typename FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator,
          typename FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator>
FoldedMultiGraspRoadmapGraph<htype>::getPredecessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  auto [rid_gid_pair, gid_valid] = getGraspRoadmapId(v);
  if (gid_valid)
  {
    _logger.logExpansion(rid_gid_pair.first, rid_gid_pair.second);
  }
  else
  {
    _logger.logExpansion(rid_gid_pair.first);
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("FoldedMultiGraspRoadmapGraph::getPredecessors");
#endif
  auto begin = FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::begin(v, false, lazy, this);
  auto end = FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::end(v, false, lazy, this);
  return {std::move(begin), std::move(end)};
}

template <BackwardsHeuristicType htype>
double FoldedMultiGraspRoadmapGraph<htype>::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("FoldedMultiGraspRoadmapGraph::getEdgeCost");
#endif
  if (_vertex_info.at(v1).layer_id == _vertex_info.at(v2).layer_id)
  {  // we are within the same layer -> default adjacency
    auto node = _roadmap->getNode(_vertex_info.at(v1).roadmap_id);
    if (!node)
      return std::numeric_limits<double>::infinity();
    _roadmap->updateAdjacency(node);
    auto edge = node->getEdge(_vertex_info.at(v2).roadmap_id);
    if (!edge)
      return std::numeric_limits<double>::infinity();

    if (_vertex_info.at(v1).layer_id > 0)
    {  // check grasp specific cost
      unsigned int gid = _vertex_info.at(v1).layer_id - 1;
      if (lazy)
      {
        return edge->getBestKnownCost(gid);
      }
      return _roadmap->computeCost(edge, gid).second;
    }
    else if (lazy)  // base layer
    {
      return edge->base_cost;
    }
    return _roadmap->computeCost(edge).second;
  }  // cross layer cost
  if ((_vertex_info.at(v1).roadmap_id != _vertex_info.at(v2).roadmap_id) ||
      (!_goal_set->isGoal(_vertex_info.at(v2).roadmap_id, _vertex_info.at(v2).layer_id - 1)))
  {
    return std::numeric_limits<double>::infinity();
  }
  assert(_registered_costs.find(v1) != _registered_costs.end());
  auto [goal_id, is_goal] = _goal_set->getGoalId(_vertex_info.at(v1).roadmap_id, _vertex_info.at(v2).layer_id - 1);
  assert(is_goal);
  return -_registered_costs.at(v1) + _cost_to_go.qualityToGoalCost(_goal_set->getGoal(goal_id).quality);
}

template <BackwardsHeuristicType htype>
bool FoldedMultiGraspRoadmapGraph<htype>::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("FoldedMultiGraspRoadmapGraph::trueEdgeCostKnown");
#endif
  if (_vertex_info.at(v1).layer_id == _vertex_info.at(v2).layer_id)
  {
    auto node = _roadmap->getNode(_vertex_info.at(v1).roadmap_id);
    if (!node)
      return true;
    _roadmap->updateAdjacency(node);
    auto edge = node->getEdge(_vertex_info.at(v2).roadmap_id);
    if (!edge)
      return true;

    if (_vertex_info.at(v1).layer_id > 0)
    {  // check grasp specific cost
      unsigned int gid = _vertex_info.at(v1).layer_id - 1;
      return edge->conditional_costs.find(gid) != edge->conditional_costs.end();
    }
    return edge->base_evaluated;
  }
  if ((_vertex_info.at(v1).roadmap_id == _vertex_info.at(v2).roadmap_id) &&
      (_goal_set->isGoal(_vertex_info.at(v2).roadmap_id, _vertex_info.at(v2).layer_id - 1)))
  {
    return _registered_costs.find(v1) != _registered_costs.end();
  }
  return true;
}

template <BackwardsHeuristicType htype>
unsigned int FoldedMultiGraspRoadmapGraph<htype>::getStartVertex() const
{
  return 0;
}

template <BackwardsHeuristicType htype>
bool FoldedMultiGraspRoadmapGraph<htype>::isGoal(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("FoldedMultiGraspRoadmapGraph::IsGoal");
#endif
  return _vertex_info.at(v).roadmap_id == _start_rid and _vertex_info.at(v).layer_id > 0;
}

template <BackwardsHeuristicType htype>
double FoldedMultiGraspRoadmapGraph<htype>::getGoalCost(unsigned int v) const
{
  return 0.0;
}

template <BackwardsHeuristicType htype>
double FoldedMultiGraspRoadmapGraph<htype>::heuristic(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("FoldedMultiGraspRoadmapGraph::heuristic");
#endif
  std::integral_constant<BackwardsHeuristicType, htype> htype_flag;
  if (_vertex_info.at(v).layer_id)
  {
    return graspLayerHeuristicImplementation(v, htype_flag);
  }
  return baseLayerHeuristic(v);
}

template <BackwardsHeuristicType htype>
void FoldedMultiGraspRoadmapGraph<htype>::registerMinimalCost(unsigned int v, double cost)
{
  _registered_costs[v] = cost;
}

template <BackwardsHeuristicType htype>
bool FoldedMultiGraspRoadmapGraph<htype>::isHeuristicValid(unsigned int v) const
{
  return _vertex_info.at(v).layer_id == 0 or
         _registered_costs.find(getVertexId(_vertex_info.at(v).roadmap_id, 0)) != _registered_costs.end();
}

template <BackwardsHeuristicType htype>
unsigned int FoldedMultiGraspRoadmapGraph<htype>::getHeuristicDependentVertex(unsigned int v) const
{
  static_assert(not heuristic_stationary::value);
  return getVertexId(_vertex_info.at(v).roadmap_id, 0);
}

template <BackwardsHeuristicType htype>
std::pair<std::pair<unsigned int, unsigned int>, bool>
FoldedMultiGraspRoadmapGraph<htype>::getGraspRoadmapId(unsigned int vid) const
{
  bool gid_valid = _vertex_info.at(vid).layer_id > 0;
  unsigned int gid = gid_valid ? _vertex_info.at(vid).layer_id - 1 : 0;
  return {{_vertex_info.at(vid).roadmap_id, gid}, gid_valid};
}

template <BackwardsHeuristicType htype>
unsigned int FoldedMultiGraspRoadmapGraph<htype>::getVertexId(unsigned int roadmap_id, unsigned int layer_id) const
{
  auto iter = _vertex_ids.find({roadmap_id, layer_id});
  if (iter == _vertex_ids.end())
  {
    unsigned int vid = _vertex_info.size();
    _vertex_info.emplace_back(roadmap_id, layer_id);
    bool insert_success;
    std::tie(iter, insert_success) = _vertex_ids.insert({{roadmap_id, layer_id}, vid});
    assert(insert_success);
  }
  return iter->second;
}

template <BackwardsHeuristicType htype>
double FoldedMultiGraspRoadmapGraph<htype>::baseLayerHeuristic(unsigned int v) const
{
  assert(_vertex_info.at(v).layer_id == 0);
  auto node = _roadmap->getNode(_vertex_info.at(v).roadmap_id);
  if (!node)
    return std::numeric_limits<double>::infinity();
  return _cost_to_go.costToGo(node->config);
}
/**** Heuristic implementations for grasp-specific layers  ****/

// heuristic that uses also registered distances. WARNING THIS HEURISTIC IS NOT CONSISTENT!
// double graspLayerHeuristic(const std::integral_constant<BackwardsHeuristicType, BestKnownDistance>& type,
//                            unsigned int v, double f)
// {
//   assert(_vertex_info.at(v).layer_id > 0);
//   // get the id of this vertex at the base layer
//   unsigned int v_0 = getVertexId(_vertex_info.at(v).roadmap_id, 0);
//   if (_registered_costs.find(v_0) == _registered_costs.end())
//   {
//     // we don't have a registered cost, so use lower bound
//     return graspLayerHeuristic(std::integral_constant<BackwardsHeuristicType, LowerBound>(), v, f);
//   }
//   return _registered_costs[v_0];
// }

template <BackwardsHeuristicType htype>
double FoldedMultiGraspRoadmapGraph<htype>::graspLayerHeuristicImplementation(
    unsigned int v,
    const std::integral_constant<BackwardsHeuristicType, BackwardsHeuristicType::LowerBound>& htype_flag) const
{
  assert(_vertex_info.at(v).layer_id > 0);
  return getLowerBound(v);
}

template <BackwardsHeuristicType htype>
double FoldedMultiGraspRoadmapGraph<htype>::graspLayerHeuristicImplementation(
    unsigned int v,
    const std::integral_constant<BackwardsHeuristicType, BackwardsHeuristicType::SearchAwareBestKnownDistance>&
        htype_flag) const
{
  assert(_vertex_info.at(v).layer_id > 0);
  // get the id of this vertex at the base layer
  unsigned int v_0 = getVertexId(_vertex_info.at(v).roadmap_id, 0);
  auto iter = _registered_costs.find(v_0);
  if (iter == _registered_costs.end())
  {  // we don't know it yet, so return infinity
    return std::numeric_limits<double>::infinity();
  }
  return iter->second;
}

// simple lower bound heuristic
template <BackwardsHeuristicType htype>
double FoldedMultiGraspRoadmapGraph<htype>::getLowerBound(unsigned int v) const
{
  assert(_vertex_info.at(v).layer_id > 0);
  auto node_v = _roadmap->getNode(_vertex_info.at(v).roadmap_id);
  auto node_s = _roadmap->getNode(_start_rid);
  if (!node_v or !node_s)
    return std::numeric_limits<double>::infinity();
  return _lower_bound(node_s->config, node_v->config);
}
