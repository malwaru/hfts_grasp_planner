#pragma once
/************************************* MultiGraspRoadmapGraph::NeighborIterator ********************************/
template <bool lazy_grasp_check>
MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator::NeighborIterator(
    unsigned int v, bool lazy, MultiGraspRoadmapGraph<lazy_grasp_check> const* parent)
  : _v(v), _lazy(lazy), _graph(parent), _edge_to_0_returned(false)
{
  if (_v == 0)
  {  // initialization for special case v == 0
    // neighbors are vertices that share the same roadmap node but with different grasp
    _grasp_iter = _graph->_grasp_ids.begin();
  }
  else
  {  // initialization for the standard case v != 0
    std::tie(_grasp_id, _roadmap_id) = _graph->toRoadmapKey(_v);
    auto node = _graph->_roadmap->getNode(_roadmap_id);
    assert(node);
    // neighbors are vertices associated with the adjacent roadmap nodes
    std::tie(_iter, _end) = node->getEdgesIterators();
    // and in case the roadmap node is the start node, we also have an adjacency to node 0
    _edge_to_0_returned = _roadmap_id != _graph->_roadmap_start_id;
    // if we do not have the special edge, we need to ensure _iter points to a valid neighbor
    if (_edge_to_0_returned)
    {
      forwardToNextValid();
    }
  }
}

template <bool lazy_grasp_check>
typename MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator&
MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator::operator++()
{
  if (_v == 0)
  {  // increase iterator for special case v == 0
    ++_grasp_iter;
  }
  else
  {  // increase iterator for default case
    // treat the special edge back to vertex 0 for start vertices
    if (not _edge_to_0_returned)
    {
      _edge_to_0_returned = true;
    }
    else
    {  // increase edge iterator
      ++_iter;
    }
    forwardToNextValid();
  }
  return (*this);
}

template <bool lazy_grasp_check>
bool MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator::operator==(
    const typename MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator& other) const
{
  // TODO take lazy into account?
  if (other._v != _v)
  {
    return false;
  }
  if (_v == 0)
  {
    return other._grasp_iter == _grasp_iter;
  }
  return other._iter == _iter and other._edge_to_0_returned == _edge_to_0_returned;
}

template <bool lazy_grasp_check>
bool MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator::operator!=(
    const MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator& other) const
{
  return not operator==(other);
}

template <bool lazy_grasp_check>
unsigned int MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator::operator*()
{
  if (_v == 0)
  {
    unsigned int gid = *_grasp_iter;
    return _graph->toGraphKey(gid, _graph->_roadmap_start_id);
  }
  // else
  if (not _edge_to_0_returned)
  {
    return 0;
  }
  // else
  unsigned int rid = _iter->first;
  return _graph->toGraphKey(_grasp_id, rid);
}

template <bool lazy_grasp_check>
void MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator::forwardToNextValid()
{
  assert(_edge_to_0_returned);
  assert(_v != 0);
  bool valid = false;
  while (!valid and _iter != _end)
  {
    if (_lazy)
    {
      double cost = _iter->second->getBestKnownCost(_grasp_id);
      valid = not std::isinf(cost);
    }
    else
    {
      if constexpr (lazy_grasp_check)
      {
        if (_graph->_roadmap->isValid(_iter->first))
        {
          _graph->_roadmap->computeCost(_iter->second);
          valid = not std::isinf(_iter->second->getBestKnownCost(_grasp_id));  // in case we know a btter cost estimate
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
    if (not valid)
    {
      ++_iter;
    }
  }
}

template <bool lazy_grasp_check>
typename MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator
MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator::begin(unsigned int v, bool lazy,
                                                                  MultiGraspRoadmapGraph<lazy_grasp_check> const* graph)
{
  if (v != 0)
  {
    // update roadmap
    auto [gid, rid] = graph->toRoadmapKey(v);
    auto node = graph->_roadmap->getNode(rid);
    graph->_roadmap->updateAdjacency(node);
  }
  return NeighborIterator(v, lazy, graph);
}

template <bool lazy_grasp_check>
typename MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator
MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator::end(unsigned int v,
                                                                MultiGraspRoadmapGraph<lazy_grasp_check> const* graph)
{
  NeighborIterator end_iter(v, true, graph);
  end_iter._grasp_iter = graph->_grasp_ids.end();  // covers end condition for v = 0
  end_iter._iter = end_iter._end;                  // covers end condition for v != 0
  end_iter._edge_to_0_returned = true;             // covers end condition for v != 0
  return end_iter;
}

/************************************* MultiGraspRoadmapGraph ********************************/
template <bool lazy_grasp_check>
MultiGraspRoadmapGraph<lazy_grasp_check>::MultiGraspRoadmapGraph(
    RoadmapPtr roadmap, MultiGraspGoalSetPtr goal_set,
    const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params, const std::set<unsigned int>& grasp_ids,
    unsigned int start_id)
  : _roadmap(roadmap)
  , _goal_set(goal_set)
  , _all_grasps_cost_to_go(goal_set, cost_params)
  , _grasp_ids(grasp_ids)
  , _roadmap_start_id(start_id)
  , _num_graph_nodes(0)
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

template <bool lazy_grasp_check>
MultiGraspRoadmapGraph<lazy_grasp_check>::~MultiGraspRoadmapGraph() = default;

template <bool lazy_grasp_check>
bool MultiGraspRoadmapGraph<lazy_grasp_check>::checkValidity(unsigned int v)
{
  if (v == 0)
  {
    // special case for start node which is not associated with any grasp
    return _roadmap->isValid(_roadmap->getNode(_roadmap_start_id));
  }
  auto [grasp_id, roadmap_id] = toRoadmapKey(v);
  auto node = _roadmap->getNode(roadmap_id);
  if (!node)
    return false;
  return _roadmap->isValid(node, grasp_id);
}

template <bool lazy_grasp_check>
void MultiGraspRoadmapGraph<lazy_grasp_check>::getSuccessors(unsigned int v, std::vector<unsigned int>& successors,
                                                             bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v == 0)
  {
    _logger.logExpansion(_roadmap_start_id);
  }
  else
  {
    auto [rid, gid] = getGraspRoadmapId(v);
    _logger.logExpansion(rid, gid);
  }
#endif
  successors.clear();
  auto begin = NeighborIterator::begin(v, lazy, this);
  auto end = NeighborIterator::end(v, this);
  successors.insert(successors.begin(), begin, end);
}

template <bool lazy_grasp_check>
std::pair<typename MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator,
          typename MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator>
MultiGraspRoadmapGraph<lazy_grasp_check>::getSuccessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v == 0)
  {
    _logger.logExpansion(_roadmap_start_id);
  }
  else
  {
    auto [rid, gid] = getGraspRoadmapId(v);
    _logger.logExpansion(rid, gid);
  }
#endif
  return {NeighborIterator::begin(v, lazy, this), NeighborIterator::end(v, this)};
}

template <bool lazy_grasp_check>
void MultiGraspRoadmapGraph<lazy_grasp_check>::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors,
                                                               bool lazy)
{
  // undirected graph
  getSuccessors(v, predecessors, lazy);
}

template <bool lazy_grasp_check>
std::pair<typename MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator,
          typename MultiGraspRoadmapGraph<lazy_grasp_check>::NeighborIterator>
MultiGraspRoadmapGraph<lazy_grasp_check>::getPredecessors(unsigned int v, bool lazy)
{
  return getSuccessors(v, lazy);
}

template <bool lazy_grasp_check>
double MultiGraspRoadmapGraph<lazy_grasp_check>::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy)
{
  // catch special case of start node
  if (v1 == 0 || v2 == 0)
  {
    return 0.0;  // TODO could return here costs for obtaining a grasp
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
  if constexpr (lazy_grasp_check)
  {
    _roadmap->computeCost(edge);
    return edge->getBestKnownCost(grasp_1);  // we might actually know a better cost estimate then the base
  }
  else
  {
    return _roadmap->computeCost(edge, grasp_1).second;
  }
}

template <bool lazy_grasp_check>
bool MultiGraspRoadmapGraph<lazy_grasp_check>::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
{
  // catch special case of start node
  if (v1 == 0 || v2 == 0)
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
  if constexpr (lazy_grasp_check)
  {
    return edge->base_evaluated;
  }
  else
  {
    return edge->conditional_costs.find(grasp_1) != edge->conditional_costs.end();
  }
}

template <bool lazy_grasp_check>
unsigned int MultiGraspRoadmapGraph<lazy_grasp_check>::getStartNode() const
{
  return 0;
}

template <bool lazy_grasp_check>
bool MultiGraspRoadmapGraph<lazy_grasp_check>::isGoal(unsigned int v) const
{
  if (v == 0)
    return false;
  auto [grasp_id, rnid] = toRoadmapKey(v);
  return _goal_set->isGoal(rnid, grasp_id);
}

template <bool lazy_grasp_check>
double MultiGraspRoadmapGraph<lazy_grasp_check>::getGoalCost(unsigned int v) const
{
  if (v == 0)
  {  // can not be a goal
    return std::numeric_limits<double>::infinity();
  }
  auto [grasp_id, rnid] = toRoadmapKey(v);
  auto node = _roadmap->getNode(rnid);
  assert(node);
  auto [goal_id, is_goal] = _goal_set->getGoalId(node->uid, grasp_id);
  if (not is_goal)
  {
    return std::numeric_limits<double>::infinity();
  }
  return _all_grasps_cost_to_go.qualityToGoalCost(_goal_set->getGoal(goal_id).quality);
}

template <bool lazy_grasp_check>
double MultiGraspRoadmapGraph<lazy_grasp_check>::heuristic(unsigned int v) const
{
  if (v == 0)
  {
    auto node = _roadmap->getNode(_roadmap_start_id);
    return _all_grasps_cost_to_go.costToGo(node->config);
  }
  // else
  auto [grasp_id, rnid] = toRoadmapKey(v);
  auto node = _roadmap->getNode(rnid);
  if (!node)
    return INFINITY;
  return _individual_cost_to_go.at(grasp_id)->costToGo(node->config);
}

template <bool lazy_grasp_check>
void MultiGraspRoadmapGraph<lazy_grasp_check>::registerMinimalCost(unsigned int v, double cost)
{
  // no-op
}

template <bool lazy_grasp_check>
double MultiGraspRoadmapGraph<lazy_grasp_check>::getEdgeCostWithGrasp(unsigned int v1, unsigned int v2)
{
  if constexpr (lazy_grasp_check)
  {
    // catch special case of start node
    if (v1 == 0 || v2 == 0)
    {
      return 0.0;  // TODO could return here costs for obtaining a grasp
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
    if (!checkValidity(v1))
      return INFINITY;
    if (!checkValidity(v2))
      return INFINITY;
    return _roadmap->computeCost(edge, grasp_1).second;
  }
  else
  {
    return getEdgeCost(v1, v2, false);
  }
}

template <bool lazy_grasp_check>
bool MultiGraspRoadmapGraph<lazy_grasp_check>::trueEdgeCostWithGraspKnown(unsigned int v1, unsigned int v2) const
{
  if constexpr (lazy_grasp_check)
  {
    // catch special case of start node
    if (v1 == 0 || v2 == 0)
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
    return edge->conditional_costs.find(grasp_1) != edge->conditional_costs.end();
  }
  else
  {
    return trueEdgeCostKnown(v1, v2);
  }
}

template <bool lazy_grasp_check>
std::pair<unsigned int, unsigned int>
MultiGraspRoadmapGraph<lazy_grasp_check>::getGraspRoadmapId(unsigned int vid) const
{
  if (vid == 0)
  {
    return {_roadmap_start_id, 0};  // TODO what makes sense to return here for the grasp?
  }
  auto [grasp_id, rid] = toRoadmapKey(vid);
  return {rid, grasp_id};
}

template <bool lazy_grasp_check>
std::pair<unsigned int, unsigned int>
MultiGraspRoadmapGraph<lazy_grasp_check>::toRoadmapKey(unsigned int graph_id) const
{
  auto iter = _graph_key_to_roadmap.find(graph_id);
  assert(iter != _graph_key_to_roadmap.end());
  return iter->second;
}

template <bool lazy_grasp_check>
unsigned int
MultiGraspRoadmapGraph<lazy_grasp_check>::toGraphKey(const std::pair<unsigned int, unsigned int>& roadmap_id) const
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

template <bool lazy_grasp_check>
unsigned int MultiGraspRoadmapGraph<lazy_grasp_check>::toGraphKey(unsigned int grasp_id, unsigned int node_id) const
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
  auto begin = FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::begin(v, false, lazy, this);
  auto end = FoldedMultiGraspRoadmapGraph<htype>::NeighborIterator::end(v, false, lazy, this);
  return {std::move(begin), std::move(end)};
}

template <BackwardsHeuristicType htype>
double FoldedMultiGraspRoadmapGraph<htype>::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy)
{
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
unsigned int FoldedMultiGraspRoadmapGraph<htype>::getStartNode() const
{
  return 0;
}

template <BackwardsHeuristicType htype>
bool FoldedMultiGraspRoadmapGraph<htype>::isGoal(unsigned int v) const
{
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
