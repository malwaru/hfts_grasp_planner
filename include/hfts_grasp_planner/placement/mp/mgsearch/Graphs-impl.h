#pragma once

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