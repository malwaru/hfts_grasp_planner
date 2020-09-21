#include <hfts_grasp_planner/placement/mp/mgsearch/LazyGraph.h>

/*********************** LazyMultiGraspRoadmapGrasph::NeightborIterator ************************/
LazyMultiGraspRoadmapGraph::NeighborIterator::NeighborIterator(
    const LazyMultiGraspRoadmapGraph::NeighborIterator& other)
{
  _impl = other._impl->copy();
}

LazyMultiGraspRoadmapGraph::NeighborIterator::NeighborIterator(unsigned int v, bool forward, bool lazy,
                                                               LazyMultiGraspRoadmapGraph const* parent)
{
  unsigned int grasp_group_id = parent->_vertex_information.at(v).grasp_group;
  // check whether v is a single grasp vertex
  if (parent->_grasp_groups.at(grasp_group_id).grasp_set.size() == 1)
  {
    if (forward)
    {
      if (lazy)
        _impl = std::make_unique<ForwardSingleGraspIterator<true>>(v, parent);
      else
        _impl = std::make_unique<ForwardSingleGraspIterator<false>>(v, parent);
    }
    else
    {
      if (lazy)
        _impl = std::make_unique<BackwardSingleGraspIterator<true>>(v, parent);
      else
        _impl = std::make_unique<BackwardSingleGraspIterator<false>>(v, parent);
    }
  }
  else
  {
    if (forward)
    {
      if (lazy)
        _impl = std::make_unique<ForwardMultiGraspIterator<true>>(v, parent);
      else
        _impl = std::make_unique<ForwardMultiGraspIterator<false>>(v, parent);
    }
    else
    {
      if (lazy)
        _impl = std::make_unique<BackwardMultiGraspIterator<true>>(v, parent);
      else
        _impl = std::make_unique<BackwardMultiGraspIterator<false>>(v, parent);
    }
  }
}

LazyMultiGraspRoadmapGraph::NeighborIterator::~NeighborIterator() = default;

LazyMultiGraspRoadmapGraph::NeighborIterator& LazyMultiGraspRoadmapGraph::NeighborIterator::operator++()
{
  _impl->next();
  return (*this);
}

bool LazyMultiGraspRoadmapGraph::NeighborIterator::operator==(
    const LazyMultiGraspRoadmapGraph::NeighborIterator& other) const
{
  return _impl->equals(other._impl);
}

bool LazyMultiGraspRoadmapGraph::NeighborIterator::operator!=(
    const LazyMultiGraspRoadmapGraph::NeighborIterator& other) const
{
  return not operator==(other);
}

unsigned int LazyMultiGraspRoadmapGraph::NeighborIterator::operator*()
{
  return _impl->dereference();
}

LazyMultiGraspRoadmapGraph::NeighborIterator LazyMultiGraspRoadmapGraph::NeighborIterator::begin(
    unsigned int v, bool forward, bool lazy, LazyMultiGraspRoadmapGraph const* parent)
{
  auto node = parent->_roadmap->getNode(parent->_vertex_information.at(v).roadmap_node_id);
  if (!node)
  {
    throw std::logic_error("Invalid vertex node");
  }
  parent->_roadmap->updateAdjacency(node);
  return NeighborIterator(v, forward, lazy, parent);
}

LazyMultiGraspRoadmapGraph::NeighborIterator LazyMultiGraspRoadmapGraph::NeighborIterator::end(
    unsigned int v, bool forward, bool lazy, LazyMultiGraspRoadmapGraph const* parent)
{
  auto iter = NeighborIterator(v, forward, lazy, parent);
  iter._impl->setToEnd();
  return iter;
}

/***************************** LazyMultiGraspRoadmapGraph::LazyMultiGraspRoadmapGraph ****************************/
LazyMultiGraspRoadmapGraph::LazyMultiGraspRoadmapGraph(
    ::placement::mp::mgsearch::RoadmapPtr roadmap, ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
    const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params, unsigned int start_id)
  : _roadmap(roadmap), _cost_params(cost_params), _quality_range(goal_set->getGoalQualityRange())
{
  GraspGroup base_group;
  base_group.goal_set = goal_set;
  base_group.grasp_set = goal_set->getGraspSet();
  base_group.cost_to_go = std::make_shared<MultiGoalCostToGo>(goal_set, _cost_params, _quality_range);
  base_group.roadmap_id_to_vertex[start_id] = 0;
  _grasp_groups.push_back(base_group);
  _grasp_group_ids[getStringRepresentation(base_group.grasp_set)] = 0;
  VertexInformation vi;
  vi.grasp_group = 0;
  vi.roadmap_node_id = start_id;
  _vertex_information.push_back(vi);
}

LazyMultiGraspRoadmapGraph::~LazyMultiGraspRoadmapGraph() = default;

bool LazyMultiGraspRoadmapGraph::checkValidity(unsigned int v) const
{
  assert(v < _vertex_information.size());
  unsigned int rid = _vertex_information.at(v).roadmap_node_id;
  unsigned int group_id = _vertex_information.at(v).grasp_group;
  auto node = _roadmap->getNode(rid);
  if (!node)
    return false;
  if (_grasp_groups.at(group_id).grasp_set.size() > 1)
  {
    // only check for the base
    return _roadmap->isValid(node);
  }
  else
  {
    assert(_grasp_groups.size() == 1);
    // check validity for the grasp that we have
    unsigned int grasp_id = *_grasp_groups.at(group_id).grasp_set.begin();
    return _roadmap->isValid(node, grasp_id);
  }
}

void LazyMultiGraspRoadmapGraph::getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy) const
{
  successors.clear();
  auto [begin, end] = getSuccessors(v, lazy);
  successors.insert(successors.begin(), begin, end);
}

std::pair<LazyMultiGraspRoadmapGraph::NeighborIterator, LazyMultiGraspRoadmapGraph::NeighborIterator>
LazyMultiGraspRoadmapGraph::getSuccessors(unsigned int v, bool lazy) const
{
  return {NeighborIterator::begin(v, true, lazy, this), NeighborIterator::end(v, true, lazy, this)};
}

void LazyMultiGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors,
                                                 bool lazy) const
{
  predecessors.clear();
  auto [begin, end] = getPredecessors(v, lazy);
  predecessors.insert(predecessors.begin(), begin, end);
}

std::pair<LazyMultiGraspRoadmapGraph::NeighborIterator, LazyMultiGraspRoadmapGraph::NeighborIterator>
LazyMultiGraspRoadmapGraph::getPredecessors(unsigned int v, bool lazy) const
{
  return {NeighborIterator::begin(v, false, lazy, this), NeighborIterator::end(v, false, lazy, this)};
}

double LazyMultiGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy) const
{
  const VertexInformation& vi1 = _vertex_information.at(v1);
  const VertexInformation& vi2 = _vertex_information.at(v2);
  const GraspGroup& group1 = _grasp_groups.at(vi1.grasp_group);
  const GraspGroup& group2 = _grasp_groups.at(vi2.grasp_group);
  auto node1 = _roadmap->getNode(vi1.roadmap_node_id);
  // capture queries where we have no adjacency
  if (!node1)
    return std::numeric_limits<double>::infinity();
  _roadmap->updateAdjacency(node1);
  auto edge = node1->getEdge(vi2.roadmap_node_id);
  if (!edge)
    return std::numeric_limits<double>::infinity();
  if (vi1.grasp_group != vi2.grasp_group && vi1.non_group_neighbors.find(v2) == vi1.non_group_neighbors.end())
  {
    return std::numeric_limits<double>::infinity();
  }
  if (group1.grasp_set.size() < group2.grasp_set.size())
  {
    // we have a directed graph so even if there is an edge between v1 and v2 it only has finite edge cost if
    // v1's grasp set is a super set of v2's
    return std::numeric_limits<double>::infinity();
  }
  // check whether we have a single grasp or a multi-grasp edge
  unsigned int gid;
  if (group2.grasp_set.size() == 1)
  {
    gid = *group2.grasp_set.begin();
    if (lazy)
    {
      return edge->getBestKnownCost(gid);
    }
    return _roadmap->computeCost(edge, gid).second;
  }
  else
  {
    // multiple grasps, so compute base cost
    if (lazy)
    {
      return edge->base_cost;
    }
    return _roadmap->computeCost(edge).second;
  }
}

bool LazyMultiGraspRoadmapGraph::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
{
  const VertexInformation& vi1 = _vertex_information.at(v1);
  const VertexInformation& vi2 = _vertex_information.at(v2);
  const GraspGroup& group1 = _grasp_groups.at(vi1.grasp_group);
  const GraspGroup& group2 = _grasp_groups.at(vi2.grasp_group);
  auto node1 = _roadmap->getNode(vi1.roadmap_node_id);
  // capture queries where we have no adjacency
  if (!node1)
    return true;
  _roadmap->updateAdjacency(node1);
  auto edge = node1->getEdge(vi2.roadmap_node_id);
  if (!edge)
    return true;
  // we know the true cost if vertices are not adjacent by definition
  if ((vi1.grasp_group != vi2.grasp_group && vi1.non_group_neighbors.find(v2) == vi1.non_group_neighbors.end()) ||
      group1.grasp_set.size() < group2.grasp_set.size())
  {
    return true;
  }
  // check whether we have a single grasp or a multi-grasp edge
  if (group2.grasp_set.size() == 1)
  {
    unsigned int gid = *group2.grasp_set.begin();
    return edge->conditional_costs.find(gid) != edge->conditional_costs.end();
  }
  // multiple grasps -> return whether the base has been evaluated
  return edge->base_evaluated;
}

unsigned int LazyMultiGraspRoadmapGraph::getStartNode() const
{
  return 0;
}

bool LazyMultiGraspRoadmapGraph::isGoal(unsigned int v) const
{
  unsigned int rid = _vertex_information.at(v).roadmap_node_id;
  unsigned int group_id = _vertex_information.at(v).grasp_group;
  return _grasp_groups.at(group_id).goal_set->canBeGoal(rid);  // as far as we know, it's a goal!
}

double LazyMultiGraspRoadmapGraph::getGoalCost(unsigned int v) const
{
  unsigned int rid = _vertex_information.at(v).roadmap_node_id;
  unsigned int group_id = _vertex_information.at(v).grasp_group;
  auto node = _roadmap->getNode(rid);
  assert(node);
  // return the minimal cost of any goal associated with this roadmap node (given the grasp group)
  std::vector<unsigned int> goal_ids(_grasp_groups.at(group_id).goal_set->getGoalIds(rid));
  double goal_cost = std::numeric_limits<double>::infinity();
  for (auto goal_id : goal_ids)
  {
    goal_cost = std::min(goal_cost, _grasp_groups.at(group_id).cost_to_go->qualityToGoalCost(
                                        _grasp_groups.at(group_id).goal_set->getGoal(goal_id).quality));
  }
  return goal_cost;  // infinity if not a goal
}

double LazyMultiGraspRoadmapGraph::heuristic(unsigned int v) const
{
  unsigned int rid = _vertex_information.at(v).roadmap_node_id;
  unsigned int group_id = _vertex_information.at(v).grasp_group;
  auto node = _roadmap->getNode(rid);
  if (!node)
    return INFINITY;
  return _grasp_groups.at(group_id).cost_to_go->costToGo(node->config);
}

std::pair<unsigned int, unsigned int> LazyMultiGraspRoadmapGraph::checkEdgeSplit(unsigned int v1, unsigned int v2,
                                                                                 unsigned int gid)
{
  auto& v1_info = _vertex_information.at(v1);
  auto& v2_info = _vertex_information.at(v2);
  if (v1_info.grasp_group != v2_info.grasp_group)
  {
    // we already know that v1 and v2 belong to different grasp groups
    return {v2, v2};
  }
  // get node and check edge
  auto node = _roadmap->getNode(v1_info.roadmap_node_id);
  assert(node);
  auto edge = node->getEdge(v2_info.roadmap_node_id);
  assert(edge);
  auto edge_iter = edge->conditional_costs.find(gid);
  assert(edge_iter == edge->conditional_costs.end());
  // compute edge cost for this grasp
  auto [valid, cost] = _roadmap->computeCost(edge, gid);
  if (cost != edge->base_cost)
  {
    // the edge cost is indeed different for the grasp, so perform a split
    // add a vertex in the grasp group covering only gid
    std::set single_grasp_subset({gid});
    unsigned int single_id = getGraspGroupId(single_grasp_subset);
    unsigned int v2_single_grasp = getVertexId(v2_info.roadmap_node_id, single_id);
    // add a vertex in the grasp group covering the remaining grasps
    auto& parent_group = _grasp_groups.at(v1_info.grasp_group);
    std::set<unsigned int> multi_grasp_subset;
    std::set_difference(parent_group.grasp_set.begin(), parent_group.grasp_set.end(), single_grasp_subset.begin(),
                        single_grasp_subset.end(), std::inserter(multi_grasp_subset, multi_grasp_subset.begin()));
    unsigned int multiple_id = getGraspGroupId(multi_grasp_subset);
    unsigned int v2_multi_grasp = getVertexId(v2_info.roadmap_node_id, multiple_id);
    // add adjacency information
    v1_info.non_group_neighbors.insert(v2_single_grasp);
    v1_info.non_group_neighbors.insert(v2_multi_grasp);
    _vertex_information.at(v2_single_grasp).non_group_neighbors.insert(v1);
    _vertex_information.at(v2_multi_grasp).non_group_neighbors.insert(v1);
    return {v2_single_grasp, v2_multi_grasp};
  }
  return {v2, v2};
}

unsigned int LazyMultiGraspRoadmapGraph::getVertexId(unsigned int rid, unsigned int grasp_group_id) const
{
  // check whether we already know this vertex
  auto iter = _grasp_groups.at(grasp_group_id).roadmap_id_to_vertex.find(rid);
  if (iter != _grasp_groups.at(grasp_group_id).roadmap_id_to_vertex.end())
  {
    return iter->second;
  }
  // else add it
  VertexInformation new_vertex_info;
  new_vertex_info.grasp_group = grasp_group_id;
  new_vertex_info.roadmap_node_id = rid;
  unsigned int vid = _vertex_information.size();
  _vertex_information.push_back(new_vertex_info);
  _grasp_groups.at(grasp_group_id).roadmap_id_to_vertex[rid] = vid;
  return vid;
}

unsigned int LazyMultiGraspRoadmapGraph::getGraspGroupId(const std::set<unsigned int>& grasp_set) const
{
  std::string key = getStringRepresentation(grasp_set);
  auto iter = _grasp_group_ids.find(key);
  if (iter == _grasp_group_ids.end())
  {
    // add new group
    GraspGroup new_group;
    new_group.grasp_set = grasp_set;
    // group 0 contains all goals
    new_group.goal_set = _grasp_groups.at(0).goal_set->createSubset(new_group.grasp_set);
    new_group.cost_to_go = std::make_shared<MultiGoalCostToGo>(new_group.goal_set, _cost_params, _quality_range);
    unsigned int new_group_id = _grasp_groups.size();
    _grasp_groups.push_back(new_group);
    _grasp_group_ids[key] = new_group_id;
  }
  return iter->second;
}

std::string LazyMultiGraspRoadmapGraph::getStringRepresentation(const std::set<unsigned int>& grasp_set) const
{
  std::stringstream ss;
  for (auto grasp_id : grasp_set)
  {
    ss << grasp_id;
  }
  return ss.str();
}

std::string LazyMultiGraspRoadmapGraph::getStringRepresentation(unsigned int gid) const
{
  /*********************** LazyMultiGraspRoadmapGrasph::NeightborIterator ************************/
  LazyMultiGraspRoadmapGraph::NeighborIterator::NeighborIterator(
      const LazyMultiGraspRoadmapGraph::NeighborIterator& other)
  {
    _impl = other._impl->copy();
  }

  LazyMultiGraspRoadmapGraph::NeighborIterator::NeighborIterator(unsigned int v, bool forward, bool lazy,
                                                                 LazyMultiGraspRoadmapGraph const* parent)
  {
    unsigned int grasp_group_id = parent->_vertex_information.at(v).grasp_group;
    // check whether v is a single grasp vertex
    if (parent->_grasp_groups.at(grasp_group_id).grasp_set.size() == 1)
    {
      if (forward)
      {
        if (lazy)
          _impl = std::make_unique<ForwardSingleGraspIterator<true>>(v, parent);
        else
          _impl = std::make_unique<ForwardSingleGraspIterator<false>>(v, parent);
      }
      else
      {
        if (lazy)
          _impl = std::make_unique<BackwardSingleGraspIterator<true>>(v, parent);
        else
          _impl = std::make_unique<BackwardSingleGraspIterator<false>>(v, parent);
      }
    }
    else
    {
      if (forward)
      {
        if (lazy)
          _impl = std::make_unique<ForwardMultiGraspIterator<true>>(v, parent);
        else
          _impl = std::make_unique<ForwardMultiGraspIterator<false>>(v, parent);
      }
      else
      {
        if (lazy)
          _impl = std::make_unique<BackwardMultiGraspIterator<true>>(v, parent);
        else
          _impl = std::make_unique<BackwardMultiGraspIterator<false>>(v, parent);
      }
    }
  }

  LazyMultiGraspRoadmapGraph::NeighborIterator::~NeighborIterator() = default;

  LazyMultiGraspRoadmapGraph::NeighborIterator& LazyMultiGraspRoadmapGraph::NeighborIterator::operator++()
  {
    _impl->next();
    return (*this);
  }

  bool LazyMultiGraspRoadmapGraph::NeighborIterator::operator==(
      const LazyMultiGraspRoadmapGraph::NeighborIterator& other) const
  {
    return _impl->equals(other._impl);
  }

  bool LazyMultiGraspRoadmapGraph::NeighborIterator::operator!=(
      const LazyMultiGraspRoadmapGraph::NeighborIterator& other) const
  {
    return not operator==(other);
  }

  unsigned int LazyMultiGraspRoadmapGraph::NeighborIterator::operator*()
  {
    return _impl->dereference();
  }

  LazyMultiGraspRoadmapGraph::NeighborIterator LazyMultiGraspRoadmapGraph::NeighborIterator::begin(
      unsigned int v, bool forward, bool lazy, LazyMultiGraspRoadmapGraph const* parent)
  {
    auto node = parent->_roadmap->getNode(parent->_vertex_information.at(v).roadmap_node_id);
    if (!node)
    {
      throw std::logic_error("Invalid vertex node");
    }
    parent->_roadmap->updateAdjacency(node);
    return NeighborIterator(v, forward, lazy, parent);
  }

  LazyMultiGraspRoadmapGraph::NeighborIterator LazyMultiGraspRoadmapGraph::NeighborIterator::end(
      unsigned int v, bool forward, bool lazy, LazyMultiGraspRoadmapGraph const* parent)
  {
    auto iter = NeighborIterator(v, forward, lazy, parent);
    iter._impl->setToEnd();
    return iter;
  }

  /***************************** LazyMultiGraspRoadmapGraph::LazyMultiGraspRoadmapGraph ****************************/
  LazyMultiGraspRoadmapGraph::LazyMultiGraspRoadmapGraph(
      ::placement::mp::mgsearch::RoadmapPtr roadmap, ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
      const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params, unsigned int start_id)
    : _roadmap(roadmap), _cost_params(cost_params), _quality_range(goal_set->getGoalQualityRange())
  {
    GraspGroup base_group;
    base_group.goal_set = goal_set;
    base_group.grasp_set = goal_set->getGraspSet();
    base_group.cost_to_go = std::make_shared<MultiGoalCostToGo>(goal_set, _cost_params, _quality_range);
    base_group.roadmap_id_to_vertex[start_id] = 0;
    _grasp_groups.push_back(base_group);
    _grasp_group_ids[getStringRepresentation(base_group.grasp_set)] = 0;
    VertexInformation vi;
    vi.grasp_group = 0;
    vi.roadmap_node_id = start_id;
    _vertex_information.push_back(vi);
  }

  LazyMultiGraspRoadmapGraph::~LazyMultiGraspRoadmapGraph() = default;

  bool LazyMultiGraspRoadmapGraph::checkValidity(unsigned int v) const
  {
    assert(v < _vertex_information.size());
    unsigned int rid = _vertex_information.at(v).roadmap_node_id;
    unsigned int group_id = _vertex_information.at(v).grasp_group;
    auto node = _roadmap->getNode(rid);
    if (!node)
      return false;
    if (_grasp_groups.at(group_id).grasp_set.size() > 1)
    {
      // only check for the base
      return _roadmap->isValid(node);
    }
    else
    {
      assert(_grasp_groups.size() == 1);
      // check validity for the grasp that we have
      unsigned int grasp_id = *_grasp_groups.at(group_id).grasp_set.begin();
      return _roadmap->isValid(node, grasp_id);
    }
  }

  void LazyMultiGraspRoadmapGraph::getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy) const
  {
    successors.clear();
    auto [begin, end] = getSuccessors(v, lazy);
    successors.insert(successors.begin(), begin, end);
  }

  std::pair<LazyMultiGraspRoadmapGraph::NeighborIterator, LazyMultiGraspRoadmapGraph::NeighborIterator>
  LazyMultiGraspRoadmapGraph::getSuccessors(unsigned int v, bool lazy) const
  {
    return {NeighborIterator::begin(v, true, lazy, this), NeighborIterator::end(v, true, lazy, this)};
  }

  void LazyMultiGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy)
      const
  {
    predecessors.clear();
    auto [begin, end] = getPredecessors(v, lazy);
    predecessors.insert(predecessors.begin(), begin, end);
  }

  std::pair<LazyMultiGraspRoadmapGraph::NeighborIterator, LazyMultiGraspRoadmapGraph::NeighborIterator>
  LazyMultiGraspRoadmapGraph::getPredecessors(unsigned int v, bool lazy) const
  {
    return {NeighborIterator::begin(v, false, lazy, this), NeighborIterator::end(v, false, lazy, this)};
  }

  double LazyMultiGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy) const
  {
    const VertexInformation& vi1 = _vertex_information.at(v1);
    const VertexInformation& vi2 = _vertex_information.at(v2);
    const GraspGroup& group1 = _grasp_groups.at(vi1.grasp_group);
    const GraspGroup& group2 = _grasp_groups.at(vi2.grasp_group);
    auto node1 = _roadmap->getNode(vi1.roadmap_node_id);
    // capture queries where we have no adjacency
    if (!node1)
      return std::numeric_limits<double>::infinity();
    _roadmap->updateAdjacency(node1);
    auto edge = node1->getEdge(vi2.roadmap_node_id);
    if (!edge)
      return std::numeric_limits<double>::infinity();
    if (vi1.grasp_group != vi2.grasp_group && vi1.non_group_neighbors.find(v2) == vi1.non_group_neighbors.end())
    {
      return std::numeric_limits<double>::infinity();
    }
    if (group1.grasp_set.size() < group2.grasp_set.size())
    {
      // we have a directed graph so even if there is an edge between v1 and v2 it only has finite edge cost if
      // v1's grasp set is a super set of v2's
      return std::numeric_limits<double>::infinity();
    }
    // check whether we have a single grasp or a multi-grasp edge
    unsigned int gid;
    if (group2.grasp_set.size() == 1)
    {
      gid = *group2.grasp_set.begin();
      if (lazy)
      {
        return edge->getBestKnownCost(gid);
      }
      return _roadmap->computeCost(edge, gid).second;
    }
    else
    {
      // multiple grasps, so compute base cost
      if (lazy)
      {
        return edge->base_cost;
      }
      return _roadmap->computeCost(edge).second;
    }
  }

  bool LazyMultiGraspRoadmapGraph::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
  {
    const VertexInformation& vi1 = _vertex_information.at(v1);
    const VertexInformation& vi2 = _vertex_information.at(v2);
    const GraspGroup& group1 = _grasp_groups.at(vi1.grasp_group);
    const GraspGroup& group2 = _grasp_groups.at(vi2.grasp_group);
    auto node1 = _roadmap->getNode(vi1.roadmap_node_id);
    // capture queries where we have no adjacency
    if (!node1)
      return true;
    _roadmap->updateAdjacency(node1);
    auto edge = node1->getEdge(vi2.roadmap_node_id);
    if (!edge)
      return true;
    // we know the true cost if vertices are not adjacent by definition
    if ((vi1.grasp_group != vi2.grasp_group && vi1.non_group_neighbors.find(v2) == vi1.non_group_neighbors.end()) ||
        group1.grasp_set.size() < group2.grasp_set.size())
    {
      return true;
    }
    // check whether we have a single grasp or a multi-grasp edge
    if (group2.grasp_set.size() == 1)
    {
      unsigned int gid = *group2.grasp_set.begin();
      return edge->conditional_costs.find(gid) != edge->conditional_costs.end();
    }
    // multiple grasps -> return whether the base has been evaluated
    return edge->base_evaluated;
  }

  unsigned int LazyMultiGraspRoadmapGraph::getStartNode() const
  {
    return 0;
  }

  bool LazyMultiGraspRoadmapGraph::isGoal(unsigned int v) const
  {
    unsigned int rid = _vertex_information.at(v).roadmap_node_id;
    unsigned int group_id = _vertex_information.at(v).grasp_group;
    return _grasp_groups.at(group_id).goal_set->canBeGoal(rid);  // as far as we know, it's a goal!
  }

  double LazyMultiGraspRoadmapGraph::getGoalCost(unsigned int v) const
  {
    unsigned int rid = _vertex_information.at(v).roadmap_node_id;
    unsigned int group_id = _vertex_information.at(v).grasp_group;
    auto node = _roadmap->getNode(rid);
    assert(node);
    // return the minimal cost of any goal associated with this roadmap node (given the grasp group)
    std::vector<unsigned int> goal_ids(_grasp_groups.at(group_id).goal_set->getGoalIds(rid));
    double goal_cost = std::numeric_limits<double>::infinity();
    for (auto goal_id : goal_ids)
    {
      goal_cost = std::min(goal_cost, _grasp_groups.at(group_id).cost_to_go->qualityToGoalCost(
                                          _grasp_groups.at(group_id).goal_set->getGoal(goal_id).quality));
    }
    return goal_cost;  // infinity if not a goal
  }

  double LazyMultiGraspRoadmapGraph::heuristic(unsigned int v) const
  {
    unsigned int rid = _vertex_information.at(v).roadmap_node_id;
    unsigned int group_id = _vertex_information.at(v).grasp_group;
    auto node = _roadmap->getNode(rid);
    if (!node)
      return INFINITY;
    return _grasp_groups.at(group_id).cost_to_go->costToGo(node->config);
  }

  std::pair<unsigned int, unsigned int> LazyMultiGraspRoadmapGraph::checkEdgeSplit(unsigned int v1, unsigned int v2,
                                                                                   unsigned int gid)
  {
    auto& v1_info = _vertex_information.at(v1);
    auto& v2_info = _vertex_information.at(v2);
    if (v1_info.grasp_group != v2_info.grasp_group)
    {
      // we already know that v1 and v2 belong to different grasp groups
      return {v2, v2};
    }
    // get node and check edge
    auto node = _roadmap->getNode(v1_info.roadmap_node_id);
    assert(node);
    auto edge = node->getEdge(v2_info.roadmap_node_id);
    assert(edge);
    auto edge_iter = edge->conditional_costs.find(gid);
    assert(edge_iter == edge->conditional_costs.end());
    // compute edge cost for this grasp
    auto [valid, cost] = _roadmap->computeCost(edge, gid);
    if (cost != edge->base_cost)
    {
      // the edge cost is indeed different for the grasp, so perform a split
      // add a vertex in the grasp group covering only gid
      std::set single_grasp_subset({gid});
      unsigned int single_id = getGraspGroupId(single_grasp_subset);
      unsigned int v2_single_grasp = getVertexId(v2_info.roadmap_node_id, single_id);
      // add a vertex in the grasp group covering the remaining grasps
      auto& parent_group = _grasp_groups.at(v1_info.grasp_group);
      std::set<unsigned int> multi_grasp_subset;
      std::set_difference(parent_group.grasp_set.begin(), parent_group.grasp_set.end(), single_grasp_subset.begin(),
                          single_grasp_subset.end(), std::inserter(multi_grasp_subset, multi_grasp_subset.begin()));
      unsigned int multiple_id = getGraspGroupId(multi_grasp_subset);
      unsigned int v2_multi_grasp = getVertexId(v2_info.roadmap_node_id, multiple_id);
      // add adjacency information
      v1_info.non_group_neighbors.insert(v2_single_grasp);
      v1_info.non_group_neighbors.insert(v2_multi_grasp);
      _vertex_information.at(v2_single_grasp).non_group_neighbors.insert(v1);
      _vertex_information.at(v2_multi_grasp).non_group_neighbors.insert(v1);
      return {v2_single_grasp, v2_multi_grasp};
    }
    return {v2, v2};
  }

  unsigned int LazyMultiGraspRoadmapGraph::getVertexId(unsigned int rid, unsigned int grasp_group_id) const
  {
    // check whether we already know this vertex
    auto iter = _grasp_groups.at(grasp_group_id).roadmap_id_to_vertex.find(rid);
    if (iter != _grasp_groups.at(grasp_group_id).roadmap_id_to_vertex.end())
    {
      return iter->second;
    }
    // else add it
    VertexInformation new_vertex_info;
    new_vertex_info.grasp_group = grasp_group_id;
    new_vertex_info.roadmap_node_id = rid;
    unsigned int vid = _vertex_information.size();
    _vertex_information.push_back(new_vertex_info);
    _grasp_groups.at(grasp_group_id).roadmap_id_to_vertex[rid] = vid;
    return vid;
  }

  unsigned int LazyMultiGraspRoadmapGraph::getGraspGroupId(const std::set<unsigned int>& grasp_set) const
  {
    std::string key = getStringRepresentation(grasp_set);
    auto iter = _grasp_group_ids.find(key);
    if (iter == _grasp_group_ids.end())
    {
      // add new group
      GraspGroup new_group;
      new_group.grasp_set = grasp_set;
      // group 0 contains all goals
      new_group.goal_set = _grasp_groups.at(0).goal_set->createSubset(new_group.grasp_set);
      new_group.cost_to_go = std::make_shared<MultiGoalCostToGo>(new_group.goal_set, _cost_params, _quality_range);
      unsigned int new_group_id = _grasp_groups.size();
      _grasp_groups.push_back(new_group);
      _grasp_group_ids[key] = new_group_id;
    }
    return iter->second;
  }

  std::string LazyMultiGraspRoadmapGraph::getStringRepresentation(const std::set<unsigned int>& grasp_set) const
  {
    std::stringstream ss;
    for (auto grasp_id : grasp_set)
    {
      ss << grasp_id;
    }
    return ss.str();
  }

  std::string LazyMultiGraspRoadmapGraph::getStringRepresentation(unsigned int gid) const
  {
    return std::to_string(gid);
  }
  return std::to_string(gid);
}