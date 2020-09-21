#include <cmath>
#include <algorithm>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>

using namespace placement::mp::mgsearch;

SingleGraspRoadmapGraph::NeighborIterator::NeighborIterator(Roadmap::Node::EdgeIterator eiter,
                                                            Roadmap::Node::EdgeIterator end, bool lazy,
                                                            SingleGraspRoadmapGraph const* parent)
  : _iter(eiter), _end(end), _lazy(lazy), _graph(parent)
{
  forwardToNextValid();
}

SingleGraspRoadmapGraph::NeighborIterator& SingleGraspRoadmapGraph::NeighborIterator::operator++()
{
  ++_iter;
  forwardToNextValid();
  return (*this);
}

bool SingleGraspRoadmapGraph::NeighborIterator::operator==(const SingleGraspRoadmapGraph::NeighborIterator& other) const
{
  return other._iter == _iter;
}

bool SingleGraspRoadmapGraph::NeighborIterator::operator!=(const SingleGraspRoadmapGraph::NeighborIterator& other) const
{
  return not operator==(other);
}

unsigned int SingleGraspRoadmapGraph::NeighborIterator::operator*()
{
  return _iter->first;
}

void SingleGraspRoadmapGraph::NeighborIterator::forwardToNextValid()
{
  bool valid = false;
  while (!valid and _iter != _end)
  {
    if (_lazy)
    {
      double cost = _iter->second->getBestKnownCost(_graph->_grasp_id);
      valid = not std::isinf(cost);
    }
    else
    {
      if (_graph->_roadmap->isValid(_iter->first, _graph->_grasp_id))
      {
        auto [lvalid, cost] = _graph->_roadmap->computeCost(_iter->second, _graph->_grasp_id);
        valid = lvalid;
      }
    }
    if (not valid)
    {
      ++_iter;
    }
  }
}

SingleGraspRoadmapGraph::NeighborIterator
SingleGraspRoadmapGraph::NeighborIterator::begin(unsigned int v, bool lazy, SingleGraspRoadmapGraph const* parent)
{
  auto node = parent->_roadmap->getNode(v);
  if (!node)
  {
    throw std::logic_error("Invalid vertex node");
  }
  parent->_roadmap->updateAdjacency(node);
  auto [begin, end] = node->getEdgesIterators();
  return NeighborIterator(begin, end, lazy, parent);
}

SingleGraspRoadmapGraph::NeighborIterator
SingleGraspRoadmapGraph::NeighborIterator::end(unsigned int v, SingleGraspRoadmapGraph const* parent)
{
  auto node = parent->_roadmap->getNode(v);
  if (!node)
  {
    throw std::logic_error("Invalid vertex node");
  }
  auto [begin, end] = node->getEdgesIterators();
  return NeighborIterator(end, end, true, parent);
}

SingleGraspRoadmapGraph::SingleGraspRoadmapGraph(RoadmapPtr roadmap, MultiGraspGoalSetPtr goal_set,
                                                 const ::placement::mp::mgsearch::GoalPathCostParameters& params,
                                                 unsigned int grasp_id, unsigned int start_id)
  : _roadmap(roadmap)
  , _goal_set(goal_set->createSubset(std::set({grasp_id})))
  , _cost_to_go(_goal_set, params, goal_set->getGoalQualityRange())
  , _grasp_id(grasp_id)
  , _start_id(start_id)
{
}

SingleGraspRoadmapGraph::~SingleGraspRoadmapGraph() = default;

bool SingleGraspRoadmapGraph::checkValidity(unsigned int v) const
{
  auto node = _roadmap->getNode(v);
  if (!node)
    return false;
  return _roadmap->isValid(node, _grasp_id);
}

void SingleGraspRoadmapGraph::getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy) const
{
  successors.clear();
  auto [iter, end] = getSuccessors(v, lazy);
  successors.insert(successors.begin(), iter, end);
}

std::pair<SingleGraspRoadmapGraph::NeighborIterator, SingleGraspRoadmapGraph::NeighborIterator>
SingleGraspRoadmapGraph::getSuccessors(unsigned int v, bool lazy) const
{
  return {NeighborIterator::begin(v, lazy, this), NeighborIterator::end(v, this)};
}

void SingleGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy) const
{
  // undirected graph
  getSuccessors(v, predecessors, lazy);
}

std::pair<SingleGraspRoadmapGraph::NeighborIterator, SingleGraspRoadmapGraph::NeighborIterator>
SingleGraspRoadmapGraph::getPredecessors(unsigned int v, bool lazy) const
{
  return getSuccessors(v, lazy);
}

double SingleGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy) const
{
  auto node_v1 = _roadmap->getNode(v1);
  // ensure v1's edges are up-to-date
  _roadmap->updateAdjacency(node_v1);
  auto edge = node_v1->getEdge(v2);
  if (!edge)
    return INFINITY;
  if (lazy)
  {
    return edge->getBestKnownCost(_grasp_id);
  }
  // else not lazy
  if (not checkValidity(v1))
    return INFINITY;
  if (not checkValidity(v2))
    return INFINITY;
  return _roadmap->computeCost(edge, _grasp_id).second;
}

bool SingleGraspRoadmapGraph::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
{
  auto node_v1 = _roadmap->getNode(v1);
  if (!node_v1)
    return true;                       // node must have got deleted -> infinite cost is known now
  _roadmap->updateAdjacency(node_v1);  // TODO do we actually need this here?
  auto edge = node_v1->getEdge(v2);
  if (!edge)
    return true;  // the edge doesn't exist -> infinite cost is known now
  return edge->conditional_costs.find(_grasp_id) != edge->conditional_costs.end();  // we have a cost for this grasp
}

unsigned int SingleGraspRoadmapGraph::getStartNode() const
{
  return _start_id;
}

bool SingleGraspRoadmapGraph::isGoal(unsigned int v) const
{
  auto node = _roadmap->getNode(v);
  if (!node)
    return false;
  return _goal_set->isGoal(node->uid, _grasp_id);
}

double SingleGraspRoadmapGraph::getGoalCost(unsigned int v) const
{
  auto node = _roadmap->getNode(v);
  assert(node);
  auto [goal_id, is_goal] = _goal_set->getGoalId(node->uid, _grasp_id);
  if (not is_goal)
  {
    return std::numeric_limits<double>::infinity();
  }
  return _cost_to_go.qualityToGoalCost(_goal_set->getGoal(goal_id).quality);
}

double SingleGraspRoadmapGraph::heuristic(unsigned int v) const
{
  auto node = _roadmap->getNode(v);
  if (!node)
    return INFINITY;
  return _cost_to_go.costToGo(node->config);
}

std::pair<unsigned int, unsigned int> SingleGraspRoadmapGraph::getGraspRoadmapId(unsigned int vid) const
{
  return {vid, _grasp_id};
}

/************************************* MultiGraspRoadmapGraph::NeighborIterator ********************************/
MultiGraspRoadmapGraph::NeighborIterator::NeighborIterator(unsigned int v, bool lazy,
                                                           MultiGraspRoadmapGraph const* parent)
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

MultiGraspRoadmapGraph::NeighborIterator& MultiGraspRoadmapGraph::NeighborIterator::operator++()
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

bool MultiGraspRoadmapGraph::NeighborIterator::operator==(const MultiGraspRoadmapGraph::NeighborIterator& other) const
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

bool MultiGraspRoadmapGraph::NeighborIterator::operator!=(const MultiGraspRoadmapGraph::NeighborIterator& other) const
{
  return not operator==(other);
}

unsigned int MultiGraspRoadmapGraph::NeighborIterator::operator*()
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

void MultiGraspRoadmapGraph::NeighborIterator::forwardToNextValid()
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
      if (_graph->_roadmap->isValid(_iter->first, _grasp_id))
      {
        auto [lvalid, cost] = _graph->_roadmap->computeCost(_iter->second, _grasp_id);
        valid = lvalid;
      }
    }
    if (not valid)
    {
      ++_iter;
    }
  }
}

MultiGraspRoadmapGraph::NeighborIterator
MultiGraspRoadmapGraph::NeighborIterator::begin(unsigned int v, bool lazy, MultiGraspRoadmapGraph const* graph)
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

MultiGraspRoadmapGraph::NeighborIterator
MultiGraspRoadmapGraph::NeighborIterator::end(unsigned int v, MultiGraspRoadmapGraph const* graph)
{
  NeighborIterator end_iter(v, true, graph);
  end_iter._grasp_iter = graph->_grasp_ids.end();  // covers end condition for v = 0
  end_iter._iter = end_iter._end;                  // covers end condition for v != 0
  end_iter._edge_to_0_returned = true;             // covers end condition for v != 0
  return end_iter;
}

/************************************* MultiGraspRoadmapGraph ********************************/
MultiGraspRoadmapGraph::MultiGraspRoadmapGraph(RoadmapPtr roadmap, MultiGraspGoalSetPtr goal_set,
                                               const ::placement::mp::mgsearch::GoalPathCostParameters& cost_params,
                                               const std::set<unsigned int>& grasp_ids, unsigned int start_id)
  : _roadmap(roadmap)
  , _goal_set(goal_set)
  , _all_grasps_cost_to_go(goal_set, cost_params)
  , _grasp_ids(grasp_ids)
  , _roadmap_start_id(start_id)
  , _num_graph_nodes(0)
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

MultiGraspRoadmapGraph::~MultiGraspRoadmapGraph() = default;

bool MultiGraspRoadmapGraph::checkValidity(unsigned int v) const
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

void MultiGraspRoadmapGraph::getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy) const
{
  successors.clear();
  auto begin = NeighborIterator::begin(v, lazy, this);
  auto end = NeighborIterator::end(v, this);
  successors.insert(successors.begin(), begin, end);
}

std::pair<MultiGraspRoadmapGraph::NeighborIterator, MultiGraspRoadmapGraph::NeighborIterator>
MultiGraspRoadmapGraph::getSuccessors(unsigned int v, bool lazy) const
{
  return {NeighborIterator::begin(v, lazy, this), NeighborIterator::end(v, this)};
}

void MultiGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy) const
{
  // undirected graph
  getSuccessors(v, predecessors, lazy);
}

std::pair<MultiGraspRoadmapGraph::NeighborIterator, MultiGraspRoadmapGraph::NeighborIterator>
MultiGraspRoadmapGraph::getPredecessors(unsigned int v, bool lazy) const
{
  return getSuccessors(v, lazy);
}

double MultiGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy) const
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
  return _roadmap->computeCost(edge, grasp_1).second;
}

bool MultiGraspRoadmapGraph::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
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

unsigned int MultiGraspRoadmapGraph::getStartNode() const
{
  return 0;
}

bool MultiGraspRoadmapGraph::isGoal(unsigned int v) const
{
  if (v == 0)
    return false;
  auto [grasp_id, rnid] = toRoadmapKey(v);
  return _goal_set->isGoal(rnid, grasp_id);
}

double MultiGraspRoadmapGraph::getGoalCost(unsigned int v) const
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

double MultiGraspRoadmapGraph::heuristic(unsigned int v) const
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

std::pair<unsigned int, unsigned int> MultiGraspRoadmapGraph::getGraspRoadmapId(unsigned int vid) const
{
  if (vid == 0)
  {
    return {_roadmap_start_id, 0};  // TODO what makes sense to return here for the grasp?
  }
  auto [grasp_id, rid] = toRoadmapKey(vid);
  return {rid, grasp_id};
}

std::pair<unsigned int, unsigned int> MultiGraspRoadmapGraph::toRoadmapKey(unsigned int graph_id) const
{
  auto iter = _graph_key_to_roadmap.find(graph_id);
  assert(iter != _graph_key_to_roadmap.end());
  return iter->second;
}

unsigned int MultiGraspRoadmapGraph::toGraphKey(const std::pair<unsigned int, unsigned int>& roadmap_id) const
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

unsigned int MultiGraspRoadmapGraph::toGraphKey(unsigned int grasp_id, unsigned int node_id) const
{
  return toGraphKey({grasp_id, node_id});
}
