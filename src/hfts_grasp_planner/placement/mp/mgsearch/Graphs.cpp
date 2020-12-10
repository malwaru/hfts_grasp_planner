#include <cmath>
#include <algorithm>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>

using namespace placement::mp::mgsearch;

VertexExpansionLogger::VertexExpansionLogger(RoadmapPtr roadmap) : _roadmap(roadmap)
{
}

VertexExpansionLogger::~VertexExpansionLogger() = default;

void VertexExpansionLogger::logExpansion(unsigned int rid)
{
  _roadmap->logCustomEvent("BASE_EXPANSION, " + std::to_string(rid));
}

void VertexExpansionLogger::logExpansion(unsigned int rid, unsigned int gid)
{
  _roadmap->logCustomEvent("EXPANSION, " + std::to_string(rid) + ", " + std::to_string(gid));
}
/********************************* SingleGraspRoadmapGraph *************************************/
SingleGraspRoadmapGraph::NeighborIterator::NeighborIterator(Roadmap::Node::EdgeIterator eiter,
                                                            Roadmap::Node::EdgeIterator end, bool lazy,
                                                            SingleGraspRoadmapGraph const* parent)
  : _iter(eiter), _end(end), _lazy(lazy), _is_end(false), _graph(parent)
{
  forwardToNextValid();
}

SingleGraspRoadmapGraph::NeighborIterator::NeighborIterator() : _lazy(false), _is_end(true)
{
}

SingleGraspRoadmapGraph::NeighborIterator& SingleGraspRoadmapGraph::NeighborIterator::operator++()
{
  assert(!_is_end);
  ++_iter;
  forwardToNextValid();
  return (*this);
}

bool SingleGraspRoadmapGraph::NeighborIterator::operator==(const SingleGraspRoadmapGraph::NeighborIterator& other) const
{
  return other._iter == _iter or (_is_end and other._is_end);
}

bool SingleGraspRoadmapGraph::NeighborIterator::operator!=(const SingleGraspRoadmapGraph::NeighborIterator& other) const
{
  return not operator==(other);
}

unsigned int SingleGraspRoadmapGraph::NeighborIterator::operator*()
{
  assert(!_is_end);
  return _iter->first;
}

void SingleGraspRoadmapGraph::NeighborIterator::forwardToNextValid()
{
  assert(!_is_end);
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
  _is_end = _iter == _end;
}

SingleGraspRoadmapGraph::NeighborIterator
SingleGraspRoadmapGraph::NeighborIterator::begin(unsigned int v, bool lazy, SingleGraspRoadmapGraph const* parent)
{
  auto node = parent->_roadmap->getNode(v);
  if (!node)
  {  // return end iterator
    return NeighborIterator();
    // throw std::logic_error("Invalid vertex node");
  }
  parent->_roadmap->updateAdjacency(node);
  auto [begin, end] = node->getEdgesIterators();
  return NeighborIterator(begin, end, lazy, parent);
}

SingleGraspRoadmapGraph::NeighborIterator
SingleGraspRoadmapGraph::NeighborIterator::end(unsigned int v, SingleGraspRoadmapGraph const* parent)
{
  return NeighborIterator();
}

SingleGraspRoadmapGraph::SingleGraspRoadmapGraph(RoadmapPtr roadmap, MultiGraspGoalSetPtr goal_set,
                                                 const ::placement::mp::mgsearch::GoalPathCostParameters& params,
                                                 unsigned int grasp_id, unsigned int start_id)
  : _roadmap(roadmap)
  , _goal_set(goal_set->createSubset(std::set({grasp_id})))
  , _cost_to_go(_goal_set, params, goal_set->getGoalQualityRange())
  , _grasp_id(grasp_id)
  , _start_id(start_id)
  , _logger(_roadmap)
{
}

SingleGraspRoadmapGraph::~SingleGraspRoadmapGraph() = default;

bool SingleGraspRoadmapGraph::checkValidity(unsigned int v)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::checkValidity");
#endif
  auto node = _roadmap->getNode(v);
  if (!node)
    return false;
  return _roadmap->isValid(node, _grasp_id);
}

void SingleGraspRoadmapGraph::getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy)
{
  successors.clear();
  auto [iter, end] = getSuccessors(v, lazy);
  successors.insert(successors.begin(), iter, end);
}

std::pair<SingleGraspRoadmapGraph::NeighborIterator, SingleGraspRoadmapGraph::NeighborIterator>
SingleGraspRoadmapGraph::getSuccessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  _logger.logExpansion(v, _grasp_id);
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::getSuccessors");
#endif
  return {NeighborIterator::begin(v, lazy, this), NeighborIterator::end(v, this)};
}

void SingleGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy)
{
  // undirected graph
  getSuccessors(v, predecessors, lazy);
}

std::pair<SingleGraspRoadmapGraph::NeighborIterator, SingleGraspRoadmapGraph::NeighborIterator>
SingleGraspRoadmapGraph::getPredecessors(unsigned int v, bool lazy)
{
  return getSuccessors(v, lazy);
}

double SingleGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::getEdgeCost");
#endif
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
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::trueEdgeCostKnown");
#endif
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
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::isGoal");
#endif
  auto node = _roadmap->getNode(v);
  if (!node)
    return false;
  return _goal_set->isGoal(node->uid, _grasp_id);
}

double SingleGraspRoadmapGraph::getGoalCost(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::getGoalCost");
#endif
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
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::heuristic");
#endif
  auto node = _roadmap->getNode(v);
  if (!node)
    return INFINITY;
  return _cost_to_go.costToGo(node->config);
}

std::pair<unsigned int, unsigned int> SingleGraspRoadmapGraph::getGraspRoadmapId(unsigned int vid) const
{
  return {vid, _grasp_id};
}