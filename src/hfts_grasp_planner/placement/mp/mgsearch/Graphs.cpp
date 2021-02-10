#include <cmath>
#include <algorithm>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>

using namespace placement::mp::mgsearch;

/******************************** DynamicNeighborIterator **************************************/
DynamicNeighborIterator::DynamicNeighborIterator() : _impl(nullptr)
{
}

DynamicNeighborIterator::DynamicNeighborIterator(std::unique_ptr<IteratorImplementation>& impl) : _impl(std::move(impl))
{
  if (_impl and _impl->isEnd())
    _impl.reset();
}

DynamicNeighborIterator::DynamicNeighborIterator(DynamicNeighborIterator&& other) : _impl(std::move(other._impl))
{
}

DynamicNeighborIterator::DynamicNeighborIterator(const DynamicNeighborIterator& other)
{
  if (other._impl)
  {
    _impl = other._impl->copy();
  }
}

DynamicNeighborIterator& DynamicNeighborIterator::operator++()
{
  assert(_impl);
  _impl->next();
  if (_impl->isEnd())
    _impl.reset();
  return *this;
}

bool DynamicNeighborIterator::operator==(const DynamicNeighborIterator& other) const
{
  return (_impl == nullptr and other._impl == nullptr) or _impl->equals(other._impl.get());
}

bool DynamicNeighborIterator::operator!=(const DynamicNeighborIterator& other) const
{
  return not operator==(other);
}

unsigned int DynamicNeighborIterator::operator*()
{
  return _impl->dereference();
}

DynamicNeighborIterator::IteratorImplementation::~IteratorImplementation() = default;

/********************************* VertexExpansionLogger *************************************/
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

void VertexExpansionLogger::logGoalExpansion()
{
  // _roadmap->logCustomEvent("GOAL_EXPANSION");
}
/************************************* SingleGraspRoadmapGraph::GoalVertexIterator ********************************/
SingleGraspRoadmapGraph::GoalVertexIterator::GoalVertexIterator(SingleGraspRoadmapGraph const* parent)
  : _graph(parent), _iter(_graph->_goal_set->begin()), _end(_graph->_goal_set->end())
{
}

bool SingleGraspRoadmapGraph::GoalVertexIterator::equals(
    const DynamicNeighborIterator::IteratorImplementation* const other) const
{
  auto other_casted = dynamic_cast<const GoalVertexIterator*>(other);
  if (!other_casted)
    return false;
  return _iter == other_casted->_iter;
}

unsigned int SingleGraspRoadmapGraph::GoalVertexIterator::dereference() const
{
  assert(_iter != _end);
  unsigned int rid = _graph->_goal_set->getRoadmapId(_iter->id);
  return _graph->toVertexId(rid);
}

void SingleGraspRoadmapGraph::GoalVertexIterator::next()
{
  ++_iter;
}

std::unique_ptr<DynamicNeighborIterator::IteratorImplementation>
SingleGraspRoadmapGraph::GoalVertexIterator::copy() const
{
  auto new_copy = std::make_unique<GoalVertexIterator>(_graph);
  new_copy->_iter = _iter;
  return new_copy;
}

bool SingleGraspRoadmapGraph::GoalVertexIterator::isEnd() const
{
  return _iter == _end;
}

/********************************* SingleGraspRoadmapGraph *************************************/

// SingleGraspRoadmapGraph::NeighborIterator
// SingleGraspRoadmapGraph::NeighborIterator::begin(unsigned int v, bool lazy, SingleGraspRoadmapGraph const* parent)
// {
//   if (v == GOAL_VERTEX_ID)
//     return NeighborIterator();
//   auto node = parent->_roadmap->getNode(v);
//   if (!node)
//   {  // return end iterator
//     return NeighborIterator();
//     // throw std::logic_error("Invalid vertex node");
//   }
//   parent->_roadmap->updateAdjacency(node);
//   auto [begin, end] = node->getEdgesIterators();
//   return NeighborIterator(begin, end, lazy, parent);
// }

// SingleGraspRoadmapGraph::NeighborIterator
// SingleGraspRoadmapGraph::NeighborIterator::end(unsigned int v, SingleGraspRoadmapGraph const* parent)
// {
//   return NeighborIterator();
// }

SingleGraspRoadmapGraph::SingleGraspRoadmapGraph(RoadmapPtr roadmap, MultiGraspGoalSetPtr goal_set,
                                                 const ::placement::mp::mgsearch::GoalPathCostParameters& params,
                                                 unsigned int grasp_id, unsigned int start_id)
  : _roadmap(roadmap)
  , _goal_set(goal_set->createSubset(std::set({grasp_id})))
  , _cost_to_go(_goal_set, params, goal_set->getGoalQualityRange())
  , _grasp_id(grasp_id)
  , _start_rid(start_id)
  , _logger(_roadmap)
{
}

SingleGraspRoadmapGraph::~SingleGraspRoadmapGraph() = default;

bool SingleGraspRoadmapGraph::checkValidity(unsigned int v)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::checkValidity");
#endif
  if (v == GOAL_VERTEX_ID)
    return true;
  auto node = _roadmap->getNode(toRoadmapId(v));
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
  if (v != GOAL_VERTEX_ID)
  {
    _logger.logExpansion(toRoadmapId(v), _grasp_id);
  }
  else
  {
    _logger.logGoalExpansion();
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::getSuccessors");
#endif
  std::unique_ptr<DynamicNeighborIterator::IteratorImplementation> impl;  // default initialization is end
  if (v != GOAL_VERTEX_ID)                                                // goal vertex has no successors
  {
    unsigned int rid = toRoadmapId(v);
    auto node = _roadmap->getNode(rid);
    _roadmap->updateAdjacency(node);
    if (node)
    {
      if (_goal_set->isGoal(rid, _grasp_id))
      {  // check if this vertex is an entrance to goal
        if (lazy)
        {
          impl.reset(new OneMoreIterator<StandardIterator<true>>(StandardIterator<true>(this, rid), GOAL_VERTEX_ID));
        }
        else
        {
          impl.reset(new OneMoreIterator<StandardIterator<false>>(StandardIterator<false>(this, rid), GOAL_VERTEX_ID));
        }
      }
      else
      {
        if (lazy)
        {
          impl.reset(new StandardIterator<true>(this, rid));
        }
        else
        {
          impl.reset(new StandardIterator<false>(this, rid));
        }
      }
    }
  }
  return {DynamicNeighborIterator(impl), DynamicNeighborIterator()};
}

void SingleGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy)
{
  predecessors.clear();
  auto [iter, end] = getPredecessors(v, lazy);
  predecessors.insert(predecessors.begin(), iter, end);
}

std::pair<SingleGraspRoadmapGraph::NeighborIterator, SingleGraspRoadmapGraph::NeighborIterator>
SingleGraspRoadmapGraph::getPredecessors(unsigned int v, bool lazy)
{
#ifdef ENABLE_GRAPH_LOGGING
  if (v != GOAL_VERTEX_ID)
  {
    _logger.logExpansion(toRoadmapId(v), _grasp_id);
  }
  else
  {
    _logger.logGoalExpansion();
  }
#endif
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::getPredecessors");
#endif
  std::unique_ptr<DynamicNeighborIterator::IteratorImplementation> impl;
  if (v == GOAL_VERTEX_ID)
  {
    impl.reset(new GoalVertexIterator(this));
  }
  else
  {
    unsigned int rid = toRoadmapId(v);
    auto node = _roadmap->getNode(rid);
    _roadmap->updateAdjacency(node);
    if (lazy)
    {
      impl.reset(new StandardIterator<true>(this, rid));
    }
    else
    {
      impl.reset(new StandardIterator<false>(this, rid));
    }
  }
  return {DynamicNeighborIterator(impl), DynamicNeighborIterator()};
}

double SingleGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy)
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::getEdgeCost");
#endif
  if (v1 == GOAL_VERTEX_ID || v2 == GOAL_VERTEX_ID)
  {  // get goal cost
    unsigned int v = v1 == GOAL_VERTEX_ID ? v2 : v1;
    auto [goal_id, is_goal] = _goal_set->getGoalId(toRoadmapId(v), _grasp_id);
    if (not is_goal)
    {
      return std::numeric_limits<double>::infinity();
    }
    return _cost_to_go.qualityToGoalCost(_goal_set->getGoal(goal_id).quality);
  }
  // standard case of a normal edge
  unsigned int rid1 = toRoadmapId(v1);
  unsigned int rid2 = toRoadmapId(v2);
  auto node_v1 = _roadmap->getNode(rid1);
  // ensure v1's edges are up-to-date
  _roadmap->updateAdjacency(node_v1);
  auto edge = node_v1->getEdge(rid2);
  if (!edge)
    return INFINITY;
  if (lazy)
  {
    return edge->getBestKnownCost(_grasp_id);
  }
  // else not lazy
  if (not _roadmap->isValid(rid1, _grasp_id))
    return INFINITY;
  if (not _roadmap->isValid(rid2, _grasp_id))
    return INFINITY;
  return _roadmap->computeCost(edge, _grasp_id).second;
}

bool SingleGraspRoadmapGraph::trueEdgeCostKnown(unsigned int v1, unsigned int v2) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::trueEdgeCostKnown");
#endif
  if (v1 == GOAL_VERTEX_ID or v2 == GOAL_VERTEX_ID)
  {
    return true;
  }
  auto node_v1 = _roadmap->getNode(toRoadmapId(v1));
  if (!node_v1)
    return true;                       // node must have got deleted -> infinite cost is known now
  _roadmap->updateAdjacency(node_v1);  // TODO do we actually need this here?
  auto edge = node_v1->getEdge(toRoadmapId(v2));
  if (!edge)
    return true;  // the edge doesn't exist -> infinite cost is known now
  return edge->conditional_costs.find(_grasp_id) != edge->conditional_costs.end();  // we have a cost for this grasp
}

unsigned int SingleGraspRoadmapGraph::getStartVertex() const
{
  return START_VERTEX_ID;
}

unsigned int SingleGraspRoadmapGraph::getGoalVertex() const
{
  return GOAL_VERTEX_ID;
}

double SingleGraspRoadmapGraph::heuristic(unsigned int v) const
{
#ifdef ENABLE_GRAPH_PROFILING
  utils::ScopedProfiler profiler("SingleGraspRoadmapGraph::heuristic");
#endif
  if (v == GOAL_VERTEX_ID)  // goal vertex
    return 0;
  auto node = _roadmap->getNode(toRoadmapId(v));
  if (!node)
    return INFINITY;
  return _cost_to_go.costToGo(node->config);
}

std::pair<unsigned int, unsigned int> SingleGraspRoadmapGraph::getGraspRoadmapId(unsigned int vid) const
{
  if (vid == GOAL_VERTEX_ID)
  {
    throw std::logic_error("Can not provide a roadmap id for goal vertex");
  }
  return {toRoadmapId(vid), _grasp_id};
}

unsigned int SingleGraspRoadmapGraph::toVertexId(unsigned int rid) const
{
  if (rid != _start_rid)
    return rid + 2;
  return 0;
}

unsigned int SingleGraspRoadmapGraph::toRoadmapId(unsigned int vid) const
{
  assert(vid != GOAL_VERTEX_ID);  // we have no roadmap id for the goal vertex
  if (vid == START_VERTEX_ID)
  {
    return _start_rid;
  }
  return vid - 2;
}
