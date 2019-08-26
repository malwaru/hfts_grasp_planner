#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>

using namespace placement::mp::mgsearch;

SingleGraspRoadmapGraph::SingleGraspRoadmapGraph(RoadmapPtr roadmap, unsigned int grasp_id)
    : _roadmap(roadmap)
    , _grasp_id(grasp_id)
    , _has_goal(false)
{
}

SingleGraspRoadmapGraph::~SingleGraspRoadmapGraph() = default;

void SingleGraspRoadmapGraph::setStartId(unsigned int start_id)
{
    _start_id = start_id;
    _has_goal = true;
}

bool SingleGraspRoadmapGraph::checkValidity(unsigned int v) const
{
    auto node = _roadmap->getNode(v);
    if (!node)
        return false;
    // TODO check whether the node is valid for the set grasp
    return _roadmap->checkNode(node);
}

void SingleGraspRoadmapGraph::getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy) const
{
    auto node = _roadmap->getNode(v);
    if (!_roadmap->checkNode(node))
        return;
    // if lazy, do not verify that edges are collision-free
    // node->getNeighbors()
    // TODO
    // if not lazy, only return edges that are valid for this grasp
}

void SingleGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy) const
{
    // undirected graph
    getSuccessors(v, predecessors, lazy);
}

double SingleGraspRoadmapGraph::heuristic(unsigned int v) const
{
}

double SingleGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy) const
{
    if (!checkValidity(v1))
        return INFINITY;
    if (!checkValidity(v2))
        return INFINITY;
    auto node_v1 = _roadmap->getNode(v1);
    auto edge = node_v1->getEdge(v2);
    if (!edge)
        return INFINITY;
    if (!_roadmap->checkEdge(edge)) {
        return INFINITY;
    }
    if (lazy) {
        return edge->base_cost;
    }
    return _roadmap->computeCost(edge, _grasp_id).second;
}

unsigned int SingleGraspRoadmapGraph::getStartNode() const
{
    assert(_has_goal);
    return _start_id;
}

bool SingleGraspRoadmapGraph::isGoal(unsigned int v) const
{
    auto node = _roadmap->getNode(v);
    assert(node);
    return node->is_goal;
}