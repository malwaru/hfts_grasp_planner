#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>

using namespace placement::mp::mgsearch;

SingleGraspRoadmapGraph::SingleGraspRoadmapGraph(RoadmapPtr roadmap, CostToGoHeuristicPtr cost_to_go, unsigned int grasp_id)
    : _roadmap(roadmap)
    , _cost_to_go(cost_to_go)
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
    return _roadmap->checkNode(node, _grasp_id);
}

void SingleGraspRoadmapGraph::getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy) const
{
    successors.clear();
    auto node = _roadmap->getNode(v);
    if (!node)
        return;
    // we need to check the node also when we are lazy because it computes the node's adjacency
    if (!_roadmap->checkNode(node, _grasp_id))
        return;
    // get edges
    auto [iter, end] = node->getEdgesIterators();
    while (iter != end) {
        if (lazy) {
            // only get best known cost
            double cost = iter->second->getBestKnownCost(_grasp_id);
            if (not std::isinf(cost))
                successors.push_back(iter->first);
        } else {
            // query true edge cost
            auto [valid, cost] = _roadmap->computeCost(iter->second, _grasp_id);
            if (valid) {
                successors.push_back(iter->first);
            }
        }
        iter++;
    }
}

void SingleGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy) const
{
    // undirected graph
    getSuccessors(v, predecessors, lazy);
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
    if (lazy) {
        return edge->getBestKnownCost(_grasp_id);
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

double SingleGraspRoadmapGraph::heuristic(unsigned int v) const
{
    auto node = _roadmap->getNode(v);
    if (!node)
        return INFINITY;
    _cost_to_go->costToGo(node->config, _grasp_id);
}