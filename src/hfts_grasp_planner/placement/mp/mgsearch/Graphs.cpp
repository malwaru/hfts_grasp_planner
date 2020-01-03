#include <cmath>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>

using namespace placement::mp::mgsearch;

SingleGraspRoadmapGraph::SingleGraspRoadmapGraph(RoadmapPtr roadmap, CostToGoHeuristicPtr cost_to_go, unsigned int grasp_id, unsigned int start_id)
    : _roadmap(roadmap)
    , _cost_to_go(cost_to_go)
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
    auto node = _roadmap->getNode(v);
    if (!node)
        return;
    // ensure the adjacency is up-to-date
    _roadmap->updateAdjacency(node);
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
    // ensure v1's edges are up-to-date
    _roadmap->updateAdjacency(node_v1);
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
    return _cost_to_go->costToGo(node->config, _grasp_id);
}

/************************************* MultiGraspRoadmapGraph ********************************/
MultiGraspRoadmapGraph::MultiGraspRoadmapGraph(RoadmapPtr roadmap, CostToGoHeuristicPtr cost_to_go,
    const std::vector<unsigned int>& grasp_ids, unsigned int start_id)
    : _roadmap(roadmap)
    , _cost_to_go(cost_to_go)
    , _grasp_ids(grasp_ids)
{
    // TODO save start node in roadmap
}

MultiGraspRoadmapGraph::~MultiGraspRoadmapGraph() = default;

bool MultiGraspRoadmapGraph::checkValidity(unsigned int v) const
{
    // TODO catch special case if v is start node
    auto [grasp_id, roadmap_id] = toRoadmapKey(v);
    auto node = _roadmap->getNode(roadmap_id);
    if (!node)
        return false;
    return _roadmap->isValid(node, grasp_id);
}

void MultiGraspRoadmapGraph::getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy) const
{
    successors.clear();
    auto node = _roadmap->getNode(v);
    if (!node)
        return;
    // ensure the adjacency is up-to-date
    _roadmap->updateAdjacency(node);
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

void MultiGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy) const
{
    // undirected graph
    getSuccessors(v, predecessors, lazy);
}

double MultiGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy) const
{
    if (!checkValidity(v1))
        return INFINITY;
    if (!checkValidity(v2))
        return INFINITY;
    auto node_v1 = _roadmap->getNode(v1);
    // ensure v1's edges are up-to-date
    _roadmap->updateAdjacency(node_v1);
    auto edge = node_v1->getEdge(v2);
    if (!edge)
        return INFINITY;
    if (lazy) {
        return edge->getBestKnownCost(_grasp_id);
    }
    return _roadmap->computeCost(edge, _grasp_id).second;
}

unsigned int MultiGraspRoadmapGraph::getStartNode() const
{
    return _start_id;
}

bool MultiGraspRoadmapGraph::isGoal(unsigned int v) const
{
    auto node = _roadmap->getNode(v);
    assert(node);
    return node->is_goal;
}

double MultiGraspRoadmapGraph::heuristic(unsigned int v) const
{
    auto node = _roadmap->getNode(v);
    if (!node)
        return INFINITY;
    return _cost_to_go->costToGo(node->config, _grasp_id);
}