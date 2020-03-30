#include <cmath>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>

using namespace placement::mp::mgsearch;

SingleGraspRoadmapGraph::SingleGraspRoadmapGraph(RoadmapPtr roadmap, MultiGraspGoalSetPtr goal_set,
    CostToGoHeuristicPtr cost_to_go, unsigned int grasp_id, unsigned int start_id)
    : _roadmap(roadmap)
    , _goal_set(goal_set)
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
    auto node_v1 = _roadmap->getNode(v1);
    // ensure v1's edges are up-to-date
    _roadmap->updateAdjacency(node_v1);
    auto edge = node_v1->getEdge(v2);
    if (!edge)
        return INFINITY;
    if (lazy) {
        return edge->getBestKnownCost(_grasp_id);
    }
    // else not lazy
    if (not checkValidity(v1))
        return INFINITY;
    if (not checkValidity(v2))
        return INFINITY;
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
    return _goal_set->isGoal(node->uid, _grasp_id);
}

double SingleGraspRoadmapGraph::getGoalCost(uint v) const {
    auto node = _roadmap->getNode(v);
    assert(node);
    auto [goal_id, is_goal] = _goal_set->getGoalId(node->uid, _grasp_id);
    if (not is_goal) {
        return 0.0;
    }
    return _cost_to_go->getGoalCost(_goal_set->getGoal(goal_id).quality);
}

double SingleGraspRoadmapGraph::heuristic(unsigned int v) const
{
    auto node = _roadmap->getNode(v);
    if (!node)
        return INFINITY;
    return _cost_to_go->costToGo(node->config, _grasp_id);
}

std::pair<uint, uint> SingleGraspRoadmapGraph::getGraspRoadmapId(uint vid) const {
    return {vid, _grasp_id};
}

/************************************* MultiGraspRoadmapGraph ********************************/
MultiGraspRoadmapGraph::MultiGraspRoadmapGraph(RoadmapPtr roadmap,
    MultiGraspGoalSetPtr goal_set, CostToGoHeuristicPtr cost_to_go,
    const std::set<unsigned int>& grasp_ids, unsigned int start_id)
    : _roadmap(roadmap)
    , _goal_set(goal_set)
    , _cost_to_go(cost_to_go)
    , _grasp_ids(grasp_ids)
    , _num_graph_nodes(0)
    , _roadmap_start_id(start_id)
{
}

MultiGraspRoadmapGraph::~MultiGraspRoadmapGraph() = default;

bool MultiGraspRoadmapGraph::checkValidity(unsigned int v) const
{
    if (v == 0) {
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
    if (v == 0) // special case for start node
    {
        // the start node is adjacent to every start roadmap node across all grasps
        for (unsigned int gid : _grasp_ids) {
            unsigned int graph_id = toGraphKey(gid, _roadmap_start_id); // same roadmap graph, but a specific grasp
            successors.push_back(graph_id);
        }
    } else {
        // standard case for every other node
        auto [grasp_id, roadmap_node_id] = toRoadmapKey(v);
        // catch special case of start node
        if (roadmap_node_id == _roadmap_start_id) {
            successors.push_back(0);
        }
        auto node = _roadmap->getNode(roadmap_node_id);
        assert(node);
        // ensure the adjacency is up-to-date
        _roadmap->updateAdjacency(node);
        // get edges
        auto [iter, end] = node->getEdgesIterators();
        while (iter != end) {
            if (lazy) {
                // only get best known cost
                double cost = iter->second->getBestKnownCost(grasp_id);
                if (not std::isinf(cost)) {
                    uint graph_id = toGraphKey(grasp_id, iter->first);
                    successors.push_back(graph_id);
                }
            } else {
                // query true edge cost
                auto [valid, cost] = _roadmap->computeCost(iter->second, grasp_id);
                if (valid) {
                    uint graph_id = toGraphKey(grasp_id, iter->first);
                    successors.push_back(graph_id);
                }
            }
            iter++;
        }
    }
}

void MultiGraspRoadmapGraph::getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy) const
{
    // undirected graph
    getSuccessors(v, predecessors, lazy);
}

double MultiGraspRoadmapGraph::getEdgeCost(unsigned int v1, unsigned int v2, bool lazy) const
{
    // catch special case of start node
    if (v1 == 0 || v2 == 0) {
        return 0.0; // TODO could return here costs for obtaining a grasp
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
    if (lazy) {
        return edge->getBestKnownCost(grasp_1);
    }
    if (!checkValidity(v1))
        return INFINITY;
    if (!checkValidity(v2))
        return INFINITY;
    return _roadmap->computeCost(edge, grasp_1).second;
}

unsigned int MultiGraspRoadmapGraph::getStartNode() const
{
    return 0;
}

bool MultiGraspRoadmapGraph::isGoal(unsigned int v) const
{
    if (v == 0) return false;
    auto [grasp_id, rnid] = toRoadmapKey(v);
    return _goal_set->isGoal(rnid, grasp_id);
}

double MultiGraspRoadmapGraph::getGoalCost(uint v) const {
    if (v == 0) { // can not be a goal
        return 0.0;
    }
    auto [grasp_id, rnid] = toRoadmapKey(v);
    auto node = _roadmap->getNode(rnid);
    assert(node);
    auto [goal_id, is_goal] = _goal_set->getGoalId(node->uid, grasp_id);
    if (not is_goal) {
        return 0.0;
    }
    return _cost_to_go->getGoalCost(_goal_set->getGoal(goal_id).quality);
}

double MultiGraspRoadmapGraph::heuristic(unsigned int v) const
{
    if (v == 0) {
        auto node = _roadmap->getNode(_roadmap_start_id);
        return _cost_to_go->costToGo(node->config);
    }
    // else
    auto [grasp_id, rnid] = toRoadmapKey(v);
    auto node = _roadmap->getNode(rnid);
    if (!node)
        return INFINITY;
    return _cost_to_go->costToGo(node->config, grasp_id);
}

std::pair<uint, uint> MultiGraspRoadmapGraph::getGraspRoadmapId(uint vid) const {
    if (vid == 0) {
        return {_roadmap_start_id, 0}; // TODO what makes sense to return here for the grasp?
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
    if (iter == _roadmap_key_to_graph.end()) {
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
    return toGraphKey({ grasp_id, node_id });
}
