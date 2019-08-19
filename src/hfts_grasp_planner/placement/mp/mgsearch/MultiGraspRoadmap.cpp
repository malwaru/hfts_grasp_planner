#include <hfts_grasp_planner/external/halton/halton.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>

using namespace placement::mp::mgsearch;

double distanceFn(const Roadmap::NodePtr& a, const Roadmap::NodePtr& b)
{
    double dist = 0.0;
    for (unsigned int i = 0; i < a->config.size(); ++i) {
        double dq = a->config.at(i) - b->config.at(i);
        dist += dq * dq;
    }
    return sqrt(dist);
}

Roadmap::Edge::Edge(Roadmap::NodePtr a, Roadmap::NodePtr b)
    : base_cost(distanceFn(a, b))
    , base_evaluated(false)
    , node_a(a)
    , node_b(b)
{
}

Roadmap::Roadmap(OpenRAVE::RobotBasePtr robot, StateValidityCheckerPtr validity_checker,
    EdgeCostComputerPtr cost_computer, unsigned int batch_size)
    : _robot(robot)
    , _validity_checker(validity_checker)
    , _cost_computer(cost_computer)
    , _batch_size(batch_size)
    , _node_id_counter(0)
    , _halton_seq_id(0)
    , _densification_gen(0)
{
    // TODO set different distance function if needed (lowerBound of _cost_computer?)
    _nn.setDistanceFunction(distanceFn);
    // compute gamma_prm - a constant used to compute the radius for adjacency
    unsigned int dof = _robot->GetActiveDOF();
    // we need the measure of X_free for this, we approximate it by the measure of X
    double mu = 1.0;
    std::vector<OpenRAVE::dReal> lower;
    std::vector<OpenRAVE::dReal> upper;
    _robot->GetActiveDOFLimits(lower, upper);
    for (unsigned int i = 0; i < lower.size(); ++i) {
        mu *= (upper.at(i) - lower.at(i));
    }
    // xi is the measure of a dof-dimensional unit ball
    double xi = pow(M_PI, dof / 2.0) / tgamma(dof / 2.0 + 1.0);
    // finally compute gamma_prm. See Sampling-based algorithms for optimal motion planning by Karaman and Frazzoli
    _gamma_prm = 2.0 * pow((1.0 + 1.0 / dof) * mu / xi, 1.0 / dof);
    // now densify the roadmap
    densify(batch_size);
}

Roadmap::~Roadmap() = default;

void Roadmap::densify()
{
    densify(_batch_size);
}

void Roadmap::densify(unsigned int batch_size)
{
    assert(batch_size > 0);
    unsigned int dof = _robot->GetActiveDOF();
    double* new_samples = halton::halton_sequence(_halton_seq_id, _halton_seq_id + batch_size - 1, dof);
    _halton_seq_id += batch_size;
    double* config_pointer = new_samples;
    Config config(dof);
    for (unsigned int id = 0; id < batch_size; ++id) {
        // TODO add random noise to config
        config.assign(config_pointer, config_pointer + dof);
        scaleToLimits(config);
        NodePtr new_node = addNode(config);
        assert(new_node->config.size() == dof);
        config_pointer += dof;
    }
    delete[] new_samples;
    _densification_gen += 1;
}

Roadmap::NodePtr Roadmap::addGoalNode(const MultiGraspMP::Goal& goal)
{
    NodePtr new_node = addNode(goal.config);
    new_node->is_goal = true;
    new_node->goal_id = goal.id;
    return new_node;
}

Roadmap::NodePtr Roadmap::addNode(const Config& config)
{
    NodePtr new_node = std::shared_ptr<Node>(new Node(_node_id_counter++, config));
    _nn.add(new_node);
    return new_node;
}

bool Roadmap::checkNode(NodePtr node)
{
    if (!node->initialized) {
        // check validity
        if (not _validity_checker->isValid(node->config)) {
            // in case of the node being invalid, remove it
            deleteNode(node);
            return false;
        }
    }
    // update the node's adjacency
    if (node->densification_gen != _densification_gen) {
        double dof = _robot->GetActiveDOF();
        // radius computed according to RRT*/PRM* paper
        double r = _gamma_prm * pow(log(_nn.size()) / _nn.size(), 1.0 / dof);
        std::vector<NodePtr> neighbors;
        _nn.nearestR(node, r, neighbors);
        // add new edges, keep old ones
        for (auto& neigh : neighbors) {
            // check whether we already have this edge
            auto edge_iter = node->edges.find(node->uid);
            if (edge_iter != node->edges.end() and neigh != node) {
                // if not, create a new edge
                auto new_edge = std::make_shared<Edge>(node, neigh);
                node->edges.insert(std::make_pair(neigh->uid, new_edge));
                neigh->edges.insert(std::make_pair(node->uid, new_edge));
                new_edge->base_cost = _cost_computer->lowerBound(node->config, neigh->config);
            }
        }
        node->densification_gen = _densification_gen;
    }
    node->initialized = true;
    return true;
}

bool Roadmap::checkEdge(EdgePtr edge)
{
    if (edge->base_evaluated) {
        return true;
    }
    NodePtr node_a = edge->node_a.lock();
    NodePtr node_b = edge->node_b.lock();
    assert(node_a);
    assert(node_b);
    edge->base_cost = _cost_computer->cost(node_a->config, node_b->config);
    edge->base_evaluated = true;
    if (std::isinf(edge->base_cost)) {
        // edge is invalid, so let's remove it
        deleteEdge(edge);
        return false;
    }
    return true;
}

bool Roadmap::computeCost(EdgePtr edge, unsigned int grasp_id)
{
    auto iter = edge->conditional_costs.find(grasp_id);
    double cost;
    if (iter == edge->conditional_costs.end()) {
        NodePtr node_a = edge->node_a.lock();
        NodePtr node_b = edge->node_b.lock();
        cost = _cost_computer->cost(node_a->config, node_b->config, grasp_id);
        edge->conditional_costs.insert(std::make_pair(grasp_id, cost));
    } else {
        cost = iter->second;
    }
    return not std::isinf(cost);
}

void Roadmap::scaleToLimits(Config& config) const
{
    std::vector<OpenRAVE::dReal> lower, upper;
    _robot->GetActiveDOFLimits(lower, upper);
    assert(config.size() == lower.size() and config.size() == upper.size());
    for (unsigned int i = 0; i < config.size(); ++i) {
        config[i] = config[i] * (upper[i] - lower[i]) + lower[i];
    }
}

void Roadmap::deleteNode(NodePtr node)
{
    _nn.remove(node);
    // remove all edges pointing to it
    for (auto out_info : node->edges) {
        // unsigned int edge_target_id = out_info.first;
        EdgePtr edge = out_info.second;
        deleteEdge(edge);
        // NodePtr adjacent_node = edge->node_a.lock();
        // if (adjacent_node->uid != edge_target_id) {
        //     assert(adjacent_node == node);
        //     adjacent_node = edge->node_b.lock();
        // }
        // // check whether the adjacent node has an edge pointing to this node
        // auto edge_iter = adjacent_node->edges.find(node->uid);
        // if (edge_iter != adjacent_node->edges.end()) {
        //     // if yes, remove the edge
        //     adjacent_node->edges.erase(edge_iter);
        // }
    }
    assert(node->edges.empty());
    node.reset();
}

void Roadmap::deleteEdge(EdgePtr edge)
{
    NodePtr node_a = edge->node_a.lock();
    NodePtr node_b = edge->node_b.lock();
    assert(node_a);
    assert(node_b);
    { // remove it from node a
        auto edge_iter = node_a->edges.find(node_b->uid);
        assert(edge_iter != node_a->edges.end());
        node_a->edges.erase(edge_iter);
    }
    { // remove it from node b
        auto edge_iter = node_b->edges.find(node_a->uid);
        assert(edge_iter != node_b->edges.end());
        node_b->edges.erase(edge_iter);
    }
    edge.reset();
}