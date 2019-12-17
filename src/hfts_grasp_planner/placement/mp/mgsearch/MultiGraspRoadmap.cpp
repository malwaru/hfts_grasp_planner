#include <Eigen/Core>
#include <hfts_grasp_planner/external/halton/halton.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>

using namespace placement::mp::mgsearch;

StateSpace::~StateSpace() = default;

EdgeCostComputer::~EdgeCostComputer() = default;

IntegralEdgeCostComputer::IntegralEdgeCostComputer(StateSpacePtr state_space, double step_size)
    : _state_space(state_space)
    , _step_size(step_size)
{
}

IntegralEdgeCostComputer::~IntegralEdgeCostComputer() = default;

double IntegralEdgeCostComputer::integrateCosts(const Config& a, const Config& b,
    const std::function<double(const Config&)>& cost_fn) const
{
    assert(a.size() == b.size());
    Eigen::Map<const Eigen::VectorXd> avec(a.data(), a.size());
    Eigen::Map<const Eigen::VectorXd> bvec(b.data(), b.size());
    Eigen::VectorXd delta = bvec - avec;
    Config q(delta.size());
    Eigen::Map<Eigen::VectorXd> qvec(q.data(), q.size());
    double norm = delta.norm();
    if (norm == 0.0) {
        return 0.0;
    }
    // iterate over path
    double integral_cost = 0.0;
    unsigned int num_steps = std::ceil(norm / _step_size);
    for (size_t t = 0; t < num_steps; ++t) {
        qvec = std::min(t * _step_size / norm, 1.0) * delta + avec;
        double dc = cost_fn(q); // qvec is a view on q's data
        if (std::isinf(dc))
            return INFINITY;
        integral_cost += dc * _step_size;
    }
    return integral_cost;
}

double IntegralEdgeCostComputer::lowerBound(const Config& a, const Config& b) const
{
    return _state_space->distance(a, b);
}

double IntegralEdgeCostComputer::cost(const Config& a, const Config& b) const
{
    auto fn = std::bind(&StateSpace::cost, _state_space, std::placeholders::_1);
    return integrateCosts(a, b, fn);
}

double IntegralEdgeCostComputer::cost(const Config& a, const Config& b, unsigned int grasp_id) const
{
    auto fn = std::bind(&StateSpace::conditional_cost, _state_space, std::placeholders::_1, grasp_id);
    return integrateCosts(a, b, fn);
}

CostToGoHeuristic::~CostToGoHeuristic() = default;

// double distanceFn(const Roadmap::NodePtr& a, const Roadmap::NodePtr& b)
// {
//     return cSpaceDistance(a->config, b->config);
// }

Roadmap::Edge::Edge(Roadmap::NodePtr a, Roadmap::NodePtr b, double bc)
    : base_cost(bc)
    , base_evaluated(false)
    , node_a(a)
    , node_b(b)
{
}

Roadmap::NodePtr Roadmap::Edge::getNeighbor(NodePtr n) const
{
    auto a = node_a.lock();
    assert(a);
    if (a->uid != n->uid)
        return a;
    auto b = node_b.lock();
    assert(b);
    assert(b->uid != n->uid);
    return b;
}

double Roadmap::Edge::getBestKnownCost(unsigned int gid) const
{
    auto iter = conditional_costs.find(gid);
    if (iter != conditional_costs.end()) {
        return iter->second;
    }
    return base_cost;
}

Roadmap::Roadmap(StateSpacePtr state_space, EdgeCostComputerPtr cost_computer, unsigned int batch_size)
    : _state_space(state_space)
    , _si(state_space->getSpaceInformation())
    , _cost_computer(cost_computer)
    , _batch_size(batch_size)
    , _node_id_counter(0)
    , _halton_seq_id(0)
    , _densification_gen(0)
{
    assert(_si.lower.size() == _si.upper.size() and _si.lower.size() == _si.dimension);
    // _nn.setDistanceFunction(distanceFn);
    _nn.setDistanceFunction([this](const Roadmap::NodePtr& a, const Roadmap::NodePtr& b) { return _state_space->distance(a->config, b->config); });
    // compute gamma_prm - a constant used to compute the radius for adjacency
    // we need the measure of X_free for this, we approximate it by the measure of X
    double mu = 1.0;
    for (unsigned int i = 0; i < _si.dimension; ++i) {
        mu *= _si.upper[i] - _si.lower[i];
    }
    // xi is the measure of a dof-dimensional unit ball
    double xi = pow(M_PI, _si.dimension / 2.0) / tgamma(_si.dimension / 2.0 + 1.0);
    // finally compute gamma_prm. See Sampling-based algorithms for optimal motion planning by Karaman and Frazzoli
    _gamma_prm = 2.0 * pow((1.0 + 1.0 / _si.dimension) * mu / xi, 1.0 / _si.dimension);
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
    double* new_samples = halton::halton_sequence(_halton_seq_id, _halton_seq_id + batch_size - 1, _si.dimension);
    _halton_seq_id += batch_size;
    double* config_pointer = new_samples;
    Config config(_si.dimension);
    for (unsigned int id = 0; id < batch_size; ++id) {
        // TODO add random noise to config
        config.assign(config_pointer, config_pointer + _si.dimension);
        scaleToLimits(config);
        NodePtr new_node = addNode(config).lock();
        assert(new_node->config.size() == _si.dimension);
        config_pointer += _si.dimension;
    }
    delete[] new_samples;
    _densification_gen += 1;
}

Roadmap::NodePtr Roadmap::getNode(unsigned int node_id) const
{
    auto iter = _nodes.find(node_id);
    if (iter == _nodes.end()) {
        return nullptr;
    }
    auto ptr = iter->second.lock();
    assert(ptr != nullptr);
    return ptr;
}

Roadmap::NodeWeakPtr Roadmap::addGoalNode(const MultiGraspMP::Goal& goal)
{
    NodePtr new_node = addNode(goal.config).lock();
    new_node->is_goal = true;
    new_node->goal_id = goal.id;
    return new_node;
}

Roadmap::NodeWeakPtr Roadmap::addNode(const Config& config)
{
    NodePtr new_node = std::shared_ptr<Node>(new Node(_node_id_counter++, config));
    _nn.add(new_node);
    _nodes.insert(std::make_pair(new_node->uid, new_node));
    return new_node;
}

void Roadmap::updateAdjacency(NodePtr node)
{
    // update the node's adjacency
    if (node->densification_gen != _densification_gen) {
        // radius computed according to RRT*/PRM* paper
        double r = _gamma_prm * pow(log(_nn.size()) / _nn.size(), 1.0 / _si.dimension);
        std::vector<NodePtr> neighbors;
        _nn.nearestR(node, r, neighbors);
        // add new edges, keep old ones
        for (auto& neigh : neighbors) {
            // check whether we already have this edge
            auto edge_iter = node->edges.find(neigh->uid);
            if (edge_iter == node->edges.end() and neigh != node) {
                // if not, create a new edge
                double bc = _cost_computer->lowerBound(node->config, neigh->config);
                auto new_edge = std::make_shared<Edge>(node, neigh, bc);
                node->edges.insert(std::make_pair(neigh->uid, new_edge));
                neigh->edges.insert(std::make_pair(node->uid, new_edge));
            }
        }
        node->densification_gen = _densification_gen;
    }
}

bool Roadmap::isValid(NodeWeakPtr inode)
{
    if (inode.expired())
        return false;
    auto node = inode.lock();
    if (!node->initialized) {
        // check validity
        if (not _state_space->isValid(node->config)) {
            // in case of the node being invalid, remove it
            deleteNode(node);
            return false;
        }
    }
    node->initialized = true;
    return true;
}

bool Roadmap::isValid(NodeWeakPtr wnode, unsigned int grasp_id)
{
    bool base_valid = isValid(wnode);
    if (base_valid) {
        // check validity for the given grasp
        NodePtr node = wnode.lock();
        auto iter = node->conditional_validity.find(grasp_id);
        if (iter == node->conditional_validity.end()) {
            bool valid = _state_space->isValid(node->config, grasp_id, true);
            node->conditional_validity[grasp_id] = valid;
            return valid;
        } else {
            return iter->second;
        }
    }
    return false;
}

std::pair<bool, double> Roadmap::computeCost(EdgePtr edge)
{
    if (edge->base_evaluated) {
        return { true, edge->base_cost };
    }
    // we have to compute base cost
    NodePtr node_a = edge->node_a.lock();
    NodePtr node_b = edge->node_b.lock();
    assert(node_a);
    assert(node_b);
    edge->base_cost = _cost_computer->cost(node_a->config, node_b->config);
    edge->base_evaluated = true;
    if (std::isinf(edge->base_cost)) {
        // edge is invalid, so let's remove it
        deleteEdge(edge);
        return { false, edge->base_cost };
    }
    return { true, edge->base_cost };
}

std::pair<bool, double> Roadmap::computeCost(EdgeWeakPtr weak_edge)
{
    if (weak_edge.expired())
        return { false, INFINITY };
    auto edge = weak_edge.lock();
    return computeCost(edge);
}

std::pair<bool, double> Roadmap::computeCost(EdgePtr edge, unsigned int grasp_id)
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
    return { not std::isinf(cost), cost };
}

void Roadmap::scaleToLimits(Config& config) const
{
    assert(config.size() == _si.dimension);
    for (unsigned int i = 0; i < _si.dimension; ++i) {
        config[i] = config[i] * (_si.upper[i] - _si.lower[i]) + _si.lower[i];
    }
}

void Roadmap::deleteNode(NodePtr node)
{
    _nn.remove(node);
    auto iter = _nodes.find(node->uid);
    assert(iter != _nodes.end());
    _nodes.erase(iter);
    // remove all edges pointing to it
    for (auto iter = node->edges.begin(); iter != node->edges.end();) {
        // unsigned int edge_target_id = out_info.first;
        EdgePtr edge = iter->second;
        // delete edge in other node
        {
            NodePtr other_node = edge->getNeighbor(node);
            auto oiter = other_node->edges.find(node->uid);
            assert(oiter != other_node->edges.end());
            other_node->edges.erase(oiter);
        }
        // delete edge in this node
        iter = node->edges.erase(iter);
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

/************************************** MGGoalDistance **************************************/

MGGoalDistance::MGGoalDistance(const std::vector<MultiGraspMP::Goal>& goals,
    const std::function<double(const Config&, const Config&)>& path_cost, double lambda)
{
    double max_q, min_q = -INFINITY, INFINITY;
    // first compute min and max quality
    for (const MultiGraspMP::Goal& goal : goals) {
        // add goal to all goals
        max_q = std::max(max_q, goal.quality);
        min_q = std::min(min_q, goal.quality);
    }
    _quality_normalizer = (max_q - min_q);
    _goal_distance.scaled_lambda = lambda / _quality_normalizer;
    _goal_distance.path_cost = path_cost;
    _max_quality = max_q;
    // now add the goals to nearest neighbor data structures
    auto dist_fun = std::bind(&MGGoalDistance::GoalDistanceFn::distance, &_goal_distance, std::placeholders::_1, std::placeholders::_2);
    _all_goals.setDistanceFunction(dist_fun);
    for (auto& goal : goals) {
        _all_goals.add(goal);
        // add it to goals with the same grasp
        auto iter = _goals.find(goal.grasp_id);
        if (iter == _goals.end()) {
            // add new gnat for this grasp
            auto new_gnat = std::make_shared<::ompl::NearestNeighborsGNAT<MultiGraspMP::Goal>>();
            new_gnat->setDistanceFunction(dist_fun);
            new_gnat->add(goal);
            _goals.insert(std::make_pair(goal.grasp_id, new_gnat));
        } else {
            iter->second->add(goal);
        }
    }
}

MGGoalDistance::~MGGoalDistance() = default;

double MGGoalDistance::costToGo(const Config& a) const
{
    if (_all_goals.size() == 0) {
        throw std::logic_error("[MGGoalDistance::costToGo] No goals known. Can not compute cost to go.");
    }
    MultiGraspMP::Goal dummy_goal;
    dummy_goal.config = a;
    dummy_goal.quality = _max_quality;
    const auto& nn = _all_goals.nearest(dummy_goal);
    return _goal_distance.distance_const(nn, dummy_goal);
}

double MGGoalDistance::costToGo(const Config& a, unsigned int grasp_id) const
{
    auto iter = _goals.find(grasp_id);
    if (iter == _goals.end()) {
        throw std::logic_error("[MGGoalDistance::costToGo] Could not find GNAT for the given grasp " + std::to_string(grasp_id));
    }
    if (iter->second->size() == 0) {
        throw std::logic_error("[MGGoalDistance::costToGo] No goal known for the given grasp " + std::to_string(grasp_id));
    }
    MultiGraspMP::Goal dummy_goal;
    dummy_goal.config = a;
    dummy_goal.quality = _max_quality;
    const auto& nn = iter->second->nearest(dummy_goal);
    return _goal_distance.distance_const(nn, dummy_goal);
}

double MGGoalDistance::goalCost(double quality) const
{
    return _quality_normalizer * (_max_quality - quality);
}