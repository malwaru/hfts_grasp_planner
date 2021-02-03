#include <Eigen/Core>
#include <hfts_grasp_planner/external/halton/halton.hpp>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>

#ifdef NDEBUG
#define FLUSH_END "\n"
#else
#define FLUSH_END std::endl
#endif

using namespace placement::mp::mgsearch;

StateSpace::~StateSpace() = default;

EdgeCostComputer::~EdgeCostComputer() = default;

IntegralEdgeCostComputer::IntegralEdgeCostComputer(StateSpacePtr state_space, double step_size)
  : _state_space(state_space), _step_size(step_size > 0.0 ? step_size : state_space->getStepSize())
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
  if (norm == 0.0)
  {
    return 0.0;
  }
  // iterate over path
  double integral_cost = 0.0;
  unsigned int num_intervals = std::ceil(norm / _step_size);
  // to ensure this function is symmetric in a and b, use an even step size
  double interval_size = norm / num_intervals;
  delta *= interval_size / norm;
  qvec = avec + 0.5 * delta;
  for (size_t t = 0; t < num_intervals and not std::isinf(integral_cost); ++t)
  {
    // qvec = (t + 0.5) * step_size * delta + avec;
    // double dc = cost_fn(q);  // qvec is a view on q's data
    // if (std::isinf(dc))
    //   return INFINITY;
    integral_cost += cost_fn(q) * interval_size;
    qvec += delta;  // qvec is a view on q's data
  }
  return integral_cost;
}

double IntegralEdgeCostComputer::lowerBound(const Config& a, const Config& b) const
{
#ifdef ENABLE_ROADMAP_PROFILING
  utils::ScopedProfiler profiler("IntegralEdgeCostComputer::lowerBound");
#endif
  return _state_space->distance(a, b);
}

double IntegralEdgeCostComputer::cost(const Config& a, const Config& b) const
{
  // auto fn = std::bind(&StateSpace::cost, _state_space, std::placeholders::_1);
  return integrateCosts(a, b, [this](const ::placement::mp::Config& c) { return this->_state_space->cost(c); });
}

double IntegralEdgeCostComputer::cost(const Config& a, const Config& b, unsigned int grasp_id) const
{
  return integrateCosts(
      a, b, [this, grasp_id](const ::placement::mp::Config& c) { return this->_state_space->cost(c, grasp_id); });
}

// double distanceFn(const Roadmap::NodePtr& a, const Roadmap::NodePtr& b)
// {
//     return cSpaceDistance(a->config, b->config);
// }

Roadmap::Edge::Edge(Roadmap::NodePtr a, Roadmap::NodePtr b, double bc)
  : base_cost(bc), base_evaluated(false), node_a(a), node_b(b)
{
}

Roadmap::NodePtr Roadmap::Edge::getNeighbor(NodePtr n) const
{
  auto a = node_a.lock();
  if (a != nullptr and a->uid != n->uid)
    return a;
  auto b = node_b.lock();
  // assert(b == nullptr or b->uid != n->uid);
  if (b != nullptr and b->uid != n->uid)
    return b;
  return nullptr;
}

double Roadmap::Edge::getBestKnownCost(unsigned int gid) const
{
  auto iter = conditional_costs.find(gid);
  if (iter != conditional_costs.end())
  {
    return iter->second;
  }
  return base_cost;
}

Roadmap::Logger::Logger() = default;

Roadmap::Logger::~Logger()
{
  _roadmap_fs.close();
  _log_fs.close();
}

void Roadmap::Logger::setLogPath(const std::string& roadmap_file, const std::string& log_file)
{
  if (_roadmap_fs.is_open())
  {
    _roadmap_fs.close();
  }
  if (_log_fs.is_open())
  {
    _log_fs.close();
  }
  if (!roadmap_file.empty())
  {
    std::experimental::filesystem::path roadmap_path(roadmap_file);
    roadmap_path.remove_filename();
    if (!std::experimental::filesystem::exists(roadmap_path))
    {
      std::experimental::filesystem::create_directories(roadmap_path);
    }
    _roadmap_fs.open(roadmap_file, std::ios::out);
  }
  if (!log_file.empty())
  {
    std::experimental::filesystem::path log_path(log_file);
    log_path.remove_filename();
    if (!std::experimental::filesystem::exists(log_path))
    {
      std::experimental::filesystem::create_directories(log_path);
    }
    _log_fs.open(log_file, std::ios::out);
  }
}

void Roadmap::Logger::newNode(NodePtr node)
{
  if (_roadmap_fs.is_open())
  {
    // a node is stored as a single line: id, dim, x1, x2, ..., xdim\n
    _roadmap_fs << node->uid << ", ";
    _roadmap_fs << node->config.size();
    for (auto ci : node->config)
    {
      _roadmap_fs << ", " << ci;
    }
    _roadmap_fs << FLUSH_END;
  }
}

void Roadmap::Logger::nodeValidityChecked(NodePtr node, bool bval)
{
  if (_log_fs.is_open())
  {
    // log that the validity of node has been checked. format: VALBASE, id, bval\n
    _log_fs << "VAL_BASE, " << node->uid << ", " << bval << FLUSH_END;
  }
}

void Roadmap::Logger::nodeValidityChecked(NodePtr node, unsigned int grasp_id, bool bval)
{
  if (_log_fs.is_open())
  {
    // log that the validity of node has been checked. format: VALGRASP, id, grasp_id, bval\n
    _log_fs << "VAL_GRASP, " << node->uid << ", " << grasp_id << ", " << bval << FLUSH_END;
  }
}

void Roadmap::Logger::edgeCostChecked(NodePtr a, NodePtr b, double cost)
{
  if (_log_fs.is_open())
  {
    // log that a cost of the edge connecting a and b has been computed
    // format: EDGE_COST, aid, bid, cost\n
    _log_fs << "EDGE_COST, " << a->uid << ", " << b->uid << ", " << cost << FLUSH_END;
  }
}

void Roadmap::Logger::edgeCostChecked(NodePtr a, NodePtr b, unsigned int grasp_id, double cost)
{
  if (_log_fs.is_open())
  {
    // log that a cost of the edge connecting a and b has been computed
    // format: EDGE_COST_GRASP, aid, bid, grasp_id, cost\n
    _log_fs << "EDGE_COST_GRASP, " << a->uid << ", " << b->uid << ", " << grasp_id << ", " << cost << FLUSH_END;
  }
}

void Roadmap::Logger::logCustomEvent(const std::string& msg)
{
  if (_log_fs.is_open())
  {
    _log_fs << msg << FLUSH_END;
  }
}

Roadmap::Roadmap(StateSpacePtr state_space, EdgeCostComputerPtr cost_computer, unsigned int batch_size,
                 const std::string& log_roadmap_path, const std::string& log_path)
  : _state_space(state_space)
  , _si(state_space->getSpaceInformation())
  , _cost_computer(cost_computer)
  , _batch_size(batch_size)
  , _node_id_counter(0)
  , _halton_seq_id(0)
  , _densification_gen(0)
{
  assert(_si.lower.size() == _si.upper.size() and _si.lower.size() == _si.dimension);
  _logger.setLogPath(log_roadmap_path, log_path);
  // _nn.setDistanceFunction(distanceFn);
  _nn.setDistanceFunction([this](const Roadmap::NodePtr& a, const Roadmap::NodePtr& b) {
    return _state_space->distance(a->config, b->config);
  });
  // compute gamma_prm - a constant used to compute the radius for adjacency
  // we need the measure of X_free for this, we approximate it by the measure of X
  double mu = 1.0;
  for (unsigned int i = 0; i < _si.dimension; ++i)
  {
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
#ifdef ENABLE_ROADMAP_PROFILING
  utils::ScopedProfiler profiler("Roadmap::densify");
#endif
  assert(batch_size > 0);
  double* new_samples = halton::halton_sequence(_halton_seq_id, _halton_seq_id + batch_size - 1, _si.dimension);
  _halton_seq_id += batch_size;
  double* config_pointer = new_samples;
  Config config(_si.dimension);
  for (unsigned int id = 0; id < batch_size; ++id)
  {
    // TODO add random noise to config
    config.assign(config_pointer, config_pointer + _si.dimension);
    scaleToLimits(config);
    NodePtr new_node = addNode(config).lock();
    assert(new_node->config.size() == _si.dimension);
    config_pointer += _si.dimension;
  }
  delete[] new_samples;
  _densification_gen += 1;
  // update adjacency radius according to RRT*/PRM* paper
  _adjacency_radius = _gamma_prm * pow(log(_halton_seq_id) / _halton_seq_id, 1.0 / _si.dimension);
}

void Roadmap::setLogging(const std::string& roadmap_path, const std::string& log_path)
{
  _logger.setLogPath(roadmap_path, log_path);
}

void Roadmap::logCustomEvent(const std::string& msg)
{
  _logger.logCustomEvent(msg);
}

Roadmap::NodePtr Roadmap::getNode(unsigned int node_id) const
{
  auto iter = _nodes.find(node_id);
  if (iter == _nodes.end())
  {
    return nullptr;
  }
  auto ptr = iter->second.lock();
  assert(ptr != nullptr);
  return ptr;
}

Roadmap::NodeWeakPtr Roadmap::addNode(const Config& config)
{
  NodePtr new_node = std::shared_ptr<Node>(new Node(_node_id_counter++, config));
  _nn.add(new_node);
  _nodes.insert(std::make_pair(new_node->uid, new_node));
  _logger.newNode(new_node);
  return new_node;
}

void Roadmap::updateAdjacency(NodePtr node)
{
#ifdef ENABLE_ROADMAP_PROFILING
  utils::ScopedProfiler profiler("Roadmap::updateAdjacency");
#endif
  // update the node's adjacency
  if (node->densification_gen != _densification_gen)
  {
#ifdef ENABLE_ROADMAP_PROFILING
    utils::ScopedProfiler prof2("Roadmap::updateAdjacency::addNeighbors");
#endif
    std::vector<NodePtr> neighbors;
    _nn.nearestR(node, _adjacency_radius, neighbors);
    // add new edges, keep old ones
    for (auto& neigh : neighbors)
    {
      // check whether we already have this edge
      auto edge_iter = node->edges.find(neigh->uid);
      if (edge_iter == node->edges.end() and neigh != node)
      {
        // if not, create a new edge
        double bc = _cost_computer->lowerBound(node->config, neigh->config);
        auto new_edge = std::make_shared<Edge>(node, neigh, bc);
        node->edges.insert(std::make_pair(neigh->uid, new_edge));
        neigh->edges.insert(std::make_pair(node->uid, new_edge));
        // set conditional validity based on node's validity
        for (auto [gid, valid] : node->conditional_validity)
        {
          if (!valid)
            new_edge->conditional_costs[gid] = std::numeric_limits<double>::infinity();
        }
        // set conditional validity based on neigh's validity
        for (auto [gid, valid] : neigh->conditional_validity)
        {
          if (!valid)
            new_edge->conditional_costs[gid] = std::numeric_limits<double>::infinity();
        }
      }
    }
    node->densification_gen = _densification_gen;
  }
  // clean up edges that are no longer needed because they are invalid
  for (unsigned int uid : node->edges_to_delete)
  {
    node->edges.erase(uid);
  }
  node->edges_to_delete.clear();
}

bool Roadmap::isValid(NodeWeakPtr inode)
{
#ifdef ENABLE_ROADMAP_PROFILING
  utils::ScopedProfiler profiler("Roadmap::isValid");
#endif
  if (inode.expired())
    return false;
  auto node = inode.lock();
  if (!node->initialized)
  {
    bool valid = _state_space->isValid(node->config);
    _logger.nodeValidityChecked(node, valid);
    // check validity
    if (not valid)
    {
      // in case of the node being invalid, remove it
      deleteNode(node);
      return false;
    }
  }
  node->initialized = true;
  return true;
}

bool Roadmap::isValid(unsigned int node_id)
{
  auto iter = _nodes.find(node_id);
  if (iter == _nodes.end())
    return false;
  return isValid(iter->second);
}

bool Roadmap::isValid(NodeWeakPtr wnode, unsigned int grasp_id)
{
  bool base_valid = isValid(wnode);
#ifdef ENABLE_ROADMAP_PROFILING
  utils::ScopedProfiler profiler("Roadmap::isValidWithGrasp");
#endif
  if (base_valid)
  {
    // check validity for the given grasp
    NodePtr node = wnode.lock();
    auto iter = node->conditional_validity.find(grasp_id);
    if (iter == node->conditional_validity.end())
    {
      bool valid = _state_space->isValid(node->config, grasp_id, true);
      node->conditional_validity[grasp_id] = valid;
      if (not valid)
      {  // also update edges
        for (auto edge_pair : node->edges)
        {
          edge_pair.second->conditional_costs[grasp_id] = std::numeric_limits<double>::infinity();
        }
      }
      _logger.nodeValidityChecked(node, grasp_id, valid);
      return valid;
    }
    else
    {
      return iter->second;
    }
  }
  return false;
}

bool Roadmap::isValid(unsigned int node_id, unsigned int grasp_id)
{
  auto iter = _nodes.find(node_id);
  if (iter == _nodes.end())
    return false;
  return isValid(iter->second, grasp_id);
}

std::pair<bool, double> Roadmap::computeCost(EdgePtr edge)
{
#ifdef ENABLE_ROADMAP_PROFILING
  utils::ScopedProfiler profiler("Roadmap::computeCost");
#endif
  if (edge->base_evaluated)
  {
    return {!std::isinf(edge->base_cost), edge->base_cost};
  }
  // we have to compute base cost
  NodePtr node_a = edge->node_a.lock();
  NodePtr node_b = edge->node_b.lock();
  assert(node_a);
  assert(node_b);
  edge->base_cost = _cost_computer->cost(node_a->config, node_b->config);
  edge->base_evaluated = true;
  bool invalid = std::isinf(edge->base_cost);
  if (invalid)
  {
    node_a->edges_to_delete.push_back(node_b->uid);
    node_b->edges_to_delete.push_back(node_a->uid);
  }
  _logger.edgeCostChecked(node_a, node_b, edge->base_cost);
  return {!invalid, edge->base_cost};
}

std::pair<bool, double> Roadmap::computeCost(EdgeWeakPtr weak_edge)
{
  if (weak_edge.expired())
    return {false, INFINITY};
  auto edge = weak_edge.lock();
  return computeCost(edge);
}

std::pair<bool, double> Roadmap::computeCost(EdgePtr edge, unsigned int grasp_id)
{
#ifdef ENABLE_ROADMAP_PROFILING
  utils::ScopedProfiler profiler("Roadmap::computeCostWithGrasp");
#endif
  if (edge->base_evaluated and std::isinf(edge->base_cost))
  {
    return {false, edge->base_cost};
  }
  auto iter = edge->conditional_costs.find(grasp_id);
  double cost;
  if (iter == edge->conditional_costs.end())
  {
    NodePtr node_a = edge->node_a.lock();
    assert(node_a);
    NodePtr node_b = edge->node_b.lock();
    assert(node_b);
    cost = _cost_computer->cost(node_a->config, node_b->config, grasp_id);
    _logger.edgeCostChecked(node_a, node_b, grasp_id, cost);
    edge->conditional_costs.insert(std::make_pair(grasp_id, cost));
  }
  else
  {
    cost = iter->second;
  }
  return {not std::isinf(cost), cost};
}

void Roadmap::scaleToLimits(Config& config) const
{
  assert(config.size() == _si.dimension);
  for (unsigned int i = 0; i < _si.dimension; ++i)
  {
    config[i] = config[i] * (_si.upper[i] - _si.lower[i]) + _si.lower[i];
  }
}

void Roadmap::deleteNode(NodePtr node)
{
  _nn.remove(node);
  auto iter = _nodes.find(node->uid);
  assert(iter != _nodes.end());
  _nodes.erase(iter);
  // set all its edges to infinite cost;
  // if the neighboring node still exists, its edge map will be eventually updated in updateAdjacency(..)
  for (auto iter = node->edges.begin(); iter != node->edges.end(); ++iter)
  {
    // unsigned int edge_target_id = out_info.first;
    EdgePtr edge = iter->second;
    edge->base_evaluated = true;
    edge->base_cost = std::numeric_limits<double>::infinity();
    // inform other node that it should delete this edge in updateAdjacency
    auto other_node = edge->getNeighbor(node);
    if (other_node)
      other_node->edges_to_delete.push_back(node->uid);
  }
  node.reset();
}

/************************************** MultiGraspGoalSet **************************************/
// GoalIterator
MultiGraspGoalSet::GoalIterator::GoalIterator(std::unordered_map<unsigned int, MultiGraspMP::Goal>::const_iterator iter)
  : _iter(iter)
{
}

MultiGraspGoalSet::GoalIterator& MultiGraspGoalSet::GoalIterator::operator++()
{
  _iter++;
  return (*this);
}

bool MultiGraspGoalSet::GoalIterator::operator==(const MultiGraspGoalSet::GoalIterator& other) const
{
  return other._iter == _iter;
}

bool MultiGraspGoalSet::GoalIterator::operator!=(const MultiGraspGoalSet::GoalIterator& other) const
{
  return other._iter != _iter;
}

const placement::mp::MultiGraspMP::Goal& MultiGraspGoalSet::GoalIterator::operator*()
{
  return _iter->second;
}

const placement::mp::MultiGraspMP::Goal* MultiGraspGoalSet::GoalIterator::operator->()
{
  return &(_iter->second);
}

// actual set
MultiGraspGoalSet::MultiGraspGoalSet(RoadmapPtr roadmap)
  : _roadmap(roadmap), _quality_range{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}
{
}

MultiGraspGoalSet::~MultiGraspGoalSet() = default;

bool MultiGraspGoalSet::addGoal(const MultiGraspMP::Goal& goal)
{
  auto new_node = _roadmap->addNode(goal.config).lock();
  assert(new_node);
  if (_roadmap->isValid(new_node, goal.grasp_id))
  {
    addGoalToMembers(goal, new_node->uid);
    return true;
  }
  return false;
}

placement::mp::MultiGraspMP::Goal MultiGraspGoalSet::getGoal(unsigned int gid) const
{
  auto goal_iter = _goals.find(gid);
  if (goal_iter == _goals.end())
  {
    throw std::logic_error("There is no goal with id " + std::to_string(gid));
  }
  return goal_iter->second;
}

void MultiGraspGoalSet::removeGoal(unsigned int gid)
{
  auto goal_iter = _goals.find(gid);
  if (goal_iter != _goals.end())
  {
    // remove goal from goal_id_to_roadmap id map
    auto gid2rid_iter = _goal_id_to_roadmap_id.find(gid);
    assert(gid2rid_iter != _goal_id_to_roadmap_id.end());
    unsigned int rid = gid2rid_iter->second;
    _goal_id_to_roadmap_id.erase(gid2rid_iter);
    // remove inverse mapping
    auto rid2gid_iter = _roadmap_id_to_goal_id.find(rid);
    assert(rid2gid_iter != _roadmap_id_to_goal_id.end());
    assert(rid2gid_iter->second == gid);
    _roadmap_id_to_goal_id.erase(rid2gid_iter);
    // remove it from grasp_id_to_goal_id
    _grasp_id_to_goal_ids.at(goal_iter->second.grasp_id).erase(gid);
    // fix quality range
    if (_goals.empty())
    {
      _quality_range = {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    }
    else if (goal_iter->second.quality == _quality_range.first || goal_iter->second.quality == _quality_range.second)
    {
      for (auto remaining_goal : _goals)
      {
        _quality_range.first = std::min(_quality_range.first, remaining_goal.second.quality);
        _quality_range.second = std::max(_quality_range.second, remaining_goal.second.quality);
      }
    }
    // finally erase goal
    _goals.erase(goal_iter);
  }
}

void MultiGraspGoalSet::removeGoals(const std::vector<unsigned int>& goal_ids)
{
  for (unsigned int gid : goal_ids)
  {
    removeGoal(gid);
  }
}

void MultiGraspGoalSet::removeGoals(const GoalIterator& begin, const GoalIterator& end)
{
  GoalIterator iter(begin);
  for (; iter != end; ++iter)
  {
    removeGoal(iter->id);
  }
}

bool MultiGraspGoalSet::isGoal(Roadmap::NodePtr node, unsigned int grasp_id) const
{
  return isGoal(node->uid, grasp_id);
}

bool MultiGraspGoalSet::isGoal(unsigned int node_id, unsigned int grasp_id) const
{
  // check whether this roadmap node is affiliated with a goal
  unsigned int goal_id;
  {
    auto iter = _roadmap_id_to_goal_id.find(node_id);
    if (iter == _roadmap_id_to_goal_id.end())
      return false;
    goal_id = iter->second;
  }
  // check whether the goal is for the given grasp
  return grasp_id == _goals.at(goal_id).grasp_id;
}

bool MultiGraspGoalSet::canBeGoal(unsigned int node_id) const
{
  return _roadmap_id_to_goal_id.find(node_id) != _roadmap_id_to_goal_id.end();
}

unsigned int MultiGraspGoalSet::getRoadmapId(unsigned int goal_id) const
{
  return _goal_id_to_roadmap_id.at(goal_id);
}

std::pair<unsigned int, bool> MultiGraspGoalSet::getGoalId(unsigned int node_id, unsigned int grasp_id)
{
  auto iter = _roadmap_id_to_goal_id.find(node_id);
  if (iter == _roadmap_id_to_goal_id.end())
  {
    return {0, false};
  }
  // get the grasp for this goal
  bool valid_grasp = _goals.at(iter->second).grasp_id == grasp_id;
  return {iter->second, valid_grasp};
}

std::vector<unsigned int> MultiGraspGoalSet::getGoalIds(unsigned int node_id) const
{
  auto iter = _roadmap_id_to_goal_id.find(node_id);
  if (iter == _roadmap_id_to_goal_id.end())
  {
    return {};
  }
  return {iter->second};  // TODO what if there are multiple goals at the same node?
}

void MultiGraspGoalSet::getGoals(std::vector<MultiGraspMP::Goal>& goals) const
{
  goals.clear();
  for (auto elem : _goals)
  {
    goals.push_back(elem.second);
  }
}

std::vector<::placement::mp::MultiGraspMP::Goal> MultiGraspGoalSet::getGoals() const
{
  std::vector<MultiGraspMP::Goal> goals;
  getGoals(goals);
  return goals;
}

MultiGraspGoalSet::GoalIterator MultiGraspGoalSet::begin() const
{
  return GoalIterator(_goals.begin());
}

MultiGraspGoalSet::GoalIterator MultiGraspGoalSet::end() const
{
  return GoalIterator(_goals.end());
}

MultiGraspGoalSetPtr MultiGraspGoalSet::createSubset(const std::set<unsigned int>& grasp_ids) const
{
  auto new_goal_set = std::make_shared<MultiGraspGoalSet>(_roadmap);
  for (unsigned int grasp_id : grasp_ids)
  {
    if (_grasp_id_to_goal_ids.count(grasp_id))
    {
      for (auto goal_id : _grasp_id_to_goal_ids.at(grasp_id))
      {
        new_goal_set->addGoalToMembers(_goals.at(goal_id), _goal_id_to_roadmap_id.at(goal_id));
      }
    }
  }
  return new_goal_set;
}

std::set<unsigned int> MultiGraspGoalSet::getGraspSet() const
{
  std::set<unsigned int> grasp_ids;
  for (auto id_goal_pair : _goals)
  {
    grasp_ids.insert(id_goal_pair.second.grasp_id);
  }
  return grasp_ids;
}

void MultiGraspGoalSet::addGoalToMembers(const MultiGraspMP::Goal& goal, unsigned int rid)
{
  _goals.insert(std::make_pair(goal.id, goal));
  _goal_id_to_roadmap_id[goal.id] = rid;
  _roadmap_id_to_goal_id[rid] = goal.id;
  // add to _grasp_id_to_goal_ids mapping
  auto grasp_iter = _grasp_id_to_goal_ids.find(goal.grasp_id);
  if (grasp_iter != _grasp_id_to_goal_ids.end())
  {
    grasp_iter->second.insert(goal.id);
  }
  else
  {
    _grasp_id_to_goal_ids[goal.grasp_id] = {goal.id};
  }
  // adjust quality range
  _quality_range.first = std::min(_quality_range.first, goal.quality);
  _quality_range.second = std::max(_quality_range.second, goal.quality);
}

/************************************** MultiGoalCostToGo **************************************/
MultiGoalCostToGo::MultiGoalCostToGo(MultiGraspGoalSetConstPtr goal_set, const GoalPathCostParameters& params,
                                     std::pair<double, double> quality_range)
  : MultiGoalCostToGo(goal_set, params.path_cost, params.lambda, quality_range)
{
}

MultiGoalCostToGo::MultiGoalCostToGo(MultiGraspGoalSetConstPtr goal_set,
                                     const std::function<double(const Config&, const Config&)>& path_cost,
                                     double lambda, std::pair<double, double> quality_range)
{
  if (quality_range.first == quality_range.second)  // check if custom goal quality range is provided or not
  {
    quality_range = goal_set->getGoalQualityRange();
  }
  _quality_normalizer = (quality_range.second - quality_range.first);
  _quality_normalizer = _quality_normalizer == 0.0 ? 1.0 : _quality_normalizer;
  _goal_distance.scaled_lambda = lambda / _quality_normalizer;
  _goal_distance.path_cost = path_cost;
  assert(_goal_distance.path_cost);
  _max_quality = quality_range.second;
  // now add the goals to nearest neighbor data structures
  auto dist_fun = std::bind(&MultiGoalCostToGo::GoalDistanceFn::distance, &_goal_distance, std::placeholders::_1,
                            std::placeholders::_2);
  _goals.setDistanceFunction(dist_fun);
  for (auto goal_iter = goal_set->begin(); goal_iter != goal_set->end(); ++goal_iter)
  {
    _goals.add(*goal_iter);
  }
}

MultiGoalCostToGo::~MultiGoalCostToGo() = default;

double MultiGoalCostToGo::costToGo(const Config& a) const
{
  return nearestGoal(a).first;
}

std::pair<double, placement::mp::MultiGraspMP::Goal> MultiGoalCostToGo::nearestGoal(const Config& a) const
{
  if (_goals.size() == 0)
  {
    throw std::logic_error("[MultiGoalCostToGo::nearestGoal] No goals known. Can not compute cost to go.");
  }
  MultiGraspMP::Goal dummy_goal;
  dummy_goal.config = a;
  dummy_goal.quality = _max_quality;
  const auto& nn = _goals.nearest(dummy_goal);
  return {_goal_distance.distance_const(nn, dummy_goal), nn};
}

double MultiGoalCostToGo::qualityToGoalCost(double quality) const
{
  return _goal_distance.scaled_lambda * (_max_quality - quality);
}
