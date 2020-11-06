#pragma once
// stl
#include <fstream>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
// own
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
// ompl
#include <ompl/datastructures/NearestNeighborsGNAT.h>

namespace placement
{
namespace mp
{
namespace mgsearch
{
// Interfaces used by Roadmap
// TODO make these to template arguments? Will we ever exchange these during runtime?
class StateSpace
{
public:
  struct SpaceInformation
  {
    Config lower;
    Config upper;
    unsigned int dimension;
  };
  virtual ~StateSpace() = 0;
  /**
   * Check whether the robot can attain configuration c without considering a grasped object.
   */
  virtual bool isValid(const Config& c) const = 0;
  /**
   * Check whether the robot can attain configuration c when grasping the object with the given grasp.
   * @param c - robot configuration
   * @param grasp_id - grasp identifier
   * @param only_obj - if true, only check the object for validity, not the robot itself
   */
  virtual bool isValid(const Config& c, unsigned int grasp_id, bool only_obj = false) const = 0;

  /**
   * Return the cost of being in configuration c.
   * This cost may simply be 0 (c is valid) or infinity (c is invalid), if there is no underlying state cost.
   * If there is an underlying state cost, the returned value may be any value r in [0, infinity]
   */
  virtual double cost(const Config& c) const = 0;
  virtual double cost(const Config& c, unsigned int grasp_id) const = 0;

  /**
   * Return the distance between the two given configurations.
   */
  virtual double distance(const Config& a, const Config& b) const = 0;

  /**
   * Return the dimension of the state space.
   */
  virtual unsigned int getDimension() const = 0;
  virtual void getBounds(Config& lower, Config& upper) const = 0;
  virtual void getValidGraspIds(std::vector<unsigned int>& grasp_ids) const = 0;
  /**
   * Return the number of valid grasps.
   */
  virtual unsigned int getNumGrasps() const = 0;

  SpaceInformation getSpaceInformation() const
  {
    SpaceInformation si;
    getBounds(si.lower, si.upper);
    si.dimension = getDimension();
    return si;
  }
};
typedef std::shared_ptr<StateSpace> StateSpacePtr;

class EdgeCostComputer
{
public:
  virtual ~EdgeCostComputer() = 0;
  /**
   * Cheap to compute lower bound of the cost to transition from config a to config b.
   */
  virtual double lowerBound(const Config& a, const Config& b) const = 0;
  /**
   * True cost to transition from config a to config b without any grasped object.
   */
  virtual double cost(const Config& a, const Config& b) const = 0;
  /**
   * True cost to transition from config a to config b when grasping an object with grasp grasp_id.
   */
  virtual double cost(const Config& a, const Config& b, unsigned int grasp_id) const = 0;
};
typedef std::shared_ptr<EdgeCostComputer> EdgeCostComputerPtr;

/**
 * An edge cost computer that computes the cost between two configurations (a, b) by integrating
 * the state cost along a straight line path.
 */
class IntegralEdgeCostComputer : public EdgeCostComputer
{
public:
  IntegralEdgeCostComputer(StateSpacePtr ss, double integral_step_size = 0.1);
  ~IntegralEdgeCostComputer();
  double lowerBound(const Config& a, const Config& b) const override;
  double cost(const Config& a, const Config& b) const override;
  double cost(const Config& a, const Config& b, unsigned int grasp_id) const override;

private:
  const StateSpacePtr _state_space;
  const double _step_size;
  double integrateCosts(const Config& a, const Config& b, const std::function<double(const Config&)>& cost_fn) const;
};

// A single function that provides a lower bound on the cost to move between two configurations.
typedef std::function<double(const Config&, const Config&)> PathCostFn;

/**
 *  A struct representing the parameters needed to compute a combined goal and path cost:
 *  cost = path_cost + lambda * goal_cost
 */
struct GoalPathCostParameters
{
  PathCostFn path_cost;
  double lambda;
  GoalPathCostParameters(const PathCostFn& pc, double l) : path_cost(pc), lambda(l)
  {
  }
};

/**
 * This class encapsulates a conditional roadmap for a robot manipulator transporting
 * an object. The roadmap is conditioned on a discrete set of grasps the robot may have
 * on the object. The roadmap is constructed lazily.
 */
class Roadmap
{
public:
  struct Node;
  struct Edge;
  typedef std::shared_ptr<Node> NodePtr;
  typedef std::weak_ptr<Node> NodeWeakPtr;
  typedef std::shared_ptr<Edge> EdgePtr;
  typedef std::weak_ptr<Edge> EdgeWeakPtr;

  struct Node
  {
    // maps target node id to edge
    typedef std::unordered_map<unsigned int, EdgePtr> EdgeMap;
    typedef EdgeMap::const_iterator EdgeIterator;
    // unique node id
    const unsigned int uid;
    // Configuration represented by this node
    const Config config;
    /**
     * Return the edge that leads to the node with target_id.
     * If the specified node is not adjacent, nullptr is returned.
     */
    EdgePtr getEdge(unsigned int target_id)
    {
      auto iter = edges.find(target_id);
      if (iter == edges.end())
        return nullptr;
      return iter->second;
    }

    // these iterators remain valid even if edges are deleted, see
    // https://en.cppreference.com/w/cpp/container/unordered_map/erase
    std::pair<EdgeMap::const_iterator, EdgeMap::const_iterator> getEdgesIterators() const
    {
      return std::make_pair(edges.cbegin(), edges.cend());
    }

    bool getConditionalValidity(unsigned int gid, bool& validity)
    {
      auto iter = conditional_validity.find(gid);
      if (iter != conditional_validity.end())
      {
        validity = iter->second;
        return true;
      }
      validity = false;
      return false;
    }

  protected:
    friend class Roadmap;
    bool initialized;  // initialized = collision-free
    // 0 = edges not initialized, >= 1 - last densification generation edges have been updated
    unsigned int densification_gen;
    // map node id to edge (maps target node id to edge)
    EdgeMap edges;
    // stores keys of edges in edges that can be deleted
    std::vector<unsigned int> edges_to_delete;
    // stores validity in dependence on grasp id
    std::unordered_map<unsigned int, bool> conditional_validity;
    // Constructor
    Node(unsigned int tuid, const Config& tconfig)
      : uid(tuid), config(tconfig), initialized(false), densification_gen(0)
    {
    }
  };

  struct Edge
  {
    double base_cost;
    // flag whether base_cost is true base cost or just a lower bound
    bool base_evaluated;
    // maps grasp id to a cost
    typedef std::unordered_map<unsigned int, double> GraspConditionalCostMap;
    GraspConditionalCostMap conditional_costs;
    NodeWeakPtr node_a;
    NodeWeakPtr node_b;
    Edge(NodePtr a, NodePtr b, double bc);
    // Convenience function returning the node that isn't n
    NodePtr getNeighbor(NodePtr n) const;
    // convenience function to return the best known approximate cost of this edge for the given grasp
    double getBestKnownCost(unsigned int gid) const;
  };

  class Logger
  {
  public:
    Logger();
    ~Logger();
    void setLogPath(const std::string& roadmap_file, const std::string& log_file);

    // log the addition of a new node
    void newNode(NodePtr node);
    // log a validity check on the given node
    void nodeValidityChecked(NodePtr node, bool val);
    void nodeValidityChecked(NodePtr node, unsigned int grasp_id, bool val);
    // log the cost computation of an edge
    void edgeCostChecked(NodePtr a, NodePtr b, double cost);
    void edgeCostChecked(NodePtr a, NodePtr b, unsigned int grasp_id, double cost);
    // log custom event
    void logCustomEvent(const std::string& msg);

  private:
    std::ofstream _roadmap_fs;
    std::ofstream _log_fs;
  };

  Roadmap(StateSpacePtr state_space, EdgeCostComputerPtr edge_cost_computer, unsigned int batch_size = 10000,
          const std::string& log_roadmap_path = "", const std::string& log_path = "");
  virtual ~Roadmap();
  // Tell the roadmap to densify
  void densify();
  void densify(unsigned int batch_size);

  // enable logging to the given files
  void setLogging(const std::string& roadmap_path, const std::string& log_path);

  /**
   * Log a custom event message to the log file logging cost evaluations.
   */
  void logCustomEvent(const std::string& msg);
  /**
   * Retrieve a node.
   * Returns nullptr if node with given id doesn't exist.
   */
  NodePtr getNode(unsigned int node_id) const;

  /**
   * Add a new node at the given configuration.
   * Use this function to add the start node.
   */
  NodeWeakPtr addNode(const Config& config);

  /**
   * Update the nodes adjacency list if needed. The adjacency list needs to be updated,
   * if this function has a) never been called before on node, or b) densify(..) has been called
   * after the last time this function was called for node. In addition, this function
   * removes edges from the adjacency list that have been found to be invalid even without any grasp.
   * To be safe, you should call this function everytime before accessing a node's neighbors.
   */
  void updateAdjacency(NodePtr node);

  /**
   * Check the given node for validity, and update roadmap if necessary.
   * @param node - the node to check
   * @return true, if the node is valid (base), else false. In case of false, the node is removed
   *  from the roadmap and node is set to nullptr.
   *  If this function returned true, you can safely acquire a lock on node, else node is no longer valid.
   */
  bool isValid(NodeWeakPtr node);

  /**
   * Convenience wrapper for isValid(NodeWeakPtr) with node id instead.
   * @param node_id: the id of the roadmap node to check for validity. If the node does not exist, return false.
   */
  bool isValid(unsigned int node_id);

  /**
   * Just like isValid, but the return value indicates whether the node is valid for the given grasp.
   * The node is of course only removed if the base is invalid, not if the collision is induced by the grasp.
   * @return true, if the node is valid and not in collision for the given grasp, else false.
   */
  bool isValid(NodeWeakPtr node, unsigned int grasp_id);

  /**
   * Convenience wrapper for isValid(NodeWeakPtr, grasp_id) with node id instead.
   * @param node_id: the id of the roadmap node to check for validity. If the node does not exist, return false.
   * @param grasp_id: the id of the grasp to test validity for.
   */
  bool isValid(unsigned int node_id, unsigned int grasp_id);

  /**
   * Return all valid grasp ids. This function simply forwards this call to the underlying state space.
   */
  void getValidGraspIds(std::vector<unsigned int>& grasp_ids) const
  {
    _state_space->getValidGraspIds(grasp_ids);
  }

  /**
   * Compute the base cost of the given edge (for no grasp).
   */
  std::pair<bool, double> computeCost(EdgePtr edge);
  std::pair<bool, double> computeCost(EdgeWeakPtr edge);

  /**
   * Compute the edge cost for the given edge given the grasp.
   * @param edge - the edge to compute cost for
   * @param grasp_id - the grasp id to compute the cost for
   * @return pair (valid, cost) - valid = true if cost is finite
   */
  std::pair<bool, double> computeCost(EdgePtr edge, unsigned int grasp_id);

private:
  const StateSpacePtr _state_space;
  const StateSpace::SpaceInformation _si;
  EdgeCostComputerPtr _cost_computer;
  ::ompl::NearestNeighborsGNAT<NodePtr> _nn;             // owner of nodes
  std::unordered_map<unsigned int, NodeWeakPtr> _nodes;  // node id to pointer
  Logger _logger;
  unsigned int _batch_size;
  unsigned int _node_id_counter;
  unsigned int _halton_seq_id;
  unsigned int _densification_gen;
  double _gamma_prm;
  double _adjacency_radius;

  void scaleToLimits(Config& config) const;
  void deleteNode(NodePtr node);
};
typedef std::shared_ptr<Roadmap> RoadmapPtr;

// Forward declarations
class MultiGraspGoalSet;
typedef std::shared_ptr<MultiGraspGoalSet> MultiGraspGoalSetPtr;
typedef std::shared_ptr<const MultiGraspGoalSet> MultiGraspGoalSetConstPtr;

class MultiGraspGoalSet
{
public:
  /**
   * Construct a new MultiGraspGoalSet.
   */
  MultiGraspGoalSet(RoadmapPtr roadmap);
  ~MultiGraspGoalSet();

  /**
   *  Add a new goal to the set.
   *  The corresponding configuration is also added to the underlying roadmap.
   * @param goal: the goal to add.
   * @return: true if goal has succesfully been added, else false (goal is invalid)
   */
  bool addGoal(const MultiGraspMP::Goal& goal);

  /**
   *  Return the goal with the given id. Throws an exception if gid is invalid.
   */
  MultiGraspMP::Goal getGoal(unsigned int gid) const;

  /**
   *  Remove the specified goal.
   *  The corresponding configuration is not removed from the underlying roadmap.
   */
  void removeGoal(unsigned int gid);

  /**
   *  Remove the specified goals.
   *  The corresponding configurations are not removed from the underlying roadmap.
   */
  void removeGoals(const std::vector<unsigned int>& goal_ids);

  /**
   *  Return whether the given node is a goal under the given grasp.
   */
  bool isGoal(Roadmap::NodePtr node, unsigned int grasp_id) const;

  /**
   *  Return whether the roadmap node with id <node_id> is a goal under the given grasp id.
   */
  bool isGoal(unsigned int node_id, unsigned int grasp_id) const;

  /**
   * Return whether there exists a grasp for which the roadmap node with id <node_id> is a goal.
   */
  bool canBeGoal(unsigned int node_id) const;

  /**
   * Return the roadmap id of the goal with the given id.
   */
  unsigned int getRoadmapId(unsigned int goal_id) const;

  /**
   * Return the goal id associated with the roadmap node under the given grasp.
   * If the node is not a goal for this grasp, the returned bool is false, and the returned id meaningless.
   */
  std::pair<unsigned int, bool> getGoalId(unsigned int node_id, unsigned int grasp_id);

  /**
   * Return the ids of all goals associated with the given roadmap node.
   */
  std::vector<unsigned int> getGoalIds(unsigned int node_id) const;

  /**
   * Return a vector containing all currently active goals.
   */
  void getGoals(std::vector<MultiGraspMP::Goal>& goals) const;

  // Provides read access to goals in goal set
  struct GoalIterator
  {
    ~GoalIterator() = default;
    GoalIterator& operator++();
    bool operator==(const GoalIterator& other) const;
    bool operator!=(const GoalIterator& other) const;
    const MultiGraspMP::Goal& operator*();
    const MultiGraspMP::Goal* operator->();
    // iterator traits
    using difference_type = long;
    using value_type = MultiGraspMP::Goal;
    using pointer = const MultiGraspMP::Goal*;
    using reference = const MultiGraspMP::Goal&;
    using iterator_category = std::forward_iterator_tag;

  private:
    friend class MultiGraspGoalSet;
    GoalIterator(std::unordered_map<unsigned int, MultiGraspMP::Goal>::const_iterator iter);
    std::unordered_map<unsigned int, MultiGraspMP::Goal>::const_iterator _iter;
  };

  /**
   * Return forward iterator over this goal set.
   */
  GoalIterator begin() const;

  /**
   * Return end-iterator for this goal set.
   */
  GoalIterator end() const;

  /**
   * Create a new MultiGraspGoalSet from this set that only contains the goals associated with
   * the grasps in grasp_ids.
   * @param grasp_ids: set of grasp ids
   * @return a new grasp goal with all goals from this set for the given grasps
   */
  MultiGraspGoalSetPtr createSubset(const std::set<unsigned int>& grasp_ids) const;

  /**
   * Return a set containing the grasp ids for which this goal set contains goals.
   */
  std::set<unsigned int> getGraspSet() const;

  /**
   * Return the minimal and maximal goal quality from this set.
   */
  std::pair<double, double> getGoalQualityRange() const
  {
    return _quality_range;
  }

private:
  // goal id -> goal
  std::unordered_map<unsigned int, MultiGraspMP::Goal> _goals;
  // goal id -> roadmap node id
  std::unordered_map<unsigned int, unsigned int> _goal_id_to_roadmap_id;
  // roadmap node id -> goal id; TODO: what if there are multiple goals for the same roadmap node?
  std::unordered_map<unsigned int, unsigned int> _roadmap_id_to_goal_id;
  // grasp id -> list of goals with this grasp
  std::unordered_map<unsigned int, std::set<unsigned int>> _grasp_id_to_goal_ids;
  const RoadmapPtr _roadmap;
  std::pair<double, double> _quality_range;
  // updates member variables to include the given goal. rid is roadmap id
  void addGoalToMembers(const MultiGraspMP::Goal& goal, unsigned int rid);
};

class MultiGoalCostToGo
{
public:
  // TODO this class can not be moved properly for some reason.
  /**
   * Construct a multi-goal cost-to-go function.
   * The cost-to-go function expresses the term
   * h(q) = min_{g in G} (d(q, g) + lambda * cost(g)), where g in G are the goals, d(q_1, q_2) a lower bound on path
   * cost. The cost of a goal cost(g) is computed as cost(g) = (o_max - o_g) / (o_max - o_min) where o_g denotes the
   * goal's quality and o_max = max_{g in G} o_g, o_min = min_{g in G} o_g (larger qualities are better). The parameter
   * lambda scales between the goal cost, which is in range [0, 1], and the path cost d(q1, q2). Note: For new goals,
   * you need to construct a new instance, due to the fact that goal quality values are normalized w.r.t min and max
   * quality.
   *
   * This function ignores the grasps that each goal is for. If you need a cost-to-go function for a specific
   * grasp, simply create a new MultiGoalCostToGo with only the goals for the grasp of interest.
   *
   * @param goal_set: goals
   * @param path_cost: lower bound on path cost to move from one configuration to another
   * @param lambda:  parameter to scale between path cost and grasp cost
   * @param quality_range: the minimal and maximal goal quality (min, max). If not provided, these values are obtained
   * by iterating over goal_set.
   */
  MultiGoalCostToGo(MultiGraspGoalSetConstPtr goal_set, const PathCostFn& path_cost, double lambda,
                    std::pair<double, double> quality_range = {0.0, 0.0});
  MultiGoalCostToGo(MultiGraspGoalSetConstPtr goal_set, const GoalPathCostParameters& params,
                    std::pair<double, double> quality_range = {0.0, 0.0});
  ~MultiGoalCostToGo();
  /**
   * Return a lower bound of the cost to go from a to any of the goals given upon construction.
   */
  double costToGo(const Config& a) const;
  // return the goal cost of a goal with the given quality
  double qualityToGoalCost(double quality) const;

private:
  struct GoalDistanceFn
  {
    double scaled_lambda;
    PathCostFn path_cost;
    double distance(const MultiGraspMP::Goal& ga, const MultiGraspMP::Goal& gb)
    {
      return distance_const(ga, gb);
    }
    double distance_const(const MultiGraspMP::Goal& ga, const MultiGraspMP::Goal& gb) const
    {
      return path_cost(ga.config, gb.config) + scaled_lambda * abs(ga.quality - gb.quality);
    }
  };
  ::ompl::NearestNeighborsGNAT<MultiGraspMP::Goal> _goals;
  GoalDistanceFn _goal_distance;
  double _max_quality;
  double _quality_normalizer;
};
typedef std::shared_ptr<MultiGoalCostToGo> MultiGoalCostToGoPtr;
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement