#pragma once
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Algorithms.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <set>
#include <vector>

namespace placement
{
namespace mp
{
namespace mgsearch
{
class MGGraphSearchMP
{
public:
  enum GraphType
  {
    SingleGraspGraph = 0,
    MultiGraspGraph = 1,
    FoldedMultiGraspGraphStationary = 2,  // naive, stationary heuristic
    FoldedMultiGraspGraphDynamic = 3,     // non-stationary heuristic, TODO: currently only compatible with LPAStar
    LazyWeightedMultiGraspGraph = 4,      // evalutes costs only for the robot without grasps unless explicity asked
  };
  static const unsigned int NUM_GRAPH_TYPES = 5;
  enum AlgorithmType
  {
    Astar = 0,
    LWAstar = 1,           // lazy weighted A*
    LPAstar = 2,           // life-long planning A*
    LWLPAstar = 3,         // lazy weighted life-long planning A*
    LazySP_LLPAstar = 4,   // Lazy SP using lazy LPAstar
    LazySP_LWLPAstar = 5,  // Lazy SP using lazy-weighted LPAstar; only makes sense with LazyWeightedMultiGraspGraph
    LazySP_LPAstar = 6,    // Lazy SP using non-lazy LPAstar; only makes sense with LazyWeightedMultiGraspGraph
  };
  static const unsigned int NUM_ALGORITHM_TYPES = 7;
  enum EdgeSelectorType
  {
    FirstUnknown = 0,
    LastUnknown = 1
  };
  struct Parameters
  {
    AlgorithmType algo_type;
    GraphType graph_type;
    EdgeSelectorType edge_selector_type;  // only for LazySP
    double lambda;                        // weight between path and goal cost
    Parameters() : algo_type(AlgorithmType::Astar), graph_type(GraphType::SingleGraspGraph), lambda(1.0)
    {
    }
  };

  /**
   * Return a string-representation of the given graph type.
   */
  static std::string getName(GraphType gtype);

  /**
   * Return a string-representation of the given algorithm type.
   */
  static std::string getName(AlgorithmType atype);

  /**
   * Return the type of the algorithm given a string representation.
   * If the string representation does not match any type, a runtime error is thrown.
   */
  static GraphType getGraphType(const std::string& name);

  /**
   * Return the type of the algorithm given a string representation.
   * If the string representation does not match any type, a runtime error is thrown.
   */
  static AlgorithmType getAlgorithmType(const std::string& name);

  MGGraphSearchMP(mgsearch::StateSpacePtr state_space, const Config& start, const Parameters& params);
  ~MGGraphSearchMP();

  bool plan(MultiGraspMP::Solution& sol);
  void addGoal(const MultiGraspMP::Goal& goal);
  void removeGoals(const std::vector<unsigned int>& goal_ids);
  Parameters _params;

private:
  template <typename G>
  void extractSolution(SearchResult& sr, MultiGraspMP::Solution& sol, const G& graph)
  {
    MultiGraspMP::WaypointPathPtr wp_path = std::make_shared<MultiGraspMP::WaypointPath>();
    // extract solution path
    for (unsigned int vid : sr.path)
    {
      auto [rid, gid] = graph.getGraspRoadmapId(vid);
      auto node = _roadmap->getNode(rid);
      assert(node);
      wp_path->push_back(node->config);
    }
    // get goal id
    auto [rid, gid] = graph.getGraspRoadmapId(sr.path.back());
    auto [goal_id, valid_goal] = _goal_set->getGoalId(rid, gid);
    assert(valid_goal);
    sol.goal_id = goal_id;
    sol.path = wp_path;
    sol.cost = sr.cost();
  }

  // overload for FoldedMultiGraspRoadmapGraph
  template <BackwardsHeuristicType htype>
  void extractSolution(SearchResult& sr, MultiGraspMP::Solution& sol, const FoldedMultiGraspRoadmapGraph<htype>& graph)
  {
    MultiGraspMP::WaypointPathPtr wp_path = std::make_shared<MultiGraspMP::WaypointPath>();
    // iterate over path in reverse order; abort when we reached the base layer
    std::pair<unsigned int, unsigned int> rid_gid_pair;
    unsigned int gid;
    unsigned int last_rid;
    for (auto iter = sr.path.rbegin(); iter != sr.path.rend(); ++iter)
    {
      bool gid_valid = false;
      std::tie(rid_gid_pair, gid_valid) = graph.getGraspRoadmapId(*iter);
      if (gid_valid)
      {
        last_rid = rid_gid_pair.first;
        gid = rid_gid_pair.second;
        auto node = _roadmap->getNode(rid_gid_pair.first);
        assert(node);
        wp_path->push_back(node->config);
      }
    }
    auto [goal_id, valid_goal] = _goal_set->getGoalId(last_rid, gid);
    assert(valid_goal);
    sol.goal_id = goal_id;
    sol.path = wp_path;
    sol.cost = sr.cost();
  }

  mgsearch::StateSpacePtr _state_space;
  mgsearch::RoadmapPtr _roadmap;
  mgsearch::Roadmap::NodeWeakPtr _start_node;
  mgsearch::MultiGraspGoalSetPtr _goal_set;
};
typedef std::shared_ptr<MGGraphSearchMP> MGGraphSearchMPPtr;
typedef std::shared_ptr<const MGGraphSearchMP> MGGraphSearchMPConstPtr;
}  // namespace mgsearch
}  // namespace mp
}  // namespace placement
