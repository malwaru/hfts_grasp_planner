#include <hfts_grasp_planner/placement/mp/mgsearch/MGGraphSearchMP.h>

using namespace placement::mp::mgsearch;

MGGraphSearchMP::MGGraphSearchMP(mgsearch::StateSpacePtr state_space, const Config& start_config,
                                 const Parameters& params)
  : _state_space(state_space), _params(params)
{
  auto edge_computer = std::make_shared<IntegralEdgeCostComputer>(_state_space);
  _roadmap = std::make_shared<Roadmap>(_state_space, edge_computer, 1000, "/tmp/roadmap", "/tmp/validation_log");
  // _roadmap->setLogging("/tmp/roadmap", "/tmp/validation_log");
  _goal_set = std::make_shared<MultiGraspGoalSet>(_roadmap);
  // add start node
  _start_node = _roadmap->addNode(start_config);
}

MGGraphSearchMP::~MGGraphSearchMP() = default;

bool MGGraphSearchMP::plan(MultiGraspMP::Solution& sol)
{
  // create goal distance function, for this collect all goals in a vector first
  auto cspace_distance = std::bind(&StateSpace::distance, _state_space, std::placeholders::_1, std::placeholders::_2);
  auto goal_distance_fn = std::make_shared<MGGoalDistance>(_goal_set, cspace_distance, _params.lambda);
  // get the grasps we actually have to plan for
  std::set<unsigned int> grasp_ids;
  std::vector<MultiGraspMP::Goal> goals;
  _goal_set->getGoals(goals);
  for (auto goal : goals)
  {
    grasp_ids.insert(goal.grasp_id);
  }
  // get start id
  assert(not _start_node.expired());
  unsigned int start_id = _start_node.lock()->uid;
  // create the graph and plan
  switch (_params.graph_type)
  {
    case GraphType::SingleGraspGraph: {
      // solve the problem for each grasp separately
      for (auto grasp_id : grasp_ids)
      {
        SingleGraspRoadmapGraph graph(_roadmap, _goal_set, goal_distance_fn, grasp_id, start_id);
        SearchResult sr;
        // now solve the problem for this grasp using the specified algorithm
        switch (_params.algo_type)
        {
          case AlgorithmType::Astar: {
            RAVELOG_DEBUG("Planning with A* on single grasp graph for grasp " + std::to_string(grasp_id));
            astar::aStarSearch<SingleGraspRoadmapGraph>(graph, sr);
            break;
          }
          case AlgorithmType::LWAstar: {
            RAVELOG_DEBUG("Planning with LWA* on single grasp graph for grasp " + std::to_string(grasp_id));
            lwastar::lwaStarSearch<SingleGraspRoadmapGraph>(graph, sr);
            break;
          }
          case AlgorithmType::LPAstar: {
            RAVELOG_DEBUG("Planning with LPA* on single grasp graph for grasp " + std::to_string(grasp_id));
            lpastar::lpaStarSearch<SingleGraspRoadmapGraph>(graph, sr);
            break;
          }
          default:
            RAVELOG_ERROR("Algorithm type not implemented yet");
        }
        // pick the best solution
        if (sr.solved && sr.path_cost < sol.cost)
        {
          extractSolution<SingleGraspRoadmapGraph>(sr, sol, graph);
        }
      }
      break;
    }
    case GraphType::MultiGraspGraph: {
      // create a graph that captures all grasps
      MultiGraspRoadmapGraph graph(_roadmap, _goal_set, goal_distance_fn, grasp_ids, start_id);
      SearchResult sr;
      // solve the problem with the specified algorithm
      switch (_params.algo_type)
      {
        case AlgorithmType::Astar: {
          RAVELOG_DEBUG("Planning with A* on multi-grasp graph");
          astar::aStarSearch<MultiGraspRoadmapGraph>(graph, sr);
          break;
        }
        case AlgorithmType::LWAstar: {
          RAVELOG_DEBUG("Planning with LWA* on multi-grasp graph");
          lwastar::lwaStarSearch<MultiGraspRoadmapGraph>(graph, sr);
          break;
        }
        case AlgorithmType::LPAstar: {
          RAVELOG_DEBUG("Planning with LPA* on multi-grasp graph");
          lpastar::lpaStarSearch<MultiGraspRoadmapGraph>(graph, sr);
          break;
        }
        default:
          RAVELOG_ERROR("Algorithm type not implemented yet");
      }
      if (sr.solved)
      {
        extractSolution<MultiGraspRoadmapGraph>(sr, sol, graph);
      }
      break;
    }
    default:
      RAVELOG_ERROR("Graph type not implemented yet");
  }
  return sol.path != nullptr;
}

void MGGraphSearchMP::addGoal(const MultiGraspMP::Goal& goal)
{
  _goal_set->addGoal(goal);
}

void MGGraphSearchMP::removeGoals(const std::vector<unsigned int>& goal_ids)
{
  _goal_set->removeGoals(goal_ids);
}