#include <hfts_grasp_planner/placement/mp/mgsearch/MGGraphSearchMP.h>

using namespace placement::mp::mgsearch;

std::string MGGraphSearchMP::getName(GraphType gtype)
{
  switch (gtype)
  {
    case GraphType::FoldedMultiGraspGraphDynamic: {
      return "FoldedMultiGraspGraphDynamic";
    }
    case GraphType::FoldedMultiGraspGraphStationary: {
      return "FoldedMultiGraspGraphStationary";
    }
    case GraphType::LazyWeightedMultiGraspGraph: {
      return "LazyWeightedMultiGraspGraph";
    }
    case GraphType::LazyEdgeWeightedMultiGraspGraph: {
      return "LazyEdgeWeightedMultiGraspGraph";
    }
    case GraphType::MultiGraspGraph: {
      return "MultiGraspGraph";
    }
    case GraphType::SingleGraspGraph: {
      return "SingleGraspGraph";
    }
    default:
      throw std::logic_error("Unknown graph type");
  }
}

std::string MGGraphSearchMP::getName(AlgorithmType atype)
{
  switch (atype)
  {
    case AlgorithmType::Astar: {
      return "Astar";
    }
    case AlgorithmType::LPAstar: {
      return "LPAstar";
    }
    case AlgorithmType::LWAstar: {
      return "LWAstar";
    }
    case AlgorithmType::LWLPAstar: {
      return "LWLPAstar";
    }
    case AlgorithmType::LazySP_LLPAstar: {
      return "LazySP_LLPAstar";
    }
    case AlgorithmType::LazySP_LPAstar: {
      return "LazySP_LPAstar";
    }
    case AlgorithmType::LazySP_LWLPAstar: {
      return "LazySP_LWLPAstar";
    }
    default:
      throw std::logic_error("Unknown algorithm type");
  }
}

std::string MGGraphSearchMP::getName(EdgeSelectorType etype)
{
  switch (etype)
  {
    case EdgeSelectorType::FirstUnknown:
      return "FirstUnknown";
    case EdgeSelectorType::LastUnknown:
      return "LastUnknown";
    default:
      throw std::logic_error("Unknown edge selector type");
  }
}

std::string strToLower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

MGGraphSearchMP::GraphType MGGraphSearchMP::getGraphType(const std::string& name)
{
  for (unsigned int enum_id = 0; enum_id < NUM_GRAPH_TYPES; ++enum_id)
  {
    std::string original_name = getName(static_cast<GraphType>(enum_id));
    if (name == original_name or name == strToLower(original_name))
    {
      return static_cast<GraphType>(enum_id);
    }
  }
  throw std::runtime_error("Invalid graph type " + name);
}

MGGraphSearchMP::AlgorithmType MGGraphSearchMP::getAlgorithmType(const std::string& name)
{
  for (unsigned int enum_id = 0; enum_id < NUM_ALGORITHM_TYPES; ++enum_id)
  {
    std::string original_name = getName(static_cast<AlgorithmType>(enum_id));
    if (name == original_name or name == strToLower(original_name))
    {
      return static_cast<AlgorithmType>(enum_id);
    }
  }
  throw std::runtime_error("Invalid algorithm type " + name);
}

MGGraphSearchMP::MGGraphSearchMP(mgsearch::StateSpacePtr state_space, const Config& start_config,
                                 const Parameters& params)
  : _params(params), _state_space(state_space)
{
  auto edge_computer = std::make_shared<IntegralEdgeCostComputer>(_state_space);
  _roadmap = std::make_shared<Roadmap>(_state_space, edge_computer, params.batchsize, params.roadmap_log_path,
                                       params.logfile_path);
  _goal_set = std::make_shared<MultiGraspGoalSet>(_roadmap);
  // add start node
  _start_node = _roadmap->addNode(start_config);
  RAVELOG_DEBUG("Created MGGraphSearch with parameters: " + params.toString());
}

MGGraphSearchMP::~MGGraphSearchMP() = default;

bool MGGraphSearchMP::plan(MultiGraspMP::Solution& sol)
{
  // create goal distance function, for this collect all goals in a vector first
  auto cspace_distance = std::bind(&StateSpace::distance, _state_space, std::placeholders::_1, std::placeholders::_2);
  // auto goal_distance_fn = std::make_shared<MGGoalDistance>(_goal_set, cspace_distance, _params.lambda);
  GoalPathCostParameters cost_parameters(cspace_distance, _params.lambda);
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
        SingleGraspRoadmapGraph graph(_roadmap, _goal_set, cost_parameters, grasp_id, start_id);
        SearchResult sr;
        // now solve the problem for this grasp using the specified algorithm
        switch (_params.algo_type)
        {
          case AlgorithmType::Astar: {
            RAVELOG_INFO("Planning with A* on single grasp graph for grasp " + std::to_string(grasp_id));
            astar::aStarSearch<SingleGraspRoadmapGraph>(graph, sr);
            break;
          }
          case AlgorithmType::LWAstar: {
            RAVELOG_INFO("Planning with LWA* on single grasp graph for grasp " + std::to_string(grasp_id));
            lwastar::lwaStarSearch<SingleGraspRoadmapGraph>(graph, sr);
            break;
          }
          case AlgorithmType::LPAstar: {
            RAVELOG_INFO("Planning with LPA* on single grasp graph for grasp " + std::to_string(grasp_id));
            lpastar::lpaStarSearch<SingleGraspRoadmapGraph, lpastar::EdgeCostEvaluationType::Explicit>(graph, sr);
            break;
          }
          case AlgorithmType::LWLPAstar: {
            RAVELOG_INFO("Planning with lazy-weighted LPA* on single grasp graph for grasp " +
                         std::to_string(grasp_id));
            lpastar::lpaStarSearch<SingleGraspRoadmapGraph, lpastar::EdgeCostEvaluationType::LazyWeighted>(graph, sr);
            break;
          }
          case AlgorithmType::LazySP_LLPAstar: {
            RAVELOG_INFO("Planning with LazySP using lazy-weighted LPA* on single grasp graph for grasp " +
                         std::to_string(grasp_id));
            typedef lpastar::LPAStarAlgorithm<SingleGraspRoadmapGraph, lpastar::EdgeCostEvaluationType::Lazy>
                SearchAlgorithmType;
            lazysp::lazySP<SingleGraspRoadmapGraph, lazysp::FirstUnknownEdgeSelector, SearchAlgorithmType>(graph, sr);
            break;
          }
          default:
            RAVELOG_ERROR("Algorithm type " + getName(_params.algo_type) + " not implemented for SingleGraspGraph.");
        }
        // pick the best solution
        if (sr.solved && sr.cost() < sol.cost)
        {
          extractSolution<SingleGraspRoadmapGraph>(sr, sol, graph);
        }
      }
      break;
    }
    case GraphType::MultiGraspGraph: {
      // create a graph that captures all grasps
      MultiGraspRoadmapGraph graph(_roadmap, _goal_set, cost_parameters, grasp_ids, start_id);
      SearchResult sr;
      // solve the problem with the specified algorithm
      switch (_params.algo_type)
      {
        case AlgorithmType::Astar: {
          RAVELOG_INFO("Planning with A* on multi-grasp graph");
          astar::aStarSearch<MultiGraspRoadmapGraph<>>(graph, sr);
          break;
        }
        case AlgorithmType::LWAstar: {
          RAVELOG_INFO("Planning with LWA* on multi-grasp graph");
          lwastar::lwaStarSearch<MultiGraspRoadmapGraph<>>(graph, sr);
          break;
        }
        case AlgorithmType::LPAstar: {
          RAVELOG_INFO("Planning with LPA* on multi-grasp graph");
          lpastar::lpaStarSearch<MultiGraspRoadmapGraph<>, lpastar::EdgeCostEvaluationType::Explicit>(graph, sr);
          break;
        }
        case AlgorithmType::LWLPAstar: {
          RAVELOG_INFO("Planning with lazy-weighted LPA* on multi-grasp graph");
          lpastar::lpaStarSearch<MultiGraspRoadmapGraph<>, lpastar::EdgeCostEvaluationType::LazyWeighted>(graph, sr);
          break;
        }
        case AlgorithmType::LazySP_LLPAstar: {
          RAVELOG_INFO("Planning with LazySP using lazy LPA* on multi-grasp graph");
          typedef lpastar::LPAStarAlgorithm<MultiGraspRoadmapGraph<>, lpastar::EdgeCostEvaluationType::Lazy>
              SearchAlgorithmType;
          lazysp::lazySP<MultiGraspRoadmapGraph<>, lazysp::FirstUnknownEdgeSelector, SearchAlgorithmType>(graph, sr);
          break;
        }
        default:
          RAVELOG_ERROR("Algorithm type " + getName(_params.algo_type) +
                        " not supported in combination with MultiGraspRoadmapGraph");
      }
      if (sr.solved)
      {
        extractSolution<MultiGraspRoadmapGraph<>>(sr, sol, graph);
      }
      break;
    }
    case GraphType::FoldedMultiGraspGraphStationary: {
      // create folded multi-grasp graph
      typedef FoldedMultiGraspRoadmapGraph<BackwardsHeuristicType::LowerBound> FoldedGraph;
      FoldedGraph graph(_roadmap, _goal_set, cost_parameters, start_id);
      SearchResult sr;
      // solve the problem with the specified algorithm
      switch (_params.algo_type)
      {
        case AlgorithmType::LWAstar: {
          RAVELOG_INFO("Planning with LWA* on FoldedMultiGraspRoadmapGraph and naive heuristic");
          lwastar::lwaStarSearch<FoldedGraph>(graph, sr);
          break;
        }
        default:
          // TODO the stationary version should be compatible with all algrotithms, no?
          RAVELOG_ERROR("Algorithm type " + getName(_params.algo_type) + "  doesn't support FoldedMultiGraspGraph yet");
      }
      if (sr.solved)
      {
        extractSolution<BackwardsHeuristicType::LowerBound>(sr, sol, graph);
      }
      break;
    }
    case GraphType::FoldedMultiGraspGraphDynamic: {
      // create folded multi-grasp graph
      typedef FoldedMultiGraspRoadmapGraph<BackwardsHeuristicType::SearchAwareBestKnownDistance> FoldedGraph;
      FoldedGraph graph(_roadmap, _goal_set, cost_parameters, start_id);
      SearchResult sr;
      // solve the problem with the specified algorithm
      switch (_params.algo_type)
      {
        case AlgorithmType::LWAstar: {
          RAVELOG_INFO("Planning with LWA* on FoldedMultiGraspRoadmapGraph and dynamic heuristic");
          lwastar::lwaStarSearch<FoldedGraph>(graph, sr);
          break;
        }
        default:
          RAVELOG_ERROR("Algorithm type " + getName(_params.algo_type) + " doesn't support FoldedMultiGraspGraph yet");
      }
      if (sr.solved)
      {
        extractSolution<BackwardsHeuristicType::SearchAwareBestKnownDistance>(sr, sol, graph);
      }
      break;
    }
    case GraphType::LazyWeightedMultiGraspGraph: {
      MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp> graph(_roadmap, _goal_set, cost_parameters,
                                                                             grasp_ids, start_id);
      SearchResult sr;
      switch (_params.algo_type)
      {
        case AlgorithmType::LazySP_LLPAstar: {
          RAVELOG_INFO("Planning with LazySP using lazy LPA* on lazy-weighted multi-grasp graph");
          // TODO this is equivalent to LazySP on normal multi-grasp graph (apart from Edge selector)
          typedef lpastar::LPAStarAlgorithm<MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>,
                                            lpastar::EdgeCostEvaluationType::Lazy>
              SearchAlgorithmType;
          lazysp::lazySP<MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>,
                         lazysp::LastUnknownEdgeSelector, SearchAlgorithmType>(graph, sr);
          break;
        }
        case AlgorithmType::LazySP_LWLPAstar: {
          RAVELOG_INFO("Planning with LazySP using lazy weighted LPA* on lazy-weighted multi-grasp graph");
          typedef lpastar::LPAStarAlgorithm<MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>,
                                            lpastar::EdgeCostEvaluationType::LazyWeighted>
              SearchAlgorithmType;
          lazysp::lazySP<MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>,
                         lazysp::LastUnknownEdgeSelector, SearchAlgorithmType>(graph, sr);
          break;
        }
        case AlgorithmType::LazySP_LPAstar: {
          RAVELOG_INFO("Planning with LazySP using (non-lazy!) LPA* on lazy-weighted multi-grasp graph");
          typedef lpastar::LPAStarAlgorithm<MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>,
                                            lpastar::EdgeCostEvaluationType::Explicit>
              SearchAlgorithmType;
          lazysp::lazySP<MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>,
                         lazysp::LastUnknownEdgeSelector, SearchAlgorithmType>(graph, sr);
          break;
        }
        default: {
          RAVELOG_ERROR("Algorithm type " + getName(_params.algo_type) +
                        " doesn't support LazyWeightedMultiGraspGraph");
        }
      }
      if (sr.solved)
      {
        extractSolution<MultiGraspRoadmapGraph<CostCheckingType::VertexEdgeWithoutGrasp>>(sr, sol, graph);
      }
      break;
    }
    case GraphType::LazyEdgeWeightedMultiGraspGraph: {
      MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp> graph(_roadmap, _goal_set, cost_parameters, grasp_ids,
                                                                       start_id);
      SearchResult sr;
      switch (_params.algo_type)
      {
        case AlgorithmType::LazySP_LLPAstar: {
          RAVELOG_INFO("Planning with LazySP using lazy LPA* on lazy-edge-weighted multi-grasp graph");
          // TODO this is equivalent to LazySP on normal multi-grasp graph (apart from Edge selector)
          typedef lpastar::LPAStarAlgorithm<MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>,
                                            lpastar::EdgeCostEvaluationType::Lazy>
              SearchAlgorithmType;
          lazysp::lazySP<MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>, lazysp::LastUnknownEdgeSelector,
                         SearchAlgorithmType>(graph, sr);
          break;
        }
        case AlgorithmType::LazySP_LWLPAstar: {
          RAVELOG_INFO("Planning with LazySP using lazy weighted LPA* on lazy-edge-weighted multi-grasp graph");
          typedef lpastar::LPAStarAlgorithm<MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>,
                                            lpastar::EdgeCostEvaluationType::LazyWeighted>
              SearchAlgorithmType;
          lazysp::lazySP<MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>, lazysp::LastUnknownEdgeSelector,
                         SearchAlgorithmType>(graph, sr);
          break;
        }
        case AlgorithmType::LazySP_LPAstar: {
          RAVELOG_INFO("Planning with LazySP using (non-lazy!) LPA* on lazy-edge-weighted multi-grasp graph");
          typedef lpastar::LPAStarAlgorithm<MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>,
                                            lpastar::EdgeCostEvaluationType::Explicit>
              SearchAlgorithmType;
          lazysp::lazySP<MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>, lazysp::LastUnknownEdgeSelector,
                         SearchAlgorithmType>(graph, sr);
          break;
        }
        default: {
          RAVELOG_ERROR("Algorithm type " + getName(_params.algo_type) +
                        " doesn't support LazyWeightedMultiGraspGraph");
        }
      }
      if (sr.solved)
      {
        extractSolution<MultiGraspRoadmapGraph<CostCheckingType::EdgeWithoutGrasp>>(sr, sol, graph);
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

void MGGraphSearchMP::addWaypoints(const std::vector<Config>& waypoints)
{
  for (auto& config : waypoints)
  {
    _roadmap->addNode(config);
  }
}