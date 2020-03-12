#include <hfts_grasp_planner/placement/mp/mgsearch/Algorithms.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MGGraphSearchMP.h>

using namespace placement::mp::mgsearch;

MGGraphSearchMP::MGGraphSearchMP(mgsearch::StateSpacePtr state_space, const Config& start_config, const Parameters& params)
    : _state_space(state_space)
    , _params(params)
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
    // create the graph
    switch (_params.graph_type) {
    case GraphType::SingleGraspGraph: {
        // solve the problem for each grasp separately
        // first determine the unique number of grasps we have
        std::set<unsigned int> grasp_ids;
        std::vector<MultiGraspMP::Goal> goals;
        _goal_set->getGoals(goals);
        for (auto goal : goals) {
            grasp_ids.insert(goal.grasp_id);
        }
        // now solve the problem for each grasp separately using the specified algorithm
        for (auto grasp_id : grasp_ids) {
            assert(not _start_node.expired());
            unsigned int start_id = _start_node.lock()->uid;
            SingleGraspRoadmapGraph graph(_roadmap, _goal_set, goal_distance_fn, grasp_id, start_id);
            SearchResult sr;
            switch (_params.algo_type) {
            case AlgorithmType::Astar: {
                RAVELOG_DEBUG("Planning with A* on single grasp graph for grasp " + std::to_string(grasp_id));
                astar::aStarSearch<SingleGraspRoadmapGraph>(graph, sr);
                break;
            }
            default:
                RAVELOG_ERROR("Algorithm type not implemented yet");
            }
            if (sr.solved) {
                // TODO we will need the same path extraction for MultiGraspGraphs -> refactor code
                MultiGraspMP::WaypointPathPtr wp_path = std::make_shared<MultiGraspMP::WaypointPath>();
                // extract solution path
                for (unsigned int vid : sr.path) {
                    auto node = _roadmap->getNode(vid);
                    assert(node);
                    wp_path->push_back(node->config);
                }
                // get goal id
                auto goal_node = _roadmap->getNode(sr.path.back()); // TODO this assumes that vertex ids are equal to roadmap ids
                assert(goal_node);
                auto [goal_id, valid_goal] = _goal_set->getGoalId(goal_node->uid, grasp_id);
                assert(valid_goal);
                auto goal = _goal_set->getGoal(goal_id);
                // get overall cost
                double overall_cost = sr.path_cost + _params.lambda * goal_distance_fn->qualityToCost(goal.quality);
                if (overall_cost < sol.cost) {
                    sol.goal_id = goal_id;
                    sol.path = wp_path;
                    sol.cost = overall_cost;
                }
            }
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
