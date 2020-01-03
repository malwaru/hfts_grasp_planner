#include <hfts_grasp_planner/placement/mp/GraphSearch.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Algorithms.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>
#include <queue>

namespace mg = ::placement::mp::mgsearch;
using namespace ::placement::mp;

Astar::Astar(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id,
    const Parameters& iparams)
    : _env(penv)
    , params(iparams)
{
    RAVELOG_DEBUG("Astar constructor!");
    _scene_interface = std::make_shared<mg::ORStateSpace>(penv, robot_id, obj_id);
    _robot = penv->GetRobot(penv->GetBodyFromEnvironmentId(robot_id)->GetName());
    // create roadmap
    auto edge_computer = std::make_shared<mg::IntegralEdgeCostComputer>(_scene_interface);
    _roadmap = std::make_shared<mg::Roadmap>(_scene_interface, edge_computer);
    // add start node
    Config start_config;
    _robot->GetActiveDOFValues(start_config);
    _start_node = _roadmap->addNode(start_config);
}

Astar::~Astar()
{
    // nothing to do
}

void Astar::plan(std::vector<Solution>& new_paths, double time_limit)
{
    // TODO create goal hierarchy, create graph, execute A* search
    // create goal distance function, for this collect all goals in a vector first
    // TODO this is a waste of resources to copy goals into a vector
    std::vector<MultiGraspMP::Goal> goals;
    for (auto elem : _goals) {
        goals.push_back(elem.second);
    }
    auto goal_distance_fn = std::make_shared<mg::MGGoalDistance>(goals, mg::cSpaceDistance, params.lambda);
    // create the graph
    switch (params.graph_type) {
    case GraphType::SingleGraspGraph: {
        // solve the problem for each grasp separately
        for (auto grasp_id : _grasp_ids) {
            RAVELOG_DEBUG("Planning with A* on single grasp graph for grasp " + std::to_string(grasp_id));
            assert(not _start_node.expired());
            unsigned int start_id = _start_node.lock()->uid;
            mg::SingleGraspRoadmapGraph graph(_roadmap, goal_distance_fn, grasp_id, start_id);
            mg::SearchResult sr;
            mg::astar::aStarSearch<mg::SingleGraspRoadmapGraph>(graph, sr);
            if (sr.solved) {
                WaypointPathPtr wp_path = std::make_shared<WaypointPath>();
                // extract solution path
                for (unsigned int vid : sr.path) {
                    auto node = _roadmap->getNode(vid);
                    assert(node);
                    wp_path->push_back(node->config);
                }
                // get goal id
                auto goal_node = _roadmap->getNode(sr.path.back());
                assert(goal_node);
                unsigned int gid = goal_node->goal_id;
                // get overall cost
                double overall_cost = sr.path_cost + params.lambda * goal_distance_fn->goalCost(gid);
                new_paths.push_back(Solution(gid, wp_path, overall_cost));
            }
        }
        break;
    }
    default:
        throw std::logic_error("Not implemented yet");
        // TODO
    }
}

void Astar::pausePlanning()
{
    // nothing to do
}

void Astar::addGrasp(const Grasp& grasp)
{
    _grasp_ids.insert(grasp.id);
    _scene_interface->addGrasp(grasp);
}

void Astar::addGoal(const Goal& goal)
{
    _goals.insert(std::make_pair(goal.id, goal));
    auto goal_node = _roadmap->addGoalNode(goal);
    _goal_nodes.insert(std::make_pair(goal.id, goal_node));
}

void Astar::removeGoals(const std::vector<unsigned int>& goal_ids)
{
    for (unsigned int gid : goal_ids) {
        auto iter = _goals.find(gid);
        if (iter != _goals.end()) {
            // remove goal node if it has been added to the roadmap
            auto goal_node_iter = _goal_nodes.find(gid);
            if (goal_node_iter != _goal_nodes.end()) {
                auto goal_node = goal_node_iter->second.lock();
                goal_node->is_goal = false;
                _goal_nodes.erase(goal_node_iter);
            }
        }
        _goals.erase(iter);
    }
}
