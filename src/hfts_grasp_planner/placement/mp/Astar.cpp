#include <hfts_grasp_planner/placement/mp/Astar.h>
#include <queue>

using namespace placement::mp;
using namespace mgsearch;

// struct AstarNode {
//     Roadmap::NodeWeakPtr roadmap_node;
//     // base f score assuming no particular grasp
//     double base_f_score;
//     // parent for this f score
//     std::weak_ptr<AstarNode> base_parent;
//     // maps grasp to f score
//     std::unordered_map<unsigned int, double> f_scores;
//     // maps grasp to respective parent
//     std::unordered_map<unsigned int, std::weak_ptr<AstarNode>> parents;
//     // constructor
//     AstarNode(Roadmap::NodePtr rnode, double dv = 0.0, std::shared_ptr<AstarNode> p = nullptr)
//         : roadmap_node(rnode)
//         , base_f_score(0.0)
//         , base_parent(p)
//     {
//     }

//     double getFScore(unsigned int gid) const
//     {
//         auto iter = f_scores.find(gid);
//         if (iter == f_scores.end()) {
//             return base_f_score;
//         }
//         return iter->second;
//     }

//     void setFScore(unsigned int gid, double val)
//     {
//         auto iter = f_scores.find(gid);
//         if (iter == f_scores.end()) {
//             f_scores.insert(std::make_pair(gid, val));
//         } else {
//             f_scores[gid] = val;
//         }
//     }
// };

// typedef std::shared_ptr<AstarNode> AstarNodePtr;

Astar::Astar(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id)
    : _env(penv)
{
    RAVELOG_DEBUG("Astar constructor!");
    _scene_interface = std::make_shared<ORSceneInterface>(penv, robot_id, obj_id);
    _goal_distance = std::make_shared<MGGoalDistance>();
    _robot = penv->GetRobot(penv->GetBodyFromEnvironmentId(robot_id)->GetName());
    // create roadmap
    Roadmap::SpaceInformation si;
    si.dimension = _robot->GetActiveDOF();
    _robot->GetActiveDOFLimits(si.lower, si.upper);
    si.distance_fn = cSpaceDistance;
    _roadmap = std::make_shared<Roadmap>(si, _scene_interface, _scene_interface);
    // add start node
    Config start_config;
    _robot->GetActiveDOFValues(start_config);
    _start_node = _roadmap->addNode(start_config);
}

Astar::~Astar()
{
    // nothing to do
}

// AstarNodePtr getAstarNode(std::unordered_map<unsigned int, AstarNodePtr>& all_nodes, unsigned int uid)
// {
//     auto iter = all_nodes.find(uid);
//     if (iter == all_nodes.end()) {
//         throw std::logic_error("Could not retrieve AstarNode for node " + std::to_string(uid));
//     }
//     return iter->second;
// }

void Astar::plan(std::vector<std::pair<unsigned int, WaypointPathPtr>>& new_paths, double time_limit)
{
    // if (_goals.size() == 0)
    //     return;
    // RAVELOG_INFO("Astar plan!");
    // // run lazy A* for multiple grasps, define PQ first (f, gid, edge, node_info)
    // typedef std::tuple<double, unsigned int, Roadmap::EdgeWeakPtr, AstarNodePtr> PQElement;
    // std::priority_queue<PQElement, std::vector<PQElement>, std::greater> pq;
    // // for each rodmap node we store an AstarNode that contains algorithm related information
    // std::unordered_map<unsigned int, AstarNodePtr> all_nodes;
    // all_nodes.insert(std::make_pair(_start_node->uid, std::make_shared<AstarNode>(_start_node)));
    // // add a job from the start node for each grasp
    // for (auto gid : _grasp_ids) {
    //     pq.push(std::make_tuple(0.0, gid, nullptr, all_nodes.at(0)));
    // }
    // // start the algorithm
    // AstarNodePtr current_node = nullptr;
    // bool solution_found = _start_node->is_goal;
    // // TODO this implementation is becoming pretty ugly
    // while (not pq.empty() and not solution_found) {
    //     const PQElement& cur_pq_elem = pq.top();
    //     double f_score = std::get<0>(cur_pq_elem);
    //     unsigned int gid = std::get<1>(cur_pq_elem);
    //     Roadmap::EdgeWeakPtr wedge = std::get<2>(cur_pq_elem);
    //     current_node = std::get<3>(cur_pq_elem);
    //     pq.pop();
    //     // first check validity of the node
    //     if (not _roadmap->checkNode(current_node->roadmap_node))
    //         continue;
    //     // next check validity of the inbound edge
    //     if (f_score > 0.0) // not a start node
    //     {
    //         if (not _roadmap->checkEdge(wedge))
    //             continue;
    //         // compute true cost of edge
    //         auto edge = wedge.lock();
    //         auto [evalid, ecost] = _roadmap->computeCost(edge, gid);
    //         if (not evalid)
    //             continue;
    //         // compute true f score
    //         auto roadmap_parent = edge.lock()->getNeighbor(current_node->roadmap_node.lock());
    //         auto parent = getAstarNode(all_nodes, roadmap_parent->uid);
    //         f_score = parent->getFScore(gid) + ecost;
    //         //  + _goal_distance->costToGo(current_node->roadmap_node.lock()->config, gid);
    //     }
    //     // current_node->f_scores
    //     current_node->setFScore(gid, f_score);
    //     // TODO run over neighbors and update accordingly
    // }
    // if (solution_found) {
    //     // TODO extract solution
    // }
    // // TODO implement Astar that computes costs for all grasps at the same time
}

void Astar::pausePlanning()
{
    // nothing to do
}

void Astar::addGrasp(const Grasp& grasp)
{
    _scene_interface->addGrasp(grasp);
}

void Astar::addGoal(const Goal& goal)
{
    _goal_distance->addGoal(goal);
    _goals.insert(std::make_pair(goal.id, goal));
    auto goal_node = _roadmap->addGoalNode(goal);
    _goal_nodes.insert(std::make_pair(goal.id, goal_node));
}

void Astar::removeGoals(const std::vector<unsigned int>& goal_ids)
{
    for (unsigned int gid : goal_ids) {
        auto iter = _goals.find(gid);
        if (iter != _goals.end()) {
            _goal_distance->removeGoal(iter->second);
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
