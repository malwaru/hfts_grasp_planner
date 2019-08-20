#pragma once
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Interfaces.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/ORCostsAndValidity.h>
#include <set>

namespace placement {
namespace mp {
    class Astar : public MultiGraspMP {
    public:
        Astar(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id);
        ~Astar();

        void plan(std::vector<std::pair<unsigned int, WaypointPathPtr>>& new_paths, double time_limit) override;
        void pausePlanning() override;
        void addGrasp(const Grasp& grasp) override;
        void addGoal(const Goal& goal) override;
        void removeGoals(const std::vector<unsigned int>& goal_ids) override;

    private:
        OpenRAVE::EnvironmentBasePtr _env;
        OpenRAVE::RobotBasePtr _robot;
        unsigned int _robot_id;
        unsigned int _obj_id;
        mgsearch::ORSceneInterfacePtr _scene_interface;
        mgsearch::MGGoalDistancePtr _goal_distance;
        mgsearch::RoadmapPtr _roadmap;
        std::set<unsigned int> _grasp_ids;
        std::unordered_map<unsigned int, Goal> _goals;
        mgsearch::Roadmap::NodePtr _start_node;
        std::unordered_map<unsigned int, mgsearch::Roadmap::NodePtr> _goal_nodes;
    };
    typedef std::shared_ptr<Astar> AstarPtr;
    typedef std::shared_ptr<const Astar> AstarConstPtr;
}
}