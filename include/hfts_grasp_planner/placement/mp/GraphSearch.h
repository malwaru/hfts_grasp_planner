#pragma once
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/ORCostsAndValidity.h>
#include <set>

namespace placement {
namespace mp {
    class Astar : public MultiGraspMP {
    public:
        enum GraphType {
            SingleGraspGraph,
            MultiGraspGraph
        };
        // enum AlgorithmType {
        //     Astar,
        //     LWAstar,
        //     LazySP_LPAstar,
        //     LazySP_MultiGraspLPAstar // only makes sense on MultiGraspGraph
        // };
        struct Parameters {
            GraphType graph_type;
            double lambda; // weight between
            bool extreme_lazy; // only for LazySP_MultiGraspLPAstar
        };

        Astar(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id,
            const Parameters& params);
        ~Astar();

        void plan(std::vector<Solution>& new_paths, double time_limit) override;
        void pausePlanning() override;
        void addGrasp(const Grasp& grasp) override;
        void addGoal(const Goal& goal) override;
        void removeGoals(const std::vector<unsigned int>& goal_ids) override;

        Parameters params;

    private:
        OpenRAVE::EnvironmentBasePtr _env;
        OpenRAVE::RobotBasePtr _robot;
        unsigned int _robot_id;
        unsigned int _obj_id;
        mgsearch::ORSceneInterfacePtr _scene_interface;
        mgsearch::RoadmapPtr _roadmap;
        std::set<unsigned int> _grasp_ids;
        // goal id -> goal
        std::unordered_map<unsigned int, Goal> _goals;
        mgsearch::Roadmap::NodeWeakPtr _start_node;
        // goal id -> goal node
        std::unordered_map<unsigned int, mgsearch::Roadmap::NodeWeakPtr> _goal_nodes;
    };
    typedef std::shared_ptr<Astar> AstarPtr;
    typedef std::shared_ptr<const Astar> AstarConstPtr;
}
}