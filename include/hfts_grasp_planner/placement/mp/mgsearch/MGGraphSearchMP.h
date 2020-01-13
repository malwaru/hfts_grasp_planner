#pragma once
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <set>
#include <vector>

namespace placement {
namespace mp {
namespace mgsearch {
    class MGGraphSearchMP {
    public:
        enum GraphType {
            SingleGraspGraph,
            MultiGraspGraph
        };
        enum AlgorithmType {
            Astar,
            LWAstar, // lazy weighted A*
            LPAstar, // life-long planning A*
            LazySP_LPAstar, // Lazy SP using LPAstar
            LazySP_MultiGraspLPAstar // only makes sense on MultiGraspGraph
        };
        struct Parameters {
            AlgorithmType algo_type;
            GraphType graph_type;
            double lambda; // weight between
            bool extreme_lazy; // only for LazySP_MultiGraspLPAstar
        };

        MGGraphSearchMP(mgsearch::StateSpacePtr state_space,
            const Config& start,
            const Parameters& params);
        ~MGGraphSearchMP();

        bool plan(MultiGraspMP::Solution& sol);
        void addGoal(const MultiGraspMP::Goal& goal);
        void removeGoals(const std::vector<unsigned int>& goal_ids);
        Parameters _params;

    private:
        mgsearch::StateSpacePtr _state_space;
        mgsearch::RoadmapPtr _roadmap;
        mgsearch::Roadmap::NodeWeakPtr _start_node;
        mgsearch::MultiGraspGoalSetPtr _goal_set;
    };
    typedef std::shared_ptr<MGGraphSearchMP> MGGraphSearchMPPtr;
    typedef std::shared_ptr<const MGGraphSearchMP> MGGraphSearchMPConstPtr;

}
}
}
