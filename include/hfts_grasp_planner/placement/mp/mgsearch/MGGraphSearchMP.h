#pragma once
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Graphs.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/Algorithms.h>
#include <set>
#include <vector>

namespace placement {
namespace mp {
namespace mgsearch {
    class MGGraphSearchMP {
    public:
        enum GraphType {
            SingleGraspGraph = 0,
            MultiGraspGraph = 1
        };
        enum AlgorithmType {
            Astar = 0,
            LWAstar = 1, // lazy weighted A*
            LPAstar = 2, // life-long planning A*
            LazySP_LPAstar = 3, // Lazy SP using LPAstar
            LazySP_MultiGraspLPAstar = 4 // only makes sense on MultiGraspGraph
        };
        struct Parameters {
            AlgorithmType algo_type;
            GraphType graph_type;
            double lambda; // weight between
            bool extreme_lazy; // only for LazySP_MultiGraspLPAstar
            Parameters():
                algo_type(AlgorithmType::Astar),
                graph_type(GraphType::SingleGraspGraph),
                lambda(1.0),
                extreme_lazy(false)
            {
            }
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
        template<typename G>
        void extractSolution(SearchResult& sr, MultiGraspMP::Solution& sol, const G& graph) {
            MultiGraspMP::WaypointPathPtr wp_path = std::make_shared<MultiGraspMP::WaypointPath>();
            // extract solution path
            for (unsigned int vid : sr.path) {
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
            sol.cost = sr.path_cost;
        }
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
