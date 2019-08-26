#pragma once

#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <vector>

namespace placement {
namespace mp {
    namespace mgsearch {
        /**
         * Type interface for grasp agnostic graphs. Instead of inheriting from this class,
         * simply implement the same interface. All graph search algorithms that use a grasp agnostic graph,
         * use this interface.
         */
        class GraspAgnosticGraph {
        public:
            bool checkValidity(unsigned int v) const;
            void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
            void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
            double heuristic(unsigned int v) const;
            double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
            unsigned int getStartNode() const;
            bool isGoal(unsigned int v) const;
        };

        class GraspAwareGraph {
            // TODO
        };

        /**
         * A class implementing the GraspAgnostic graph interface for a single grasp.
         */
        class SingleGraspRoadmapGraph {
        public:
            SingleGraspRoadmapGraph(RoadmapPtr roadmap, unsigned int grasp_id);
            ~SingleGraspRoadmapGraph();
            // needs to be set before planning!
            void setStartId(unsigned int start_id);
            // GraspAgnostic graph interface
            bool checkValidity(unsigned int v) const;
            void getSuccessors(unsigned int v, std::vector<unsigned int>& successors) const;
            void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors) const;
            double heuristic(unsigned int v) const;
            double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
            unsigned int getStartNode() const;
            bool isGoal(unsigned int v) const;

        private:
            RoadmapPtr _roadmap;
            const unsigned int _grasp_id;
            unsigned int _start_id;
            bool _has_goal;
        };

        class MultiGraspRoadmapGraph {
        };
    }
}
}
}