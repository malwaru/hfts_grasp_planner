#pragma once

#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <vector>

namespace placement {
namespace mp {
    namespace mgsearch {
        /**
         * Type interface for grasp agnostic graphs. Instead of inheriting from this class,
         * simply implement the same interface. All graph search algorithms that use a grasp agnostic graph
         * use this interface.
         */
        class GraspAgnosticGraph {
        public:
            bool checkValidity(unsigned int v) const;
            void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
            void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
            double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
            unsigned int getStartNode() const;
            bool isGoal(unsigned int v) const;
            // Technically not a function of the graph, but the graph might have its own encoding of vertices, so the
            // heuristic needs to be connected to the graph anyways.
            double heuristic(unsigned int v) const;
        };

        // class GraspAwareGraph {
        //     // TODO
        // };

        /**
         * A class implementing the GraspAgnostic graph interface for a single grasp.
         */
        class SingleGraspRoadmapGraph {
        public:
            SingleGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                ::placement::mp::mgsearch::CostToGoHeuristicPtr cost_to_go,
                unsigned int grasp_id);
            ~SingleGraspRoadmapGraph();
            // needs to be set before planning!
            void setStartId(unsigned int start_id);
            // GraspAgnostic graph interface
            bool checkValidity(unsigned int v) const;
            void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
            void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
            double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
            unsigned int getStartNode() const;
            bool isGoal(unsigned int v) const;
            double heuristic(unsigned int v) const;

        private:
            ::placement::mp::mgsearch::RoadmapPtr _roadmap;
            ::placement::mp::mgsearch::CostToGoHeuristicPtr _cost_to_go;
            const unsigned int _grasp_id;
            unsigned int _start_id;
            bool _has_goal;
        };

        // class MultiGraspRoadmapGraph {
        // };
    }
}
}