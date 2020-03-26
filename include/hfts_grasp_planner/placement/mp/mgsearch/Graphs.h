#pragma once

#include <boost/functional/hash.hpp>
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
            /**
             * Check the validity of v.
             */
            bool checkValidity(unsigned int v) const;
            /**
             * Return all successor nodes of the node v.
             * @param v - node id to return successor node for
             * @param successors - vector to store successors in
             * @lazy - if true, no cost evaluation is performed, meaning that a returned successor u may in fact
             *      not be reachable through v, i.e. cost(v, u) could be infinity.
             *      if false, the true cost cost(v, u) is computed first and only us with finite cost are returned.
             * TODO implement function that returns an iterator instead
             */
            void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
            /**
             * Just like getSuccessors but predecessors. In case of a directed graph, identical to getSuccessors.
             */
            void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
            /**
             * Get a cost for the edge from v1 to v2. Optionally, a lower bound of the cost.
             * @param v1 - id of first node
             * @param v2 - id of second node
             * @param lazy - if true, return only the best known lower bound cost of the edge, else compute true cost
             */
            double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
            /**
             * Return the id of the start node.
             */
            unsigned int getStartNode() const;
            /**
             * Return whether the node with the given id is a goal.
             */
            bool isGoal(unsigned int v) const;

            // Technically not a function of the graph, but the graph might have its own encoding of vertices, so the
            // heuristic needs to be connected to the graph anyways.
            double heuristic(unsigned int v) const;
        };

        // class GraspAwareGraph {
        //     // TODO
        // };

        /**
         * The SingleGraspRoadmapGraph class implements a view on a MultiGraspRoadmap for a single grasp.
         * that implements the GraspAgnostic graph interface.
         */
        class SingleGraspRoadmapGraph {
        public:
            /**
             * Create a new roadmap graph defined by the given roadmap for the given grasp.
             * @param roadmap - roadmap to use
             * @param goal_set - goal set
             * @param cost_to_go - cost-to-go-heuristic
             * @param grasp_id - the id of the grasp
             * @param start_id - the id of the roadmap node that defines the start node
             */
            SingleGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
                ::placement::mp::mgsearch::CostToGoHeuristicPtr cost_to_go,
                unsigned int grasp_id,
                unsigned int start_id);
            ~SingleGraspRoadmapGraph();
            // GraspAgnostic graph interface
            bool checkValidity(unsigned int v) const;
            void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
            void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
            double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
            unsigned int getStartNode() const;
            bool isGoal(unsigned int v) const;
            double heuristic(unsigned int v) const;

            std::pair<uint, uint> getGraspRoadmapId(uint vid) const;

        private:
            ::placement::mp::mgsearch::RoadmapPtr _roadmap;
            ::placement::mp::mgsearch::MultiGraspGoalSetPtr _goal_set;
            ::placement::mp::mgsearch::CostToGoHeuristicPtr _cost_to_go;
            const unsigned int _grasp_id;
            const unsigned int _start_id;
        };

        /**
         * The MultiGraspRoadmapGraph class implements a view on a MultiGraspRoadmap for multiple grasps, and
         * implements the GraspAgnostic graph interface.
         * The start vertex of this graph is a special vertex that is not associated with any grasp yet.
         * It is adjacent with cost 0 to #grasps vertices associated with the start configuration - one for each grasp.
         */
        class MultiGraspRoadmapGraph {
        public:
            /**
             * Create a new MultiGraspRoadmapGraph defined by the given roadmap for the given grasps.
             * @param roadmap - roadmap to use
             * @param cost_to_go - cost-to-go-heuristic
             * @param grasp_ids - the ids of the grasps
             * @param start_id - the id of the roadmap node that defines the start node
             */
            MultiGraspRoadmapGraph(::placement::mp::mgsearch::RoadmapPtr roadmap,
                ::placement::mp::mgsearch::MultiGraspGoalSetPtr goal_set,
                ::placement::mp::mgsearch::CostToGoHeuristicPtr cost_to_go,
                const std::set<unsigned int>& grasp_ids,
                unsigned int start_id);
            ~MultiGraspRoadmapGraph();
            // GraspAgnostic graph interface
            bool checkValidity(unsigned int v) const;
            void getSuccessors(unsigned int v, std::vector<unsigned int>& successors, bool lazy = false) const;
            void getPredecessors(unsigned int v, std::vector<unsigned int>& predecessors, bool lazy = false) const;
            double getEdgeCost(unsigned int v1, unsigned int v2, bool lazy = false) const;
            unsigned int getStartNode() const;
            bool isGoal(unsigned int v) const;
            double heuristic(unsigned int v) const;

            std::pair<uint, uint> getGraspRoadmapId(uint vid) const;
        private:
            ::placement::mp::mgsearch::RoadmapPtr _roadmap;
            ::placement::mp::mgsearch::MultiGraspGoalSetPtr _goal_set;
            ::placement::mp::mgsearch::CostToGoHeuristicPtr _cost_to_go;
            const std::set<unsigned int> _grasp_ids;
            // hash table mapping (grasp_id, roadmap_id) to graph id
            typedef std::pair<unsigned int, unsigned int> GraspNodeIDPair;
            mutable std::unordered_map<GraspNodeIDPair, unsigned int, boost::hash<GraspNodeIDPair>> _roadmap_key_to_graph;
            // hash table mapping graph id to (grasp_id, roadmap_id)
            mutable std::unordered_map<unsigned int, GraspNodeIDPair> _graph_key_to_roadmap;
            unsigned int _roadmap_start_id;

            std::pair<unsigned int, unsigned int> toRoadmapKey(unsigned int graph_id) const;
            unsigned int toGraphKey(const std::pair<unsigned int, unsigned int>& roadmap_id) const;
            unsigned int toGraphKey(unsigned int grasp_id, unsigned int roadmap_id) const;
            mutable unsigned int _num_graph_nodes;
        };
    }
}
}