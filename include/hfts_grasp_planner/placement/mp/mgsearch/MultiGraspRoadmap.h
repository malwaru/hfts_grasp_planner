#pragma once
#include <hfts_grasp_planner/placement/mp/mgsearch/Interfaces.h>
#include <memory>
#include <ompl/datastructures/NearestNeighborsGNAT.h>
#include <unordered_map>

namespace placement {
namespace mp {
    namespace mgsearch {

        /**
         * This class encapsulates a conditional roadmap for a robot manipulator transporting 
         * an object. The roadmap is conditioned on a discrete set of grasps the robot may have 
         * on the object. The roadmap is constructed lazily.
         */
        class Roadmap {
        public:
            struct Node;
            struct Edge;
            typedef std::shared_ptr<Node> NodePtr;
            typedef std::weak_ptr<Node> NodeWeakPtr;
            typedef std::shared_ptr<Edge> EdgePtr;
            typedef std::weak_ptr<Edge> EdgeWeakPtr;

            struct Node {
                // unique node id
                const unsigned int uid;
                // Configuration represented by this node
                const Config config;
                // goal annotation
                bool is_goal;
                unsigned int goal_id;

            protected:
                friend class Roadmap;
                // 0 = edges not initialized, >= 1 - last densification generation edges have been updated
                unsigned int densification_gen;
                bool initialized; // initialized = collision-free
                // map node id to edge
                std::unordered_map<unsigned int, EdgePtr> edges;
                // Constructor
                Node(unsigned int tuid, const Config& tconfig)
                    : uid(tuid)
                    , initialized(false)
                    , config(tconfig)
                    , densification_gen(0)
                {
                }
                // TODO public means to iterate over neighbors
            };

            struct Edge {
                double base_cost;
                // flag whether base_cost is true base cost or just a lower bound
                bool base_evaluated;
                // maps grasp id to a cost
                std::unordered_map<unsigned int, double> conditional_costs;
                NodeWeakPtr node_a;
                NodeWeakPtr node_b;
                Edge(NodePtr a, NodePtr b);
            };

            Roadmap(OpenRAVE::RobotBasePtr robot, StateValidityCheckerPtr validity_checker,
                EdgeCostComputerPtr edge_cost_computer, unsigned int batch_size = 10000);
            virtual ~Roadmap();
            // Tell the roadmap to densify
            void densify();
            void densify(unsigned int batch_size);
            /**
             * Add a new goal node.
             * @param goal - goal information
             * @return node - a node representing the goal (node->is_goal = true, node->goal_id == goal.id, node->config == goal.config)
             */
            NodePtr addGoalNode(const MultiGraspMP::Goal& goal);
            /**
             * Add a new node at the given configuration.
             * Use this function to add the start node.
             */
            NodePtr addNode(const Config& config);
            /**
             * Check the given node for validity, and update roadmap if necessary.
             * In addition, update the node's adjacency list.
             * @param node - the node to check
             * @return true, if the node is valid (base), else false. In case of false, the node is removed
             * from the roadmap and node is set to nullptr.
             */
            bool checkNode(NodePtr node);
            /**
             * Check the given edge for validity, and update roadmap if necessary.
             * This means the base cost of the edge is computed.
             * @param edge - the edge to check
             * @return true, if the edge is valid (base), else false. In case of false, the edge is removed
             * from the roadmap and edge is set to nullptr.
             */
            bool checkEdge(EdgePtr edge);
            /**
             * Compute the edge cost for the given edge given the grasp.
             * @param edge - the edge to compute cost for
             * @param grasp_id - the grasp id to compute the cost for
             * @return true, if the edge cost is finite, return true, else false.
             */
            bool computeCost(EdgePtr edge, unsigned int grasp_id);

        private:
            OpenRAVE::RobotBasePtr _robot;
            StateValidityCheckerPtr _validity_checker;
            EdgeCostComputerPtr _cost_computer;
            ::ompl::NearestNeighborsGNAT<NodePtr> _nn;
            unsigned int _batch_size;
            unsigned int _node_id_counter;
            unsigned int _halton_seq_id;
            unsigned int _densification_gen;
            double _gamma_prm;

            void scaleToLimits(Config& config) const;
            void deleteNode(NodePtr node);
            void deleteEdge(EdgePtr edge);
        };
    }
}
}