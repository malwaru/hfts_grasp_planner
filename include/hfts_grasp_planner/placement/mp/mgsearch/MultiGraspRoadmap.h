#pragma once
// stl
#include <functional>
#include <memory>
#include <unordered_map>
// own
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
// ompl
#include <ompl/datastructures/NearestNeighborsGNAT.h>

namespace placement {
namespace mp {
    namespace mgsearch {
        // Interfaces used by Roadmap
        // TODO make these to template arguments? Will we ever exchange these during runtime?
        class StateValidityChecker {
        public:
            virtual ~StateValidityChecker() = 0;
            virtual bool isValid(const Config& c) const = 0;
            virtual bool isValid(const Config& c, unsigned int grasp_id, bool only_obj = false) const = 0;
        };
        typedef std::shared_ptr<StateValidityChecker> StateValidityCheckerPtr;

        class EdgeCostComputer {
        public:
            virtual ~EdgeCostComputer() = 0;
            /**
             * Cheap to compute lower bound of the cost to transition from config a to config b.
             */
            virtual double lowerBound(const Config& a, const Config& b) const = 0;
            /**
             * True cost to transition from config a to config b without any grasped object.
             */
            virtual double cost(const Config& a, const Config& b) const = 0;
            /**
             * True cost to transition from config a to config b when grasping an object with grasp grasp_id.
             */
            virtual double cost(const Config& a, const Config& b, unsigned int grasp_id) const = 0;
        };
        typedef std::shared_ptr<EdgeCostComputer> EdgeCostComputerPtr;

        class CostToGoHeuristic {
            // TODO do we need this to be a class? Does it have an internal state?
        public:
            virtual ~CostToGoHeuristic() = 0;
            virtual double costToGo(const Config& a) const = 0;
            virtual double costToGo(const Config& a, unsigned int grasp_id) const = 0;
        };
        typedef std::shared_ptr<CostToGoHeuristic> CostToGoHeuristicPtr;

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
                typedef std::unordered_map<unsigned int, EdgePtr> EdgeMap;
                // unique node id
                const unsigned int uid;
                // Configuration represented by this node
                const Config config;
                // goal annotation
                bool is_goal;
                unsigned int goal_id;
                /**
                 * Return the edge that leads to the node with target_id.  
                 * If the specified node is not adjacent, nullptr is returned.
                 */
                EdgePtr getEdge(unsigned int target_id)
                {
                    auto iter = edges.find(target_id);
                    if (iter == edges.end())
                        return nullptr;
                    return iter->second;
                }

                std::pair<EdgeMap::const_local_iterator, EdgeMap::const_local_iterator> getEdgesIterators() const
                {
                    return std::make_pair(edges.cbegin(), edges.cend());
                }

            protected:
                friend class Roadmap;
                // 0 = edges not initialized, >= 1 - last densification generation edges have been updated
                unsigned int densification_gen;
                bool initialized; // initialized = collision-free
                // map node id to edge
                EdgeMap edges;
                // Constructor
                Node(unsigned int tuid, const Config& tconfig)
                    : uid(tuid)
                    , initialized(false)
                    , config(tconfig)
                    , densification_gen(0)
                {
                }
            };

            struct Edge {
                double base_cost;
                // flag whether base_cost is true base cost or just a lower bound
                bool base_evaluated;
                // maps grasp id to a cost
                std::unordered_map<unsigned int, double> conditional_costs;
                NodeWeakPtr node_a;
                NodeWeakPtr node_b;
                Edge(NodePtr a, NodePtr b, double bc);
                // Convenience function returning the node that isn't n
                NodePtr getNeighbor(NodePtr n) const;
                // convenience function to return the best known approximate this edge's cost for the given grasp
                double getBestKnownCost(unsigned int gid) const;
            };

            struct SpaceInformation {
                unsigned int dimension;
                std::vector<double> lower;
                std::vector<double> upper;
                std::function<double(const Config&, const Config&)> distance_fn;
            };

            Roadmap(const SpaceInformation& si, StateValidityCheckerPtr validity_checker,
                EdgeCostComputerPtr edge_cost_computer, unsigned int batch_size = 10000);
            virtual ~Roadmap();
            // Tell the roadmap to densify
            void densify();
            void densify(unsigned int batch_size);
            /**
             * Retrieve a node.
             * Returns nullptr if node with given id doesn't exist.
             */
            NodePtr getNode(unsigned int node_id) const;

            /**
             * Add a new goal node.
             * @param goal - goal information
             * @return node - a node representing the goal
             *  (node->is_goal = true, node->goal_id == goal.id, node->config == goal.config)
             *  The returned pointer is weak. To check validity of the node use checkNode.
             */
            NodeWeakPtr addGoalNode(const MultiGraspMP::Goal& goal);
            /**
             * Add a new node at the given configuration.
             * Use this function to add the start node.
             */
            NodeWeakPtr addNode(const Config& config);
            /**
             * Check the given node for validity, and update roadmap if necessary.
             * In addition, update the node's adjacency list.
             * @param node - the node to check
             * @return true, if the node is valid (base), else false. In case of false, the node is removed
             *  from the roadmap and node is set to nullptr.
             *  If this function returned true, you can safely acquire a lock on node, else node is no longer valid.
             */
            bool checkNode(NodeWeakPtr node);
            /**
             * Just like checkNode, but the return value indicates whether the node is valid for the given grasp.
             * The node is of course only removed if the base is invalid, not if the collision is induced by the grasp.
             * @return true, if the node is valid and not in collision for the given grasp, else false.
             */
            bool checkNode(NodeWeakPtr node, unsigned int grasp_id);
            /**
             * Compute the base cost of the given edge (for no grasp).
             * If the edge is found to be invalid, the edge is removed from the roadmap.
             * In this case, if you called computeCost(EdgeWeakPtr edge), edge will be expired after the function returns.
             * If you called computeCost(EdgePtr edge)
             * 
             */
            std::pair<bool, double> computeCost(EdgePtr edge);
            std::pair<bool, double> computeCost(EdgeWeakPtr edge);
            /**
             * Compute the edge cost for the given edge given the grasp.
             * @param edge - the edge to compute cost for
             * @param grasp_id - the grasp id to compute the cost for
             * @return pair (valid, cost) - valid = true if cost is finite
             */
            std::pair<bool, double> computeCost(EdgePtr edge, unsigned int grasp_id);

        private:
            SpaceInformation _si;
            StateValidityCheckerPtr _validity_checker;
            EdgeCostComputerPtr _cost_computer;
            ::ompl::NearestNeighborsGNAT<NodePtr> _nn; // owner of nodes
            std::unordered_map<unsigned int, NodeWeakPtr> _nodes; // node id to pointer
            unsigned int _batch_size;
            unsigned int _node_id_counter;
            unsigned int _halton_seq_id;
            unsigned int _densification_gen;
            double _gamma_prm;

            void scaleToLimits(Config& config) const;
            void deleteNode(NodePtr node);
            void deleteEdge(EdgePtr edge);
        };
        typedef std::shared_ptr<Roadmap> RoadmapPtr;

        class MGGoalDistance : public CostToGoHeuristic {
        public:
            /**
             * Construct a new multi-grasp cost-to-go function.
             * The cost-to-go function expresses the term 
             * h(q) = min_{g in G} (d(q, g) + lambda * cost(g)), where g in G are the goals, d(q_1, q_2) a lower bound on path cost.
             * The cost of a goal cost(g) is computed as cost(g) = (o_max - o_g) / (o_max - o_min) where o_g denotes
             * the goal's quality and o_max = max_{g in G} o_g, o_min = min_{g in G} o_g (larger qualities are better).
             * The parameter lambda scales between the grasp cost, which is in range [0, 1], and the path cost d(q1, q2).
             * Note:
             *   For new goals, you need to construct a new instance, due to the fact that goal quality values are 
             *   normalized w.r.t min and max quality.
             * 
             * @param goals - list of goals 
             * @param path_cost - lower bound on path cost to move from one configuration to another
             * @param lambda - parameter to scale between path cost and grasp cost
             */
            MGGoalDistance(const std::vector<const MultiGraspMP::Goal> goals,
                std::function<double(const Config&, const Config&)> path_cost, double lambda);
            ~MGGoalDistance();
            // interface functions
            double costToGo(const Config& a) const override;
            double costToGo(const Config& a, unsigned int grasp_id) const override;

        private:
            struct GoalDistanceFn {
                double scaled_lambda;
                std::function<double(const Config&, const Config&)> path_cost;
                double distance(const MultiGraspMP::Goal& ga, const MultiGraspMP::Goal& gb)
                {
                    return path_cost(ga.config, gb.config) + scaled_lambda * abs(ga.quality - gb.quality);
                }
            };
            // grasp id -> gnat per grasp
            std::unordered_map<unsigned int, std::shared_ptr<::ompl::NearestNeighborsGNAT<MultiGraspMP::Goal>>> _goals;
            ::ompl::NearestNeighborsGNAT<MultiGraspMP::Goal> _all_goals;
            GoalDistanceFn _goal_distance;
            double _max_quality;
        };
        typedef std::shared_ptr<MGGoalDistance> MGGoalDistancePtr;
    }
}
}