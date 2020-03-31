#pragma once
#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>
#include <cmath>
#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>
#include <vector>

namespace placement {
namespace mp {
    namespace mgsearch {
        struct SearchResult {
            bool solved;
            std::vector<unsigned int> path; // starting from start node
            double path_cost;
        };

        namespace astar {
            struct PQElement {
                unsigned int v;
                double g_value;
                double h_value;
                PQElement(unsigned int _v, double _g, double _h)
                    : v(_v)
                    , g_value(_g)
                    , h_value(_h)
                {
                }
            };

            struct PQElementCompare {
                // for max heaps
                bool operator()(const PQElement& a, const PQElement& b) const
                {
                    double af = a.h_value + a.g_value;
                    double bf = b.h_value + b.g_value;
                    if (af == bf) { // tie breaker
                        return a.g_value < b.g_value;
                    }
                    return af > bf;
                }
            };

            // used for internal purposes
            template <typename PQ>
            struct VertexData {
                const unsigned int v;
                double g;
                unsigned int p;
                typename PQ::handle_type pq_handle;
                bool closed;
                VertexData(unsigned int _v, double _g, unsigned int _p)
                    : v(_v)
                    , g(_g)
                    , p(_p)
                {
                    closed = false;
                }
            };

            /**
             * A* search algorithm.
             * The template parameter G needs to be of a type implementing the GraspAgnosticGraph interface specified in Graphs.h.
             * The template parameter PQ needs to be a boost::heap
             */
            template <typename G, typename PQ = boost::heap::fibonacci_heap<PQElement, boost::heap::compare<PQElementCompare>>>
            void aStarSearch(const G& graph, SearchResult& result)
            {
                utils::ScopedProfiler("aStarSearch");
                // initialize result structure
                result.solved = false;
                result.path.clear();
                result.path_cost = INFINITY;
                // initialize algorithm data structures
                PQ pq;
                std::unordered_map<unsigned int, VertexData<PQ>> vertex_data;
                unsigned int v_start = graph.getStartNode();
                if (graph.checkValidity(v_start)) {
                    vertex_data.emplace(std::make_pair(v_start, VertexData<PQ>(v_start, 0.0, v_start)));
                    vertex_data.at(v_start).pq_handle = pq.push(PQElement(v_start, 0.0, graph.heuristic(v_start)));
                }
                // main iteration - is skipped if start vertex is invalid
                while (not pq.empty() and not result.solved) {
                    PQElement current_el = pq.top();
                    pq.pop();
                    vertex_data.at(current_el.v).closed = true;
                    if (graph.isGoal(current_el.v)) {
                        result.solved = true;
                        result.path_cost = vertex_data.at(current_el.v).g + current_el.h_value; // h_value includes potential goal cost
                        // extract path
                        unsigned int v = current_el.v;
                        while (v != v_start) {
                            result.path.push_back(v);
                            v = vertex_data.at(v).p;
                        }
                        result.path.push_back(v_start);
                        std::reverse(result.path.begin(), result.path.end());
                    } else {
                        // extend current_el.v
                        auto [siter, send] = graph.getSuccessors(current_el.v, true);
                        for (; siter != send; ++siter) {
                            uint s = *siter;
                            // check vertex and edge validity
                            double wvs = graph.getEdgeCost(current_el.v, s, false);
                            if (std::isinf(wvs)) {
                                continue;
                            }
                            // s is reachable from v. compute the g value it can reach.
                            double g_s = current_el.g_value + wvs;
                            // create a VertexData element if it doesn't exist yet.
                            auto iter = vertex_data.find(s);
                            if (iter != vertex_data.end()) {
                                if (iter->second.closed)
                                    continue;
                                // s has been reached from another node before, check whether we can decrease its key
                                if (iter->second.g > g_s) {
                                    iter->second.g = g_s;
                                    iter->second.p = current_el.v;
                                    (*(iter->second.pq_handle)).g_value = g_s;
                                    pq.increase(iter->second.pq_handle); // increase priority
                                }
                            } else {
                                // s hasn't been reached before, add a new VertexData element and push it to pq
                                auto [iter, valid] = vertex_data.emplace(std::make_pair(s, VertexData<PQ>(s, g_s, current_el.v)));
                                assert(valid);
                                iter->second.pq_handle = pq.push(PQElement(s, g_s, graph.heuristic(s)));
                            }
                        }
                    }
                }
            }
        } // astar namespace

        namespace lwastar {
            struct PQElement {
                /**
                 * Stores information that node v can be reached from node p with at least
                 * cost g_value. h_value stores h(v)
                 */
                const unsigned int v; // the node that can be reached
                const unsigned int p; // the parent
                double g_value; // g value that v can be reached with if going through p
                const double h_value; // h(v)
                PQElement(unsigned int _v, unsigned int _p, double _g, double _h)
                    : v(_v)
                    , p(_p)
                    , g_value(_g)
                    , h_value(_h)
                {
                }
            };

            inline double f(const PQElement& el)
            {
                return el.g_value + el.h_value;
            }

            struct PQElementCompare {
                // for max heaps
                bool operator()(const PQElement& a, const PQElement& b) const
                {
                    double af = f(a);
                    double bf = f(b);
                    if (af == bf) { // tie breaker
                        return a.g_value < b.g_value;
                    }
                    return af > bf;
                }
            };

            // used for internal purposes
            struct VertexData {
                /**
                 *  Stores whether a node is closed, its parent and its true g value.
                 */
                const unsigned int v;
                double g;
                unsigned int p;
                bool closed;
                VertexData(unsigned int _v, double _g, unsigned int _p)
                    : v(_v)
                    , g(_g)
                    , p(_p)
                {
                    closed = false;
                }
            };

            /**
             * LWA* search algorithm.
             * The template parameter G needs to be of a type implementing the GraspAgnosticGraph interface specified in Graphs.h.
             * The template parameter PQ needs to be a boost::heap // TODO we do not use an addressable PQ here, so check what the most efficient one is and use that
             */
            template <typename G, typename PQ = boost::heap::fibonacci_heap<PQElement, boost::heap::compare<PQElementCompare>>>
            void lwaStarSearch(const G& graph, SearchResult& result)
            {
                utils::ScopedProfiler("lwaStarSearch");
                // initialize result structure
                result.solved = false;
                result.path.clear();
                result.path_cost = INFINITY;
                // initialize algorithm data structures
                PQ pq;
                std::unordered_map<unsigned int, VertexData> vertex_data;
                unsigned int v_start = graph.getStartNode();
                if (graph.checkValidity(v_start)) {
                    vertex_data.emplace(std::make_pair(v_start, VertexData(v_start, 0.0, v_start)));
                    pq.push(PQElement(v_start, v_start, 0.0, graph.heuristic(v_start)));
                }
                // unsigned int v_goal = v_start; // best goal node
                // double f_v_goal = std::numeric_limits<double>::infinity();
                // main iteration - is skipped if start vertex is invalid
                // while (not pq.empty() && f(pq.top()) <= f_v_goal)
                while (not pq.empty() && not result.solved) {
                    PQElement current_el = pq.top();
                    pq.pop();
                    // check whether we already extended this node and check its validity; skip if necessary
                    if (vertex_data.at(current_el.v).closed or not graph.checkValidity(current_el.v)) {
                        vertex_data.at(current_el.v).closed = true;
                        continue;
                    }
                    // compute true edge cost
                    double true_edge_cost = current_el.p != current_el.v ? graph.getEdgeCost(current_el.p, current_el.v, false) : 0.0;
                    double true_g = vertex_data.at(current_el.p).g + true_edge_cost;
                    // skip if we already know a better path (includes the case that the edge is invalid)
                    // if (true_g > vertex_data.at(current_el.v).g)
                    if (std::isinf(true_edge_cost)) {
                        continue;
                    } else if (true_g > current_el.g_value) { // add element back to queue if cost is larger now
                        current_el.g_value = true_g;
                        pq.push(current_el);
                    } else { // we can extend v
                        assert(current_el.g_value == true_g);
                        vertex_data.at(current_el.v).closed = true;
                        vertex_data.at(current_el.v).g = true_g;
                        vertex_data.at(current_el.v).p = current_el.p;
                        if (graph.isGoal(current_el.v)) { // is it a goal?
                            // v_goal = current_el.v;
                            // f_v_goal = f(current_el);
                            result.solved = true;
                            result.path_cost = f(current_el);
                            // extract path
                            unsigned int v = current_el.v;
                            while (v != v_start) {
                                result.path.push_back(v);
                                v = vertex_data.at(v).p;
                            }
                            result.path.push_back(v_start);
                            std::reverse(result.path.begin(), result.path.end());
                        } else {
                            // actually extend v
                            auto [siter, send] = graph.getSuccessors(current_el.v, true);
                            for (; siter != send; ++siter) {
                                uint s = *siter;
                                // get lower bound of edge cost
                                double wvs = graph.getEdgeCost(current_el.v, s, true);
                                if (std::isinf(wvs)) { // skip edges that are already known to be invalid
                                    continue;
                                }
                                // compute the g value s might reach by going through v
                                double g_s = current_el.g_value + wvs;
                                // get VertexData
                                auto iter = vertex_data.find(s);
                                if (iter == vertex_data.end()) {
                                    // create a VertexData element if it doesn't exist yet.
                                    bool valid;
                                    std::tie(iter, valid) = vertex_data.emplace(std::make_pair(s, VertexData(s, g_s, current_el.v)));
                                    assert(valid);
                                } else if (iter->second.closed) {
                                    // if its closed, we can skip
                                    continue;
                                }
                                // in any case, add a new pq element representing the possibility to reach s from v
                                pq.push(PQElement(s, current_el.v, g_s, graph.heuristic(s)));
                            }
                        }
                    }
                }
            }
        } // lwastar namespace
    } // mgsearch namespace
} // mp namespace
} // placement namespace
