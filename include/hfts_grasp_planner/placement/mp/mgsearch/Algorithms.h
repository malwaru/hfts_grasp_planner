#pragma once
#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>
#include <cmath>
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
                        result.path_cost = vertex_data.at(current_el.v).g;
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
                        std::vector<unsigned int> successors;
                        graph.getSuccessors(current_el.v, successors, true);
                        for (unsigned int& s : successors) {
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
        }
    }
}
}
