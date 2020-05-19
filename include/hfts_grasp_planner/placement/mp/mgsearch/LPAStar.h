#include <hfts_grasp_planner/placement/mp/mgsearch/SearchCommon.h>

/**
 *  Defines Lifelong Planning A* and needed data structures
 */
namespace placement {
namespace mp {
    namespace mgsearch {
        namespace lpastar {
            struct PQElement {
                // TODO update to whatever we need for LPA*
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

            inline double f(const PQElement& el)
            {
                return el.g_value + el.h_value;
            }

            struct PQElementCompare {
                // TODO update to whatever we need for LPA*
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
            template <typename PQ>
            struct VertexData {
                // TODO update to whatever we need for LPA*
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
             * LPA* search algorithm.
             * The template parameter G needs to be of a type implementing the GraspAgnosticGraph interface specified in Graphs.h.
             * The template parameter PQ needs to be a boost::heap
             */
            template <typename G, typename PQ = boost::heap::fibonacci_heap<PQElement, boost::heap::compare<PQElementCompare>>>
            void lpaStarSearch(const G& graph, SearchResult& result)
            {
                utils::ScopedProfiler("lpaStarSearch");
                // get start node
                unsigned int v_start = graph.getStartNode();
                // initialize result structure
                result.solved = false;
                result.path.clear();
                result.path_cost = std::numeric_limits<double>::infinity();
                result.goal_cost = std::numeric_limits<double>::infinity();
                result.goal_node = v_start;
                // initialize algorithm data structures
                PQ pq;
                typedef std::unordered_map<unsigned int, VertexData<PQ>> VertexDataMap;
                VertexDataMap vertex_data;
                if (graph.checkValidity(v_start)) {
                    vertex_data.emplace(std::make_pair(v_start, VertexData<PQ>(v_start, 0.0, v_start)));
                    vertex_data.at(v_start).pq_handle = pq.push(PQElement(v_start, 0.0, graph.heuristic(v_start)));
                }
                // main iteration - is skipped if start vertex is invalid
                // TODO update to LPA*
                while (not pq.empty() and result.cost() > f(pq.top())) {
                    // PQElement current_el = pq.top();
                    // pq.pop();
                    // vertex_data.at(current_el.v).closed = true;
                    // if (graph.isGoal(current_el.v)) {
                    //     result.solved = true;
                    //     double path_cost = vertex_data.at(current_el.v).g;
                    //     double goal_cost = graph.getGoalCost(current_el.v);
                    //     double new_cost = path_cost + goal_cost;
                    //     if (new_cost < result.cost()) {
                    //         result.path_cost = path_cost;
                    //         result.goal_cost = goal_cost;
                    //         result.goal_node = current_el.v;
                    //     }
                    // } else {
                    //     // extend current_el.v
                    //     auto [siter, send] = graph.getSuccessors(current_el.v, true);
                    //     for (; siter != send; ++siter) {
                    //         uint s = *siter;
                    //         // check vertex and edge validity
                    //         double wvs = graph.getEdgeCost(current_el.v, s, false);
                    //         if (std::isinf(wvs)) {
                    //             continue;
                    //         }
                    //         // s is reachable from v. compute the g value it can reach.
                    //         double g_s = current_el.g_value + wvs;
                    //         // create a VertexData element if it doesn't exist yet.
                    //         auto iter = vertex_data.find(s);
                    //         if (iter != vertex_data.end()) {
                    //             if (iter->second.closed)
                    //                 continue;
                    //             // s has been reached from another node before, check whether we can decrease its key
                    //             if (iter->second.g > g_s) {
                    //                 iter->second.g = g_s;
                    //                 iter->second.p = current_el.v;
                    //                 (*(iter->second.pq_handle)).g_value = g_s;
                    //                 pq.increase(iter->second.pq_handle); // increase priority
                    //             }
                    //         } else {
                    //             // s hasn't been reached before, add a new VertexData element and push it to pq
                    //             auto [iter, valid] = vertex_data.emplace(std::make_pair(s, VertexData<PQ>(s, g_s, current_el.v)));
                    //             assert(valid);
                    //             iter->second.pq_handle = pq.push(PQElement(s, g_s, graph.heuristic(s)));
                    //         }
                    //     }
                    // }
                }
                // extract path
                if (result.solved) {
                    extractPath<VertexDataMap>(v_start, vertex_data, result);
                }
            }

        } // lpastar
    } // mgsearch
} // mp
} // placement