#pragma once
#include <Eigen/Core>
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <vector>

namespace placement {
namespace mp {
    namespace mgsearch {
        class StateValidityChecker {
        public:
            virtual ~StateValidityChecker() = 0;
            virtual bool isValid(const Config& c) const = 0;
            virtual bool isValid(const Config& c, unsigned int grasp_id) const = 0;
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

        double cSpaceDistance(const Config& a, const Config& b);
    }
}
}