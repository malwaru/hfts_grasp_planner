#pragma once
#include <hfts_grasp_planner/placement/mp/mgsearch/Interfaces.h>
#include <unordered_map>

namespace placement {
namespace mp {
    namespace mgsearch {
        class ORSceneInterface : public StateValidityChecker, EdgeCostComputer {
        public:
            ORSceneInterface(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id);
            ~ORSceneInterface();
            // Grasp management
            void addGrasp(const MultiGraspMP::Grasp& g);
            void removeGrasp(unsigned int gid);
            // State validity
            bool isValid(const Config& c) const override;
            bool isValid(const Config& c, unsigned int grasp_id) const override;
            // edge cost
            double lowerBound(const Config& a, const Config& b) const override;
            double cost(const Config& a, const Config& b) const override;
            double cost(const Config& a, const Config& b, unsigned int grasp_id) const override;

        private:
            OpenRAVE::EnvironmentBasePtr _penv;
            OpenRAVE::RobotBasePtr _robot;
            OpenRAVE::KinBodyPtr _object;
            OpenRAVE::CollisionReportPtr _report;

            std::unordered_map<unsigned int, MultiGraspMP::Grasp> _grasps;
            void setGrasp(unsigned int gid) const;
            inline double costPerConfig(const Config& c) const;
            double integrateCosts(const Config& a, const Config& b) const;
        };

        class MGGoalDistance : public CostToGoHeuristic {
        public:
            MGGoalDistance();
            ~MGGoalDistance();
            void addGoal(const MultiGraspMP::Goal& goal);
            void removeGoal(unsigned int goal_id);
            // interface functions
            double costToGo(const Config& a) const override;
            double costToGo(const Config& a, unsigned int grasp_id) const override;
        };
    }
}
}
