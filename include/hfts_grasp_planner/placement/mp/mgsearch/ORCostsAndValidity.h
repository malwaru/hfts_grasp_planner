#pragma once
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <unordered_map>

namespace placement {
namespace mp {
    namespace mgsearch {
        double cSpaceDistance(const Config& a, const Config& b);

        class ORSceneInterface : public StateSpace {
        public:
            ORSceneInterface(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id);
            ~ORSceneInterface();
            // Grasp management
            void addGrasp(const MultiGraspMP::Grasp& g);
            void removeGrasp(unsigned int gid);
            // State validity
            bool isValid(const Config& c) const override;
            bool isValid(const Config& c, unsigned int grasp_id, bool only_obj = false) const override;
            // state cost
            double cost(const Config& a) const override;
            double conditional_cost(const Config& a, unsigned int grasp_id) const override;
            // distance
            double distance(const Config& a, const Config& b) const override;
            // space information
            unsigned int getDimension() const override;
            void getBounds(Config& lower, Config& upper) const override;

        private:
            OpenRAVE::EnvironmentBasePtr _penv;
            OpenRAVE::RobotBasePtr _robot;
            OpenRAVE::KinBodyPtr _object;
            OpenRAVE::CollisionReportPtr _report;
            OpenRAVE::CollisionCheckerBasePtr _col_checker;
            mutable bool _distance_check_enabled;

            void enableDistanceCheck(bool enable) const;
            std::unordered_map<unsigned int, MultiGraspMP::Grasp> _grasps;
            void setGrasp(unsigned int gid) const;
            inline double costPerConfig(const Config& c) const;
            double integrateCosts(const Config& a, const Config& b) const;
        };

        typedef std::shared_ptr<ORSceneInterface> ORSceneInterfacePtr;
    }
}
}
