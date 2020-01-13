#pragma once
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MGGraphSearchMP.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <hfts_grasp_planner/placement/mp/mgsearch/ORStateSpace.h>
#include <set>
#include <vector>

namespace placement {
namespace mp {
    // Wrapper around MGGraphSearchMP class for OpenRAVE environments that follows the MultiGraspMP interface
    class ORGraphSearch : public MultiGraspMP {
    public:
        ORGraphSearch(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id,
            const mgsearch::MGGraphSearchMP::Parameters& params);
        ~ORGraphSearch();

        void plan(std::vector<Solution>& new_paths, double time_limit) override;
        void pausePlanning() override;
        void addGrasp(const Grasp& grasp) override;
        void addGoal(const Goal& goal) override;
        void removeGoals(const std::vector<unsigned int>& goal_ids) override;

    private:
        OpenRAVE::EnvironmentBasePtr _env;
        OpenRAVE::RobotBasePtr _robot;
        Config _start_config;
        unsigned int _robot_id;
        unsigned int _obj_id;
        mgsearch::ORStateSpacePtr _scene_interface;
        mgsearch::MGGraphSearchMPPtr _planner;
    };
    typedef std::shared_ptr<ORGraphSearch> ORGraphSearchPtr;
    typedef std::shared_ptr<const ORGraphSearch> ORGraphSearchConstPtr;
}
}