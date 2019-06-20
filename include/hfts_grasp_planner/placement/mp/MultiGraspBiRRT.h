#pragma once

#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/ompl/ORRedirectableBiRRT.h>
#include <openrave/openrave.h>
#include <unordered_map>
#include <vector>

namespace placement {
namespace mp {
    /**
         * Implements a multi-grasp motion planner that runs a separate BiRRT for each grasp.
         * All BiRRTs are executed in sequence for a limited time duration. Accordingly, the more
         * grasps there are, the longer plan() takes to compute.
         */
    class SequentialMGBiRRT : public MultiGraspMP {
    public:
        /**
         * Create a new sequential multi-grasp BiRRT algorithm.
         * @param penv, environment to plan in. This object takes ownership of this environment
         *      and its Destroy function will be called upon the destruction of this object.
         * @param robot_id, Environment id of the robot to plan for
         * @param obj_id, Environment id of the object to plan for
         */
        SequentialMGBiRRT(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id);
        ~SequentialMGBiRRT();

        void plan(std::vector<std::pair<unsigned int, WaypointPath>>& new_paths, double time_limit) override;
        void addGrasp(const Grasp& grasp) override;
        void addGoal(const Goal& goal) override;
        void removeGoals(const std::vector<unsigned int>& goal_ids) override;

        // protected:

    private:
        OpenRAVE::EnvironmentBasePtr _base_env;
        unsigned int _robot_id;
        unsigned int _obj_id;
        std::unordered_map<unsigned int, ompl::ORRedirectableBiRRTPtr> _planners;
        std::vector<Grasp> _grasps;
        std::unordered_map<unsigned int, Goal> _goals;
    };
    typedef std::shared_ptr<SequentialMGBiRRT> SequentialBiRRTPtr;
    typedef std::shared_ptr<const SequentialMGBiRRT> SequentialBiRRTConstPtr;

    /**
         * Implements a multi-grasp motion planner that runs a separate BiRRT for each grasp.
         * Each BiRRT is executed in its own thread (the BiRRT algorithm itself is single-threaded).
         */
    class ParallelMGBiRRT : public MultiGraspMP {
    public:
        ParallelMGBiRRT(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id);
        ~ParallelMGBiRRT();

        void plan(std::vector<std::pair<unsigned int, WaypointPath>>& new_paths, double time_limit) override;
        void addGrasp(const Grasp& grasp) override;
        void addGoal(const Goal& goal) override;
        void removeGoals(const std::vector<unsigned int>& goal_ids) override;

    private:
        std::vector<Grasp> _grasps;
        std::vector<Goal> _goals;
    };
    typedef std::shared_ptr<ParallelMGBiRRT> ParallelBiRRTPtr;
    typedef std::shared_ptr<const ParallelMGBiRRT> ParallelBiRRTConstPtr;
}
}