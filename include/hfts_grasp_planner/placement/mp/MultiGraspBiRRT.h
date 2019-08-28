#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <hfts_grasp_planner/placement/mp/ompl/ORRedirectableBiRRT.h>
#include <mutex>
#include <openrave/openrave.h>
#include <thread>
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

        void plan(std::vector<Solution>& new_paths, double time_limit) override;
        void pausePlanning() override;
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
        /**
         * Wrapper that runs ORRedirectableBiRRT in separate thread.
         */
        class AsynchPlanner {
        public:
            AsynchPlanner(ompl::ORRedirectableBiRRTPtr planner);
            ~AsynchPlanner();

            /**
             * Add the given goal(s) to the planner.
             */
            void addGoal(const Goal& goal);
            void addGoals(const std::vector<Goal>& goals);
            /**
             * Remove the given goal(s) to the planner.
             */
            void removeGoal(unsigned int goal_id);
            void removeGoals(const std::vector<unsigned int>& goals);

            /**
             * Return all new paths that have been found since this function has been called
             * last time. This may include paths to goals that have been removed.
             */
            void getNewPaths(std::vector<Solution>& paths);

            /**
             * Pause or unpause the thread.
             */
            void pause(bool bpause);

        protected:
            void run(); // loop that the planner thread is executing

        private:
            // data structures to synchronize adding and removing goals
            std::mutex _goal_modification_mutex;
            std::vector<Goal> _goals_to_add;
            std::vector<unsigned int> _goals_to_remove;
            std::condition_variable _goal_modification_cv;
            // data structures to store paths
            std::mutex _path_list_mutex;
            std::vector<Solution> _new_paths;
            // thread and thread termination
            std::atomic<bool> _terminate;
            std::atomic<bool> _paused;
            std::thread _thread;
            // actual planning algorithm
            ompl::ORRedirectableBiRRTPtr _planner;
        };

        ParallelMGBiRRT(OpenRAVE::EnvironmentBasePtr penv, unsigned int robot_id, unsigned int obj_id);
        ~ParallelMGBiRRT();

        void plan(std::vector<Solution>& new_paths, double time_limit) override;
        void pausePlanning() override;
        void addGrasp(const Grasp& grasp) override;
        // TODO it would be better to add goals as a batch
        void addGoal(const Goal& goal) override;
        void removeGoals(const std::vector<unsigned int>& goal_ids) override;

    private:
        OpenRAVE::EnvironmentBasePtr _base_env;
        unsigned int _robot_id;
        unsigned int _obj_id;
        std::unordered_map<unsigned int, std::shared_ptr<AsynchPlanner>> _planners;
        std::vector<Grasp> _grasps;
        std::unordered_map<unsigned int, Goal> _goals;
    };
    typedef std::shared_ptr<ParallelMGBiRRT> ParallelBiRRTPtr;
    typedef std::shared_ptr<const ParallelMGBiRRT> ParallelBiRRTConstPtr;
}
}