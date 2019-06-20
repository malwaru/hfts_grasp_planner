#pragma once
#include <hfts_grasp_planner/external/ompl/geometric/RedirectableRRTConnect.h>
#include <iostream>
#include <memory>
#include <ompl/base/StateSpace.h>
#include <ompl/base/goals/GoalStates.h>
#include <ompl/datastructures/NearestNeighborsGNATNoThreadSafety.h>
#include <ompl/geometric/SimpleSetup.h>
#include <openrave/openrave.h>
#include <or_ompl/StateSpaces.h>
#include <unordered_map>

namespace placement {
namespace mp {
    namespace ompl {
        // store current goals here to easily identify them when we have a solution
        struct GoalWithId {
            std::vector<double> config;
            unsigned int id;
            GoalWithId(const std::vector<double>& iconfig, unsigned int iid)
                : config(iconfig)
                , id(iid)
            {
            }

            static double distanceId(const std::shared_ptr<GoalWithId>& a, const std::shared_ptr<GoalWithId>& b)
            {
                return distance(a->config, b->config);
            }

            static double distance(const std::vector<double>& config_a, const std::vector<double>& config_b)
            {
                double delta = 0.0;
                for (size_t i = 0; i < config_a.size(); ++i) {
                    delta += std::abs(config_a.at(i) - config_b.at(i));
                }
                return delta;
            }
        };
        // Data structure that maps configuration to goals, allowing us to identify what goal a path leads to
        class GoalStorage {
        public:
            GoalStorage()
            {
                gnat.setDistanceFunction(GoalWithId::distanceId);
                query_el = std::make_shared<GoalWithId>(std::vector<double>(), 0);
            }
            ~GoalStorage() = default;

            void add(const std::shared_ptr<GoalWithId>& c)
            {
                gnat.add(c);
                goals[c->id] = c;
            }

            void remove(std::shared_ptr<GoalWithId>& cid)
            {
                goals.erase(cid->id);
                gnat.remove(cid);
            }

            bool contains(unsigned int id) const
            {
                return goals.find(id) != goals.end();
            }

            bool contains(const std::vector<double>& config) const
            {
                if (gnat.size() == 0)
                    return false;
                auto nearest_el = nearest(config);
                float dist = GoalWithId::distance(nearest_el->config, config);
                return dist <= 1e-5;
            }

            std::shared_ptr<GoalWithId> nearest(const std::vector<double>& config) const
            {
                query_el->config = config;
                return gnat.nearest(query_el);
            }

            std::shared_ptr<GoalWithId> nearest(const GoalWithId& cid) const
            {
                return nearest(cid.config);
            }

            std::shared_ptr<GoalWithId> nearest(std::shared_ptr<const GoalWithId> state) const
            {
                return nearest(state->config);
            }

            std::shared_ptr<GoalWithId> getGoal(unsigned int id) const
            {
                auto iter = goals.find(id);
                if (iter != goals.end()) {
                    return iter->second;
                }
                return nullptr;
            }

            std::shared_ptr<GoalWithId> getGoal(const std::vector<double>& config) const
            {
                if (gnat.size() == 0)
                    return nullptr;
                auto nearest_el = nearest(config);
                float dist = GoalWithId::distance(nearest_el->config, config);
                if (dist < 1e-5) {
                    return nearest_el;
                }
                return nullptr;
            }

            unsigned int size() const
            {
                return goals.size();
            }

            void clear()
            {
                gnat.clear();
                goals.clear();
            }

            void getGoals(std::vector<std::shared_ptr<GoalWithId>>& gs)
            {
                for (const auto& key_value : goals) {
                    gs.push_back(key_value.second);
                }
            }

        private:
            mutable std::shared_ptr<GoalWithId> query_el;
            std::unordered_map<unsigned int, std::shared_ptr<GoalWithId>> goals;
            ::ompl::NearestNeighborsGNATNoThreadSafety<std::shared_ptr<GoalWithId>> gnat;
        };

        class ORRedirectableBiRRT {
        public:
            /**
             * Create a new ORRedirectableBiRRT algorithm.
             * @param probot, the robot to plan for (must be from penv)
             * @param penv, the environment probot lives in. This object assumes ownership over penv
             *      and calls its Destroy function when destructed.
             */
            ORRedirectableBiRRT(OpenRAVE::RobotBasePtr probot, OpenRAVE::EnvironmentBasePtr penv);
            ~ORRedirectableBiRRT();
            /**
             * Plan towards the set goals until either a new path has been found or
             * timeout has exceeded.
             * @param timeout - number of seconds to plan for
             * @param goal_id - if a new path is found, goal_id is set to the goal this path leads to
             * @return true if a new path has been found
             */
            bool plan(double timeout, unsigned int& goal_id);
            /**
             * Plan towards the set goals until either a new path has been found or
             * the given function returns true.
             * @param interrupt_fn - a function that returns true if planning shall be interrupted
             * @param goal_id - if a new path is found, goal_id is set to the goal this path leads to
             * @return true if a new path has been found
             */
            bool plan(std::function<bool()> interrupt_fn, unsigned int& goal_id);

            /**
             *  Return the number of unreached goals.
             */
            unsigned int getNumGoals() const;

            /**
             * Return the path leading to the specified goal. If this path has not been found yet,
             * or the goal id is unknown, an empty path is returned.
             */
            void getPath(unsigned int id, std::vector<std::vector<double>>& path) const;

            /**
             * Add the given configuration as a new goal that is identified by the given id.
             * The caller has to ensure that this id is unique. If a goal with same already exists,
             * a logic error is thrown.
             */
            void addGoal(const std::vector<double>& config, unsigned int id);
            /**
             * Remove the goal with the given id from the active list of goals.
             * If a path to the given goal has already been found, this is a no-op.
             */
            void removeGoal(unsigned int id);

        private:
            OpenRAVE::EnvironmentBasePtr _env;
            OpenRAVE::RobotBasePtr _robot;
            ::ompl::base::StateSpacePtr _state_space;
            ::ompl::geometric::SimpleSetupPtr _simple_setup;
            or_ompl::OrStateValidityCheckerPtr _or_validity_checker;
            ::ompl::geometric::RedirectableRRTConnectPtr _planner;
            std::shared_ptr<::ompl::base::GoalStates> _ompl_goal_states;

            GoalStorage _goal_storage;
            std::unordered_map<unsigned int, std::vector<std::vector<double>>> _path_storage;

            bool _handlePlanningStatus(::ompl::base::PlannerStatus status, unsigned int& goal_id);
            void _synchronizeGoals();
        };

        typedef std::shared_ptr<ORRedirectableBiRRT> ORRedirectableBiRRTPtr;
    }
}
}
