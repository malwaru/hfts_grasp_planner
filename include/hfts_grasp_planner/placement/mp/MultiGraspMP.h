#pragma once

#include <memory>
#include <openrave/openrave.h>
#include <string>
#include <vector>

namespace placement {
namespace mp {
    typedef std::vector<double> Config;

    class MultiGraspMP {
    public:
        struct Grasp {
            unsigned int id;
            OpenRAVE::geometry::RaveVector<double> pos;
            OpenRAVE::geometry::RaveVector<double> quat;
            Config gripper_values;

            std::string print() const
            {
                std::stringstream ss;
                ss << "id=" << id << " "
                   << "x=" << pos.x << " y=" << pos.y << " z=" << pos.z
                   << " qx=" << quat.x << " qy=" << quat.y << " qz=" << quat.z << " qw=" << quat.w;
                ss << " q=[";
                for (const double& v : gripper_values) {
                    ss << v << ", ";
                }
                ss << "]";
                return ss.str();
            }
        };
        struct Goal {
            unsigned int id; // should be unique. Note operator== and operator!=
            unsigned int grasp_id;
            double quality;
            Config config;
            std::string print() const
            {
                std::stringstream ss;
                ss << "id=" << id << " "
                   << "grasp_id=" << grasp_id << " "
                   << "quality=" << quality;
                ss << " q=[";
                for (const double& v : config) {
                    ss << v << ", ";
                }
                ss << "]";
                return ss.str();
            }

            bool operator==(const Goal& other) const
            {
                return other.id == id;
            }

            bool operator!=(const Goal& other) const
            {
                return !(*this == other);
            }
        };

        typedef std::vector<Config> WaypointPath;
        typedef std::shared_ptr<WaypointPath> WaypointPathPtr;

        struct Solution {
            unsigned int goal_id;
            WaypointPathPtr path;
            double cost;
            Solution()
                : goal_id(0)
                , path(nullptr)
                , cost(INFINITY)
            {
            }

            Solution(unsigned int goal, WaypointPathPtr ipath, double icost)
                : goal_id(goal)
                , path(ipath)
                , cost(icost)
            {
            }
        };
        virtual ~MultiGraspMP() = 0;
        /**
         * Plan (start or continue) until either a timeout is reached or some new solutions were found.
         * @param new_paths - new solutions are stored in this vector
         * @param time_limit - maximal planning duration, if supported
         */
        virtual void plan(std::vector<Solution>& new_paths, double time_limit = 0.0f) = 0;
        /**
         * In case the underlying planner is run asynchronously, this function notifies the planner
         * to pause planning until either plan is called again, or the planner is desctructed.
         */
        virtual void pausePlanning() = 0;
        /**
         * Add a new grasp to plan for.
         */
        virtual void addGrasp(const Grasp& grasp) = 0;
        /**
         * Add a new goal to plan to.
         */
        virtual void addGoal(const Goal& goal) = 0;
        /**
         * Remove the goals with the given IDs.
         */
        virtual void removeGoals(const std::vector<unsigned int>& goal_ids) = 0;
        // virtual void getReachedGoals(const std::vector<unsigned int>& goals, bool new_only) const = 0;
    };
    typedef std::shared_ptr<MultiGraspMP> MultiGraspMPPtr;
    typedef std::shared_ptr<const MultiGraspMP> MultiGraspMPConstPtr;
}
}