#pragma once

#include <memory>
#include <openrave/openrave.h>
#include <string>
#include <vector>

namespace placement {
namespace mp {
    class MultiGraspMP {
    public:
        struct Grasp {
            unsigned int id;
            OpenRAVE::geometry::RaveVector<float> pos;
            OpenRAVE::geometry::RaveVector<float> quat;
            std::vector<float> gripper_values;

            std::string print() const
            {
                std::stringstream ss;
                ss << "id=" << id << " "
                   << "x=" << pos.x << " y=" << pos.y << " z=" << pos.z
                   << " qx=" << quat.x << " qy=" << quat.y << " qz=" << quat.z << " qw=" << quat.w;
                ss << " q=[";
                for (const float& v : gripper_values) {
                    ss << v << ", ";
                }
                ss << "]";
                return ss.str();
            }
        };
        struct Goal {
            unsigned int id;
            unsigned int grasp_id;
            std::vector<float> config;
            std::string print() const
            {
                std::stringstream ss;
                ss << "id=" << id << " "
                   << "grasp_id=" << grasp_id;
                ss << " q=[";
                for (const float& v : config) {
                    ss << v << ", ";
                }
                ss << "]";
                return ss.str();
            }
        };
        typedef std::vector<std::vector<float>> WaypointPath;
        virtual ~MultiGraspMP() = 0;
        virtual void plan(std::vector<std::pair<unsigned int, WaypointPath>>& new_paths, float time_limit = 0.0f) = 0;
        virtual void addGrasp(const Grasp& grasp) = 0;
        virtual void addGoal(const Goal& goal) = 0;
        virtual void removeGoals(const std::vector<unsigned int>& goal_ids) = 0;
        // virtual void getReachedGoals(const std::vector<unsigned int>& goals, bool new_only) const = 0;
    };
    typedef std::shared_ptr<MultiGraspMP> MultiGraspMPPtr;
    typedef std::shared_ptr<const MultiGraspMP> MultiGraspMPConstPtr;
}
}