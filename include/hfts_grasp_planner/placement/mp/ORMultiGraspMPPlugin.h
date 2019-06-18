#pragma once
#include <openrave/plugin.h>

#include <openrave/openrave.h>

#include <boost/bind.hpp>
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <unordered_map>

namespace placement {
namespace mp {
    class ORMultiGraspMPPlugin : public OpenRAVE::ModuleBase {
    public:
        ORMultiGraspMPPlugin(OpenRAVE::EnvironmentBasePtr penv, const std::string& algorithm);
        ~ORMultiGraspMPPlugin();

        /**
         * Initialize planner for a new problem given the current state of the environment.
         * Resets all previously stored information, i.e. all grasps, goals and all motion planning data.
         * You need to add goals using the addGoal function. In addition, you need to add at least one grasp.
         * Input format: robot_name obj_name
         * where
         *  robot_id, int - environment id of the robot to plan for (planning for its active manipulator)
         *  obj_name, int - environment id of the kinbody that is going to be grasped
         */
        bool initPlan(std::ostream& sout, std::istream& sinput);

        /**
         * Plan (start or continue) until either a timeout is reached or some new solutions were found.
         * Input format: max_time
         * Output format: id0 ... idk
         * where
         *  max_time, double - maximal planning duration (the actual runtime of this function may be longer than this value),
         *      set 0.0 if no timeout should be used
         *  idX, int - ids of goals to which a new solution was found
         * The actual paths can be retrieved calling getPath(..).
         */
        bool plan(std::ostream& sout, std::istream& sinput);

        /**
         * Return the path to a given goal. If no path to the goal has been found yet, an empty string is returned.
         * Input format: gid
         * Output format: q0_0 ... qn_0
         *                q0_1 ... qn_1
         *                ...
         *                q0_k ... qn_k
         * where
         *  gid, int - id of the goal
         *  In the output each line represents one waypoint of the path with
         *  qj_i, double - value of joint j at waypoint i
         */
        bool getPath(std::ostream& sout, std::istream& sinput);

        // TODO optionally set parameters for the algorithms
        // bool setParameters(std::ostream& sout, std::istream& sinput);

        /**
         * Inform the motion planner about a new grasp.
         * Input format: id x y z qx qy qz qw q0 ... qn
         * where
         *  id, int - unique identifier of the grasp
         *  x, y, z, double - end-effector position in object frame
         *  qw, qx, qy, qz - end-effector orientation in object frame (quaternion)
         *  q0, ..., qn - gripper joint configuration
         */
        bool addGrasp(std::ostream& sout, std::istream& sinput);

        /**
         * Add a new goal.
         * Input format: id gid q0 ... qn
         * where
         *  id, int - unique identifier for this goal
         *  gid, int - grasp id for which this goal is defined
         *  q0, ..., qn, double - goal arm configuration
         */
        bool addGoal(std::ostream& sout, std::istream& sinput);

        /**
         * Inform the motion planner to stop planning towards the given goals.
         * Input format: id0 id1 id2 ... idn
         * where
         *  idX, int - goal identifiers
         */
        bool removeGoals(std::ostream& sout, std::istream& sinput);

    private:
        std::string _algorithm_name;
        MultiGraspMPPtr _planner;
        std::unordered_map<unsigned int, MultiGraspMP::WaypointPath> _solutions;
        OpenRAVE::EnvironmentBasePtr _original_env;
    };
    OpenRAVE::InterfaceBasePtr CreateInterfaceValidated(OpenRAVE::InterfaceType type, const std::string& interfacename, std::istream& sinput, OpenRAVE::EnvironmentBasePtr penv);
    void GetPluginAttributesValidated(OpenRAVE::PLUGININFO& info);
}
}