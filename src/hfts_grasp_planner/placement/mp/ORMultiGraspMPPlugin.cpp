#include <hfts_grasp_planner/placement/mp/MultiGraspBiRRT.h>
#include <hfts_grasp_planner/placement/mp/ORMultiGraspMPPlugin.h>
#include <iostream>

#define PARALLEL_BIRRT "parallelmgbirrt"
#define SEQUENTIAL_BIRRT "sequentialmgbirrt"

using namespace OpenRAVE;
using namespace placement::mp;

ORMultiGraspMPPlugin::ORMultiGraspMPPlugin(EnvironmentBasePtr penv, const std::string& algorithm)
    : ModuleBase(penv)
{
    __description = "A motion planning plugin for redirectable multi-grasp motion planning";
    RegisterCommand("initPlan", boost::bind(&ORMultiGraspMPPlugin::initPlan, this, _1, _2),
        "Initialize planner for a new problem given the current state of the environment.\n"
        "Resets all previously stored information, i.e. all grasps, goals and all motion planning data.\n"
        "You need to add goals using the addGoal function. In addition, you need to add at least one grasp.\n"
        "Input format: robot_name obj_name\n"
        "where \n"
        " robot_name, string - name of the robot to plan for (planning for its active manipulator)\n"
        " obj_name, string - name of the kinbody that is going to be grasped\n");
    RegisterCommand("plan", boost::bind(&ORMultiGraspMPPlugin::plan, this, _1, _2),
        "Plan (start or continue) until either a timeout is reached or some new solutions were found.\n"
        "Input format: max_time\n"
        "Output format: id0 ... idk\n"
        " id q0_0 q1_0 q2_0 .. qn_0 q0_1 .. qn_1 .. q0_k .. qn_k \\n \n"
        "where\n"
        " max_time, double - maximal planning duration (the actual runtime of this function may be longer than this value)\n"
        "      set 0.0 if no timeout should be used"
        " idX, int - ids of goals to which a new solution was found\n"
        "The actual paths can be retrieved calling getPath(..).");
    RegisterCommand("getPath", boost::bind(&ORMultiGraspMPPlugin::getPath, this, _1, _2),
        "Return the path to a given goal. If no path to the goal has been found yet, an empty string is returned.\n"
        "Input format: gid\n"
        "Output format: q0_0 ... qn_0\n"
        "               q0_1 ... qn_1\n"
        "               ...\n"
        "               q0_k ... qn_k\n"
        "where\n"
        " gid, int - id of the goal\n"
        " In the output each line represents one waypoint of the path with\n"
        " qj_i, double - value of joint j at waypoint i");
    RegisterCommand("addGrasp", boost::bind(&ORMultiGraspMPPlugin::addGrasp, this, _1, _2),
        "Inform the motion planner about a new grasp. \n"
        "Input format: id x y z qx qy qz qw q0 ... qn \n"
        "where \n"
        " id, int - unique identifier of the grasp \n"
        " x, y, z, double - end-effector position in object frame \n"
        " qx, qy, qz, qw - end-effector orientation in object frame (quaternion) \n"
        " q0, ..., qn - gripper joint configuration \n");
    RegisterCommand("addGoal", boost::bind(&ORMultiGraspMPPlugin::addGoal, this, _1, _2),
        "Add a new goal.\n"
        "Input format: id gid q0 ... qn\n"
        "where\n"
        " id, int - unique identifier for this goal\n"
        " gid, int - grasp id for which this goal is defined\n"
        " q0, ..., qn, double - goal arm configuration");
    RegisterCommand("removeGoals", boost::bind(&ORMultiGraspMPPlugin::removeGoals, this, _1, _2),
        "Inform the motion planner to stop planning towards the given goals. \n"
        "Input format: id0 id1 id2 ... idN \n"
        "where\n"
        " idX, int - goal identifiers");
    _algorithm_name = algorithm;
    _original_env = penv;
    RAVELOG_DEBUG("Constructed ORMultiGraspMPPlugin");
}

ORMultiGraspMPPlugin::~ORMultiGraspMPPlugin()
{
    RAVELOG_DEBUG("Destructing ORMultiGraspMPPlugin");
}

bool ORMultiGraspMPPlugin::initPlan(std::ostream& sout, std::istream& sinput)
{
    unsigned int robot_id;
    sinput >> robot_id;
    unsigned int obj_id;
    sinput >> obj_id;
    boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_original_env->GetMutex());
    auto obj_body = _original_env->GetBodyFromEnvironmentId(obj_id);
    if (!obj_body) {
        std::string error_msg = "Could not retrieve object with id " + std::to_string(obj_id);
        RAVELOG_ERROR(error_msg);
        throw std::runtime_error(error_msg);
    }
    auto robot = _original_env->GetBodyFromEnvironmentId(robot_id);
    if (!robot) {
        std::string error_msg = "Could not retrieve robot with id " + std::to_string(robot_id);
        RAVELOG_ERROR(error_msg);
        throw std::runtime_error(error_msg);
    }
    RAVELOG_DEBUG("Initializing plan for robot " + robot->GetName() + " and object " + obj_body->GetName());
    auto query_env = _original_env->CloneSelf(OpenRAVE::Clone_Bodies);
    query_env->StopSimulation();
    if (_algorithm_name == PARALLEL_BIRRT) {
        _planner = std::make_shared<ParallelMGBiRRT>(query_env, robot_id, obj_id);
    } else if (_algorithm_name == SEQUENTIAL_BIRRT) {
        _planner = std::make_shared<SequentialMGBiRRT>(query_env, robot_id, obj_id);
    } else {
        RAVELOG_ERROR("Unknown algorithm " + _algorithm_name + ". Can not plan.");
        throw std::logic_error("Unknown planning algorithm name " + _algorithm_name);
    }
    return true;
}

bool ORMultiGraspMPPlugin::plan(std::ostream& sout, std::istream& sinput)
{
    double timeout;
    sinput >> timeout;
    std::vector<std::pair<unsigned int, MultiGraspMP::WaypointPathPtr>> new_paths;
    _planner->plan(new_paths, timeout);
    if (!new_paths.empty()) {
        for (unsigned int i = 0; i < new_paths.size(); ++i) {
            auto new_sol = new_paths.at(i);
            sout << new_sol.first;
            _solutions[new_sol.first] = new_sol.second;
            if (i + 1 < new_paths.size())
                sout << " ";
        }
    }
    return true;
}

bool ORMultiGraspMPPlugin::getPath(std::ostream& sout, std::istream& sinput)
{
    unsigned int id;
    sinput >> id;
    auto iter = _solutions.find(id);
    if (iter != _solutions.end()) {
        MultiGraspMP::WaypointPathPtr path = iter->second;
        for (unsigned int wi = 0; wi < path->size(); ++wi) {
            auto& wp = path->at(wi);
            for (unsigned int i = 0; i < wp.size(); ++i) {
                sout << wp.at(i);
                if (i + 1 < wp.size())
                    sout << " ";
            }
            if (wi + 1 < path->size())
                sout << "\n";
        }
    }
    return true;
}

bool ORMultiGraspMPPlugin::addGrasp(std::ostream& sout, std::istream& sinput)
{
    if (!_planner)
        return false;
    MultiGraspMP::Grasp grasp;
    // first read id
    sinput >> grasp.id;
    // next read x, y, z
    sinput >> grasp.pos.x >> grasp.pos.y >> grasp.pos.z;
    // quaternion
    sinput >> grasp.quat.w >> grasp.quat.x >> grasp.quat.y >> grasp.quat.z;
    // finally read configuration
    while (sinput.good()) {
        double q;
        sinput >> q;
        grasp.gripper_values.push_back(q);
    }
    // for debug purposes serialize grasp again
    RAVELOG_DEBUG("Adding new grasp: " + grasp.print());
    _planner->addGrasp(grasp);
    return false; // TODO what to return?
}

bool ORMultiGraspMPPlugin::addGoal(std::ostream& sout, std::istream& sinput)
{
    if (!_planner)
        return false;
    MultiGraspMP::Goal goal;
    // first read id
    sinput >> goal.id;
    // next read grasp id
    sinput >> goal.grasp_id;
    // finally read configuration
    while (sinput.good()) {
        double q;
        sinput >> q;
        goal.config.push_back(q);
    }
    RAVELOG_DEBUG("Adding new goal: " + goal.print());
    _planner->addGoal(goal);
    // TODO no clue what the return value is supposed to mean
    // TODO In Python, the only difference appears to be whether sout is returned to the caller
    return false;
}

bool ORMultiGraspMPPlugin::removeGoals(std::ostream& sout, std::istream& sinput)
{
    if (!_planner)
        return false;
    std::vector<unsigned int> goals_to_remove;
    std::stringstream debug_ss;
    while (sinput.good()) {
        unsigned int id;
        sinput >> id;
        debug_ss << id << " ";
        goals_to_remove.push_back(id);
    }

    RAVELOG_DEBUG("Removing goals: " + debug_ss.str());
    _planner->removeGoals(goals_to_remove);
    return false; // TODO what to return?
}

InterfaceBasePtr CreateInterfaceValidated(InterfaceType type, const std::string& interfacename, std::istream& sinput, EnvironmentBasePtr penv)
{
    // std::cout << "CreateInterfaceValidated" << std::endl;
    // std::cout << interfacename << PARALLEL_BIRRT << (interfacename == PARALLEL_BIRRT) << std::endl;
    if (type == PT_Module && interfacename == PARALLEL_BIRRT) {
        return InterfaceBasePtr(new ORMultiGraspMPPlugin(penv, PARALLEL_BIRRT));
    } else if (type == PT_Module && interfacename == SEQUENTIAL_BIRRT) {
        return InterfaceBasePtr(new ORMultiGraspMPPlugin(penv, SEQUENTIAL_BIRRT));
    }
    // TODO for other planning algoirthms check here and create plugin accordingly
    return InterfaceBasePtr();
}

void GetPluginAttributesValidated(PLUGININFO& info)
{
    // std::cout << "GetPluginAttributedValidated" << std::endl;
    info.interfacenames[PT_Module].push_back(PARALLEL_BIRRT);
    info.interfacenames[PT_Module].push_back(SEQUENTIAL_BIRRT);
    // TODO add other planners here, too
}

void DestroyPlugin()
{
    // TODO do we need to do anything here?
}