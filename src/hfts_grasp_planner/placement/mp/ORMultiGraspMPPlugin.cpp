#include <hfts_grasp_planner/placement/mp/MultiGraspBiRRT.h>
#include <hfts_grasp_planner/placement/mp/ORGraphSearch.h>
#include <hfts_grasp_planner/placement/mp/ORMultiGraspMPPlugin.h>
#include <iostream>
#include <algorithm>

#define PARALLEL_BIRRT "parallelmgbirrt"
#define SEQUENTIAL_BIRRT "sequentialmgbirrt"

using namespace OpenRAVE;
using namespace placement::mp;

ORMultiGraspMPPlugin::ORMultiGraspMPPlugin(EnvironmentBasePtr penv, const std::string& algorithm) : ModuleBase(penv)
{
  __description = "A motion planning plugin for redirectable multi-grasp motion planning";
  RegisterCommand("initPlan", boost::bind(&ORMultiGraspMPPlugin::initPlan, this, _1, _2),
                  "Initialize planner for a new problem given the current state of the environment.\n"
                  "Resets all previously stored information, i.e. all grasps, goals and all motion planning data.\n"
                  "You need to add goals using the addGoal function. In addition, you need to add at least one grasp.\n"
                  "Input format: <robot_name> <obj_name> lambda=<val> sdf_file=<file_name>\n"
                  "where \n"
                  " <robot_name>, string - name of the robot to plan for (planning for its active manipulator)\n"
                  " <obj_name>, string - name of the kinbody that is going to be grasped\n"
                  " <val>, double - value for lambda, the tradeoff between path cost and goal cost\n"
                  " <file_name>, string - optionally, the path to a sdf file to maximize clearance\n");
  RegisterCommand("plan", boost::bind(&ORMultiGraspMPPlugin::plan, this, _1, _2),
                  "Plan (start or continue) until either a timeout is reached or some new solutions were found.\n"
                  "Input format: max_time\n"
                  "Output format: id0 ... idk\n"
                  " id q0_0 q1_0 q2_0 .. qn_0 q0_1 .. qn_1 .. q0_k .. qn_k \\n \n"
                  "where\n"
                  " max_time, double - maximal planning duration (the actual runtime of this function may be longer "
                  "than this value)\n"
                  "      set 0.0 if no timeout should be used"
                  " idX, int - ids of goals to which a new solution was found\n"
                  "The actual paths can be retrieved calling getPath(..).");
  RegisterCommand("pausePlanning", boost::bind(&ORMultiGraspMPPlugin::pausePlanning, this, _1, _2),
                  "In case the underlying planner is run asynchronously, this function notifies"
                  "the planner to pause plannung until either plan is called again, or the planner is destructed.");
  RegisterCommand("clear", boost::bind(&ORMultiGraspMPPlugin::clear, this, _1, _2),
                  "Clears all resources. After calling this you need to call initPlan again.");
  RegisterCommand("getPath", boost::bind(&ORMultiGraspMPPlugin::getPath, this, _1, _2),
                  "Return the path to a given goal. If no path to the goal has been found yet, an empty string is "
                  "returned.\n"
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
                  " quality, double - the quality/objective value of the goal\n"
                  " q0, ..., qn, double - goal arm configuration");
  RegisterCommand("addWaypoints", boost::bind(&ORMultiGraspMPPlugin::addWaypoints, this, _1, _2),
                  "Add sample configurations that are not goals but may be beneficial for the planner to use. The "
                  "configurations may be in collision."
                  "Input format: q0_0 ... qn_0\\n\n"
                  "              ...\n"
                  "              q0_k ... qn_k\\n\n"
                  "where\n"
                  " q0_i, ..., qn_i, double - arm configuration i");
  RegisterCommand("removeGoals", boost::bind(&ORMultiGraspMPPlugin::removeGoals, this, _1, _2),
                  "Inform the motion planner to stop planning towards the given goals. \n"
                  "Input format: id0 id1 id2 ... idN \n"
                  "where\n"
                  " idX, int - goal identifiers");
  RegisterCommand("saveStats", boost::bind(&ORMultiGraspMPPlugin::saveStats, this, _1, _2),
                  "Save planning statistics since the last time initPlanner was called to file. \n"
                  "Input format: <filename> \n"
                  "where\n"
                  " <filename>, str - name of a file in which to store the statistics");
  RegisterCommand("saveSolutions", boost::bind(&ORMultiGraspMPPlugin::saveSolutions, this, _1, _2),
                  "Save found solutions since the last time initPlanner was called to file. \n"
                  "Input format: <filename> \n"
                  "where\n"
                  " <filename>, str - name of a file in which to store the statistics");
  _algorithm_name = algorithm;
  _original_env = penv;
  RAVELOG_DEBUG("Constructed ORMultiGraspMPPlugin");
}

ORMultiGraspMPPlugin::~ORMultiGraspMPPlugin()
{
  RAVELOG_DEBUG("Destructing ORMultiGraspMPPlugin");
}

void parseGraphSearchType(const std::string& name_to_parse, mgsearch::MGGraphSearchMP::Parameters& params)
{
  // try to split string by ';'
  auto split_index = name_to_parse.find(';');
  if (split_index < name_to_parse.npos)
  {
    std::string algorithm_name = name_to_parse.substr(0, split_index);
    std::string graph_name = name_to_parse.substr(split_index + 1);
    params.algo_type = mgsearch::MGGraphSearchMP::getAlgorithmType(algorithm_name);
    params.graph_type = mgsearch::MGGraphSearchMP::getGraphType(graph_name);
    // check whether this is a valid combination
    std::pair<mgsearch::MGGraphSearchMP::AlgorithmType, mgsearch::MGGraphSearchMP::GraphType> algo_graph_pair = {
        params.algo_type, params.graph_type};
    auto iter = std::find(std::cbegin(mgsearch::MGGraphSearchMP::VALID_ALGORITHM_GRAPH_COMBINATIONS),
                          std::cend(mgsearch::MGGraphSearchMP::VALID_ALGORITHM_GRAPH_COMBINATIONS), algo_graph_pair);
    if (iter == std::cend(mgsearch::MGGraphSearchMP::VALID_ALGORITHM_GRAPH_COMBINATIONS))
    {
      std::string err_msg("Invalid algorithm/graph combination " + name_to_parse);
      RAVELOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
  }
  else
  {
    std::string err_msg("Invalid algorithm encoding " + name_to_parse +
                        ". Could not parse it. Required format is '<algorithm>;<graph_type>'.");
    RAVELOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
}

std::pair<std::string, std::string> parseParameterNameValue(const std::string& input_string)
{
  auto separator_pos = input_string.find('=');
  if (separator_pos == std::string::npos)
  {
    std::string error_msg("Could not parse parameter name and value from input string \"" + input_string +
                          "\". Could not find '='.");
    RAVELOG_ERROR(error_msg);
    throw std::invalid_argument(error_msg);
  }
  return {input_string.substr(0, separator_pos), input_string.substr(separator_pos + 1)};
}

std::istream& removePreceedingSymbol(std::istream& sinput, char symbol)
{
  while (sinput.good() and sinput.peek() == symbol)
  {
    sinput.ignore(1);
  };
  return sinput;
}

void ORMultiGraspMPPlugin::parseParameters(std::istream& sinput, mgsearch::MGGraphSearchMP::Parameters& params)
{
  const char delim = ' ';
  while (removePreceedingSymbol(sinput, delim).good())
  {
    std::stringstream ss;
    // read next substring until ' ' into ss
    sinput.get(*ss.rdbuf(), delim);
    // RAVELOG_DEBUG("READ STRING: " + ss.str() + std::string(" Fail flag is: ") + std::to_string(sinput.fail()));
    if (!sinput.fail())
    {  // in case we read something
      // try to get name and value as strings (throws invalid_argument if that does not work)
      auto [param_name, param_value] = parseParameterNameValue(ss.str());
      RAVELOG_DEBUG("Received parameter " + param_name + " with value " + param_value);
      if (param_name == "lambda")
      {
        params.lambda = std::stod(param_value);
      }
      else if (param_name == "sdf_file")
      {
        _sdf_filename = param_value;
      }
      else if (param_name == "batchsize")
      {
        params.batchsize = std::stoul(param_value);
      }
      else if (param_name == "log_file")
      {
        params.roadmap_log_path = param_value + "_roadmap";
        params.logfile_path = param_value + "_evaluation";
      }
      else
      {
        throw std::invalid_argument("Unknown parameter name " + param_name);
      }
    }
    else
    {
      RAVELOG_ERROR("Failed to read parameter-name-value-tuple from input stream. Read so far " + ss.str());
    }
  }
}

bool ORMultiGraspMPPlugin::initPlan(std::ostream& sout, std::istream& sinput)
{
  unsigned int robot_id;
  sinput >> robot_id;
  unsigned int obj_id;
  sinput >> obj_id;
  boost::lock_guard<OpenRAVE::EnvironmentMutex> lock(_original_env->GetMutex());
  auto obj_body = _original_env->GetBodyFromEnvironmentId(obj_id);
  if (!obj_body)
  {
    std::string error_msg = "Could not retrieve object with id " + std::to_string(obj_id);
    RAVELOG_ERROR(error_msg);
    throw std::runtime_error(error_msg);
  }
  auto robot = _original_env->GetBodyFromEnvironmentId(robot_id);
  if (!robot)
  {
    std::string error_msg = "Could not retrieve robot with id " + std::to_string(robot_id);
    RAVELOG_ERROR(error_msg);
    throw std::runtime_error(error_msg);
  }
  auto query_env = _original_env->CloneSelf(OpenRAVE::Clone_Bodies);
  query_env->StopSimulation();
  {
    auto cloned_robot = query_env->GetRobot(query_env->GetBodyFromEnvironmentId(robot_id)->GetName());
    auto cloned_obj = query_env->GetBodyFromEnvironmentId(obj_id);
    auto manip = cloned_robot->GetActiveManipulator();
    RAVELOG_DEBUG("Initializing plan for robot " + cloned_robot->GetName() + " with manipulator " + manip->GetName() +
                  " and object " + cloned_obj->GetName());
  }
  if (_algorithm_name == PARALLEL_BIRRT)
  {
    _planner = std::make_shared<ParallelMGBiRRT>(query_env, robot_id, obj_id);
  }
  else if (_algorithm_name == SEQUENTIAL_BIRRT)
  {
    _planner = std::make_shared<SequentialMGBiRRT>(query_env, robot_id, obj_id);
  }
  else
  {  // split algorithm name in algorithm + graph name for graph-based planner
    mgsearch::MGGraphSearchMP::Parameters params;
    parseGraphSearchType(_algorithm_name, params);
    parseParameters(sinput, params);
    _planner = std::make_shared<ORGraphSearch>(query_env, robot_id, obj_id, params, _sdf_filename);
  }
  return true;
}

bool ORMultiGraspMPPlugin::clear(std::ostream& sout, std::istream& sin)
{
  RAVELOG_DEBUG("Clearing all planner data");
  _planner.reset();
  _solutions.clear();
  return false;
}

bool ORMultiGraspMPPlugin::plan(std::ostream& sout, std::istream& sinput)
{
  double timeout;
  sinput >> timeout;
  std::vector<MultiGraspMP::Solution> new_paths;
  _planner->plan(new_paths, timeout);
  // TODO save stats to file or move this into plan(..)
  if (!new_paths.empty())
  {
    for (unsigned int i = 0; i < new_paths.size(); ++i)
    {
      auto new_sol = new_paths.at(i);
      sout << new_sol.goal_id;
      _solutions[new_sol.goal_id] = new_sol;
      if (i + 1 < new_paths.size())
        sout << " ";
    }
  }
  return true;
}

bool ORMultiGraspMPPlugin::pausePlanning(std::ostream& sout, std::istream& sinput)
{
  if (_planner)
  {
    _planner->pausePlanning();
  }
  return false;
}

bool ORMultiGraspMPPlugin::getPath(std::ostream& sout, std::istream& sinput)
{
  unsigned int id;
  sinput >> id;
  auto iter = _solutions.find(id);
  if (iter != _solutions.end())
  {
    MultiGraspMP::WaypointPathPtr path = iter->second.path;
    for (unsigned int wi = 0; wi < path->size(); ++wi)
    {
      auto& wp = path->at(wi);
      for (unsigned int i = 0; i < wp.size(); ++i)
      {
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

// bool ORMultiGraspMPPlugin::setParameters(std::ostream& sout, std::istream& sinput)
// {
// }

bool ORMultiGraspMPPlugin::addGrasp(std::ostream& sout, std::istream& sinput)
{
  if (!_planner)
    return false;
  MultiGraspMP::Grasp grasp;
  // first read id
  sinput >> grasp.id;
  // next read x, y, z
  sinput >> grasp.pos.x >> grasp.pos.y >> grasp.pos.z;
  // quaternion (expected to be w x y z, where w is the real part)
  // OpenRAVE expects the first component of a RaveVector (x, y, z, w) to be the real part.
  // So the statement below is correct, despite the confusing use of names.
  sinput >> grasp.quat.x >> grasp.quat.y >> grasp.quat.z >> grasp.quat.w;
  // finally read configuration
  while (sinput.good())
  {
    double q;
    sinput >> q;
    grasp.gripper_values.push_back(q);
  }
  // for debug purposes serialize grasp again
  RAVELOG_DEBUG("Adding new grasp: " + grasp.print());
  _planner->addGrasp(grasp);
  return false;  // TODO what to return?
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
  // next read goal quality
  sinput >> goal.quality;
  // finally read configuration
  while (sinput.good())
  {
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

bool ORMultiGraspMPPlugin::addWaypoints(std::ostream& sout, std::istream& sinput)
{
  if (!_planner)
    return false;
  std::vector<Config> configs;
  Config next;
  while (sinput.good())
  {
    if (sinput.peek() != '\n')
    {
      double q;
      sinput >> q;
      next.push_back(q);
    }
    else
    {
      sinput.ignore();
      configs.push_back(next);
      next.clear();
    }
  }
  RAVELOG_DEBUG("Adding " + std::to_string(configs.size()) + " new waypoints");
  _planner->addWaypoints(configs);
  return false;
}

bool ORMultiGraspMPPlugin::removeGoals(std::ostream& sout, std::istream& sinput)
{
  if (!_planner)
    return false;
  std::vector<unsigned int> goals_to_remove;
  std::stringstream debug_ss;
  while (sinput.good())
  {
    unsigned int id;
    sinput >> id;
    debug_ss << id << " ";
    goals_to_remove.push_back(id);
  }

  RAVELOG_DEBUG("Removing goals: " + debug_ss.str());
  _planner->removeGoals(goals_to_remove);
  return false;  // TODO what to return?
}

bool ORMultiGraspMPPlugin::saveStats(std::ostream& sout, std::istream& sinput)
{
  removePreceedingSymbol(sinput, ' ');
  std::stringstream ss;
  sinput.get(*ss.rdbuf());
  _planner->savePlanningStats(ss.str());
  return false;
}

bool ORMultiGraspMPPlugin::saveSolutions(std::ostream& sout, std::istream& sinput)
{
  removePreceedingSymbol(sinput, ' ');
  std::stringstream ss;
  sinput.get(*ss.rdbuf());
  // TODO there is probably a more elegant way to solve this
  std::vector<MultiGraspMP::Solution> sols;
  sols.reserve(_solutions.size());
  for (auto pair_elem : _solutions)
  {
    sols.push_back(pair_elem.second);
  }
  _planner->saveSolutions(sols, ss.str());
  return false;
}

InterfaceBasePtr CreateInterfaceValidated(InterfaceType type, const std::string& interfacename, std::istream& sinput,
                                          EnvironmentBasePtr penv)
{
  if (type == PT_Module)
  {
    // TODO check if interfacename is valid and if not return InterfaceBasePtr()
    return InterfaceBasePtr(new ORMultiGraspMPPlugin(penv, interfacename));
  }
  return InterfaceBasePtr();
}

void GetPluginAttributesValidated(PLUGININFO& info)
{
  // std::cout << "GetPluginAttributedValidated" << std::endl;
  info.interfacenames[PT_Module].push_back(PARALLEL_BIRRT);
  info.interfacenames[PT_Module].push_back(SEQUENTIAL_BIRRT);
  for (auto algo_graph_pair : mgsearch::MGGraphSearchMP::VALID_ALGORITHM_GRAPH_COMBINATIONS)
  {
    std::stringstream ss;
    ss << mgsearch::MGGraphSearchMP::getName(algo_graph_pair.first) << ';'
       << mgsearch::MGGraphSearchMP::getName(algo_graph_pair.second);
    info.interfacenames[PT_Module].push_back(ss.str());
  }
}

void DestroyPlugin()
{
  // TODO do we need to do anything here?
}