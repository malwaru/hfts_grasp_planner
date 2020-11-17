#pragma once
#include <openrave/plugin.h>

#include <openrave/openrave.h>

#include <boost/bind.hpp>
#include <hfts_grasp_planner/placement/mp/MultiGraspMP.h>
#include <unordered_map>

namespace placement
{
namespace mp
{
class ORMultiGraspMPPlugin : public OpenRAVE::ModuleBase
{
public:
  ORMultiGraspMPPlugin(OpenRAVE::EnvironmentBasePtr penv, const std::string& algorithm);
  ~ORMultiGraspMPPlugin();

  /**
   * Initialize planner for a new problem given the current state of the environment.
   * Resets all previously stored information, i.e. all grasps, goals and all motion planning data.
   * You need to add goals using the addGoal function. In addition, you need to add at least one grasp.
   * Input format: robot_name obj_name param_name_1=value_1 ... param_name_n=value_n
   * where
   *  robot_id, int - environment id of the robot to plan for (planning for its active manipulator)
   *  obj_name, int - environment id of the kinbody that is going to be grasped
   *  param_name_i - name of algorithm-specific parameter (without whitespaces)
   *  param_value_i - value of that parameter (either a string representation of an int, float or a string itself)
   */
  bool initPlan(std::ostream& sout, std::istream& sinput);

  /**
   * Free all allocated resources. After calling this, you need to call initPlan again.
   */
  bool clear(std::ostream& sout, std::istream& sin);

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
   * In case the underlying planner is run asynchronously, this function notifies
   * the planner to pause planning until either plan is called again, or the planner is destructed.
   */
  bool pausePlanning(std::ostream& sout, std::istream& sinput);

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

  /**
   * Set additional parameters for the algorithm.
   * Input format: <param_name_1>=<param_value_1>
   *               ...
   *               <param_name_n>=<param_value_n>
   * where
   *  <param_name_i> is a string without whitespaces representing the parameter name
   *  <param_value_i> is a string representation of an integer, float or simply a string
   */
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
   * Input format: id gid quality q0 ... qn
   * where
   *  id, int - unique identifier for this goal
   *  gid, int - grasp id for which this goal is defined
   *  quality, double - quality of the goal
   *  q0, ..., qn, double - goal arm configuration
   */
  bool addGoal(std::ostream& sout, std::istream& sinput);

  /**
   * Add sample configurations that are not a goal but may be useful for the planner to use.
   * The configurations may be in collision.
   * Input format: q0_0 ... qn_0\n
   *               q0_1 ... qn_1\n
   *               ...
   *               q0_k ... qn_k\n
   * where
   *  qi_j, double - the ith value of the jth configuration
   */
  bool addWaypoints(std::ostream& sout, std::istream& sinput);

  /**
   * Inform the motion planner to stop planning towards the given goals.
   * Input format: id0 id1 id2 ... idn
   * where
   *  idX, int - goal identifiers
   */
  bool removeGoals(std::ostream& sout, std::istream& sinput);

  /**
   * Save planning statistics since the last time initPlanner was called to file.
   * Input format: <filename>
   * where
   *  <filename>, string - name of a file in which to store the statistics
   */
  bool saveStats(std::ostream& sout, std::istream& sinput);

  /**
   * Save all solutions that have been computed since the last time initPlanner was called to file.
   * Input format: <filename>
   * where
   *  <filename>, string - name of a file to store solutions in (as csv)
   */
  bool saveSolutions(std::ostream& sout, std::istream& sinput);

private:
  std::string _algorithm_name;
  std::string _sdf_filename;
  MultiGraspMPPtr _planner;
  std::unordered_map<unsigned int, MultiGraspMP::Solution> _solutions;
  OpenRAVE::EnvironmentBasePtr _original_env;
  /**
   * Read parameters (name=value) from sinput and store in params or in member variables.
   * Throws invalid_argument, out_of_range
   */
  void parseParameters(std::istream& sinput, mgsearch::MGGraphSearchMP::Parameters& params);
};
OpenRAVE::InterfaceBasePtr CreateInterfaceValidated(OpenRAVE::InterfaceType type, const std::string& interfacename,
                                                    std::istream& sinput, OpenRAVE::EnvironmentBasePtr penv);
void GetPluginAttributesValidated(OpenRAVE::PLUGININFO& info);
}  // namespace mp
}  // namespace placement