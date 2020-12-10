#include <boost/exception/diagnostic_information.hpp>
#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <hfts_grasp_planner/placement/mp/ImgGraphSearch.h>
#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <string>

namespace po = boost::program_options;
namespace mgs = placement::mp::mgsearch;
namespace pmp = placement::mp;
namespace fs = std::experimental::filesystem;

void printHelp(const po::options_description& po, const std::string& err_msg = "")
{
  if (not err_msg.empty())
  {
    std::cerr << "ERROR: " << err_msg << std::endl;
  }
  // std::cout << "Usage: "
  po.print(std::cout);
}

int main(int argc, char** argv)
{
  po::options_description desc("Test multi-grasp motion planner on 2D image state spaces.");
  bool print_profile = false;
  // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("image_path", po::value<std::string>()->required(), "image file")
        ("start_config", po::value<std::vector<double>>()->multitoken(), "start configuration (2d)")
        ("goal_configs", po::value<std::vector<double>>()->multitoken(), "end configurations (list of quadrets (grasp_id, x, y, quality))")
        ("configs_file", po::value<std::string>()->default_value(""), "Optionally path to configuration file containing start and goal configurations")
        ("lambda", po::value<double>()->default_value(1.0), "Scaling factor between path and goal cost.")
        ("algorithm_type", po::value<std::string>(), "Algorithm name")
        ("graph_type", po::value<std::string>(), "Graph name")
        ("edge_selector_type", po::value<std::string>()->default_value("LastUnknown"), "Edge selector type for LazySP")
        ("roadmap_log_file", po::value<std::string>()->default_value("/tmp/roadmap"), "Filename to log roadmap to")
        ("evaluation_log_file", po::value<std::string>()->default_value("/tmp/evaluation_log"), "Filename to log roadmap evaluations to")
        ("stats_file", po::value<std::string>()->default_value("/tmp/stats_log"), "Filename to log runtime statistics to")
        ("results_file", po::value<std::string>()->default_value("/tmp/results_log"), "Filename to log planning results to")
        ("print_profile", po::bool_switch(&print_profile), "If set, print profiling information");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help"))
  {
    printHelp(desc);
    return 0;
  }
  try
  {
    po::notify(vm);
  }
  catch (boost::exception& err)
  {
    printHelp(desc, boost::diagnostic_information(err));
    return -1;
  }
  // load start and goal either from configuration file or from command line arguments
  pmp::Config start;
  std::vector<pmp::MultiGraspMP::Goal> goals;
  if (vm.count("configs_file"))
  {  // read start and goals from yaml
    YAML::Node config_file = YAML::LoadFile(vm.at("configs_file").as<std::string>());
    if (!config_file["start_config"] or not config_file["start_config"].IsSequence() or
        config_file["start_config"].size() != 2)
    {
      printHelp(desc, "The configuration file has no or an invalid entry 'start_config'");
      return 0;
    }
    for (auto sub_entry : config_file["start_config"])
    {
      start.push_back(sub_entry.as<double>());
    }
    if (not config_file["goal_configs"] or not config_file["goal_configs"].IsSequence())
    {
      printHelp(desc, "The configuration file has no or an invalid entry 'goal_configs");
      return 0;
    }
    unsigned int goal_id = 0;
    for (auto goal_node : config_file["goal_configs"])
    {
      if (goal_node.size() != 4)
      {
        printHelp(desc, "There is an invalid goal configuration in the configuration file. Make sure each goal is a "
                        "quadret (grasp_id, x, y, quality)");
        return 0;
      }
      pmp::MultiGraspMP::Goal goal;
      auto goal_data = goal_node.as<std::vector<double>>();
      goal.id = goal_id++;
      goal.grasp_id = static_cast<unsigned int>(goal_data.at(0));
      goal.config.push_back(goal_data.at(1));
      goal.config.push_back(goal_data.at(2));
      goal.quality = goal_data.at(3);
      goals.push_back(goal);
    }
  }
  else
  {  // read from command line
    if (!vm.count("start_config"))
    {
      start.push_back(0.0);
      start.push_back(0.0);
      std::cout << "No start configuration provided. Will use (0.0f, 0.0f)" << std::endl;
    }
    else
    {
      start = vm.at("start_config").as<std::vector<double>>();
      if (start.size() != 2)
      {
        printHelp(desc, "Start configuration has an invalid dimension.");
        return 0;
      }
      std::cout << "Using (" << start.at(0) << ", " << start.at(1) << ") as start." << std::endl;
    }
    // parse goals
    std::vector<double> goal_values = vm.at("goal_configs").as<std::vector<double>>();
    if (goal_values.size() % 4 != 0)
    {
      printHelp(desc, "Goal configurations must be a list of quadrets");
      return -1;
    }
    for (unsigned gid = 0; gid < goal_values.size() / 4; ++gid)
    {
      pmp::MultiGraspMP::Goal goal;
      goal.id = gid;
      goal.grasp_id = (unsigned int)goal_values.at(gid * 4);
      goal.config.push_back(goal_values.at(gid * 4 + 1));
      goal.config.push_back(goal_values.at(gid * 4 + 2));
      goal.quality = goal_values.at(gid * 4 + 3);
      goals.push_back(goal);
    }
  }
  // parse remaining inputs
  fs::path image_file_path(vm["image_path"].as<std::string>());
  std::cout << "Received input path: " << image_file_path << std::endl;
  pmp::mgsearch::MGGraphSearchMP::Parameters params;
  params.algo_type = pmp::mgsearch::MGGraphSearchMP::getAlgorithmType(vm["algorithm_type"].as<std::string>());
  params.graph_type = pmp::mgsearch::MGGraphSearchMP::getGraphType(vm["graph_type"].as<std::string>());
  params.edge_selector_type =
      pmp::mgsearch::MGGraphSearchMP::getEdgeSelectorType(vm["edge_selector_type"].as<std::string>());
  params.lambda = vm["lambda"].as<double>();
  params.roadmap_log_path = vm["roadmap_log_file"].as<std::string>();
  params.logfile_path = vm["evaluation_log_file"].as<std::string>();
  // create search and add goals
  pmp::ImgGraphSearch imgs(image_file_path, start, params);
  for (auto& goal : goals)
  {
    imgs.addGoal(goal);
  }
  // plan
  std::vector<pmp::MultiGraspMP::Solution> sols;
  imgs.plan(sols, 0.0);
  if (sols.empty())
  {
    std::cout << "No solution returned :(" << std::endl;
  }
  else
  {
    std::cout << "Solution returned!" << std::endl;
    std::cout << "  Goal id: " << sols.at(0).goal_id << std::endl;
    std::cout << "  cost: " << sols.at(0).cost << std::endl;
  }
  if (print_profile)
  {
    pmp::utils::ScopedProfiler::printProfiles(std::cout, true);
  }
  imgs.savePlanningStats(vm["stats_file"].as<std::string>());
  if (not sols.empty())
  {  // log planning results
    auto results_file_path = vm["results_file"].as<std::string>();
    imgs.saveSolutions(sols, std::experimental::filesystem::path(results_file_path));
  }
  return 0;
}
