#include <boost/program_options.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <experimental/filesystem>
#include <hfts_grasp_planner/placement/mp/ImgGraphSearch.h>
#include <iostream>
#include <string>

namespace po = boost::program_options;
namespace mgs = placement::mp::mgsearch;
namespace pmp = placement::mp;
namespace fs = std::experimental::filesystem;

void printHelp(const po::options_description& po, const std::string& err_msg = "")
{
    if (not err_msg.empty()) {
        std::cerr << "ERROR: " << err_msg << std::endl;
    }
    // std::cout << "Usage: "
    po.print(std::cout);
}

int main(int argc, char** argv)
{
    po::options_description desc("Test multi-grasp motion planner on 2D image state spaces.");
    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("image-path", po::value<std::string>()->required(), "image file")
        ("start-config", po::value<std::vector<double>>()->multitoken(), "start configuration (2d)")
        ("goal-configs", po::value<std::vector<double>>()->multitoken()->required(), "end configurations (list of triplets (grasp_id, x, y))");
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        printHelp(desc);
        return 0;
    }
    try {
        po::notify(vm);
    } catch (boost::exception& err) {
        printHelp(desc, boost::diagnostic_information(err));
        return -1;
    }
    pmp::Config start;
    if (!vm.count("start-config")) {
        start.push_back(0.0);
        start.push_back(0.0);
        std::cout << "No start configuration provided. Will use (0.0f, 0.0f)" << std::endl;
    } else {
        start = vm.at("start-config").as<std::vector<double>>();
        if (start.size() != 2) {
            printHelp(desc, "Start configuration has an invalid dimension.");
            return 0;
        }
        std::cout << "Using (" << start.at(0) << ", " << start.at(1) << ") as start." << std::endl;
    }
    // parse goals
    std::vector<pmp::MultiGraspMP::Goal> goals;
    std::vector<double> goal_values = vm.at("goal-configs").as<std::vector<double>>();
    if (goal_values.size() % 3 != 0) {
        printHelp(desc, "Goal configurations must be a list of triplets");
        return -1;
    }
    for (unsigned gid = 0; gid < goal_values.size() / 3; ++gid) {
        pmp::MultiGraspMP::Goal goal;
        goal.id = gid;
        goal.grasp_id = (unsigned int)goal_values.at(gid * 3);
        goal.config.push_back(goal_values.at(gid * 3 + 1));
        goal.config.push_back(goal_values.at(gid * 3 + 2));
        goal.quality = 0.0; // TODO set goal quality
        goals.push_back(goal);
    }
    fs::path image_file_path(vm["image-path"].as<std::string>());
    std::cout << "Received input path: " << image_file_path << std::endl;
    pmp::mgsearch::MGGraphSearchMP::Parameters params;
    params.algo_type = pmp::mgsearch::MGGraphSearchMP::AlgorithmType::Astar;
    params.graph_type = pmp::mgsearch::MGGraphSearchMP::GraphType::SingleGraspGraph;
    pmp::ImgGraphSearch imgs(image_file_path, start, params);
    for (auto& goal : goals ) {
        imgs.addGoal(goal);
    }
    std::vector<pmp::MultiGraspMP::Solution> sols;
    imgs.plan(sols, 0.0);
    if (sols.empty()) {
        std::cout << "No solution found :(" << std::endl;
    } else {
        std::cout << "Solution found!" << std::endl;
        std::cout << "  Goal id: " << sols.at(0).goal_id << std::endl;
        std::cout << "  cost: " << sols.at(0).cost << std::endl;
    }
    // mgs::ImageStateSpace state_space(image_file_path);
    // std::cout << "Read in " << state_space.getNumGrasps() << " grasps" << std::endl;
    // pmp::Config lower;
    // pmp::Config upper;
    // state_space.getBounds(lower, upper);
    // std::cout << "Image dimensions are [" << lower[0] << ", " << lower[1] << "], [ "
    //           << upper[0] << ", " << upper[1] << "]" << std::endl;
    // // print images
    // std::cout << " Printing read in images " << std::endl;
    // for (float x = lower[0]; x < upper[0]; x += 1.0){
    //     for (float y = lower[1]; y < upper[1]; y += 1.0) {
    //         std::cout << state_space.conditional_cost({x, y}, 1) << ", ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
