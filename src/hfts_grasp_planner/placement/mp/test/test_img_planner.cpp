#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <hfts_grasp_planner/placement/mp/ImgGraphSearch.h>
#include <iostream>
#include <string>

namespace po = boost::program_options;
namespace mgs = placement::mp::mgsearch;
namespace pmp = placement::mp;
namespace fs = std::experimental::filesystem;

void printHelp(const po::options_description& po)
{
    // std::cout << "Usage: "
    po.print(std::cout);
}

int main(int argc, char** argv)
{
    po::options_description desc("Test multi-grasp motion planner on 2D image state spaces.");
    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("image-path", po::value<std::string>(), "image file")
        ("start-config", po::value<std::vector<double>>()->multitoken(), "start configuration (2d)");
    // clang-format on  
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        printHelp(desc);
        return 0;
    }
    if (!vm.count("image-path")) {
        printHelp(desc);
        return 0;
    }
    pmp::Config start;
    if (!vm.count("start-config")) {
        start.push_back(0.0);
        start.push_back(0.0);
        std::cout << "No start configuration provided. Will use (0.0f, 0.0f)" << std::endl;
    } else {
        start = vm.at("start-config").as<std::vector<double>>();
        if (start.size() != 2) {
            std::cerr << "Start configuration has an invalid dimension." << std::endl;
            printHelp(desc);
            return 0;
        }
        std::cout << "Using (" << start.at(0) << ", " << start.at(1) << ") as start." << std::endl;
    }
    fs::path image_file_path(vm["image-path"].as<std::string>());
    std::cout << "Received input path: " << image_file_path << std::endl;
    pmp::mgsearch::MGGraphSearchMP::Parameters params;
    params.algo_type = pmp::mgsearch::MGGraphSearchMP::AlgorithmType::Astar;
    params.graph_type = pmp::mgsearch::MGGraphSearchMP::GraphType::SingleGraspGraph;
    pmp::ImgGraphSearch imgs(image_file_path, start, params);
    std::vector<pmp::MultiGraspMP::Solution> sols;
    imgs.plan(sols, 0.0);
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
