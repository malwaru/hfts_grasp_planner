#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <hfts_grasp_planner/placement/mp/mgsearch/ImageStateSpace.h>
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
    po::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("image-path", po::value<std::string>(), "image file");
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
    fs::path image_file_path(vm["image-path"].as<std::string>());
    std::cout << "Received input path: " << image_file_path << std::endl;
    mgs::ImageStateSpace state_space(image_file_path);
    std::cout << "Read in " << state_space.getNumGrasps() << " grasps" << std::endl;
    pmp::Config lower;
    pmp::Config upper;
    state_space.getBounds(lower, upper);
    std::cout << "Image dimensions are [" << lower[0] << ", " << lower[1] << "], [ " 
              << upper[0] << ", " << upper[1] << "]" << std::endl;
    // print images
    std::cout << " Printing read in images " << std::endl;
    for (float x = lower[0]; x < upper[0]; x += 1.0){
        for (float y = lower[1]; y < upper[1]; y += 1.0) {
            std::cout << state_space.conditional_cost({x, y}, 1) << ", ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}