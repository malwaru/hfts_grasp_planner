#include "cnpy.h"
#include <cmath>
#include <hfts_grasp_planner/placement/mp/mgsearch/ImageStateSpace.h>

using namespace placement::mp::mgsearch;

ImageStateSpace::ImageStateSpace(const std::experimental::filesystem::path& root_path)
{
    init(root_path);
}

ImageStateSpace::~ImageStateSpace() = default;

void ImageStateSpace::init(const std::experimental::filesystem::path& root_path)
{
    std::map<unsigned int, ImagePtr> unsorted_images;
    // ensure the given path is a directory
    assert(std::experimental::filesystem::is_directory(root_path));
    // load grasp cost spaces
    for (auto& child_elem : std::experimental::filesystem::directory_iterator(root_path)) {
        if (std::experimental::filesystem::is_regular_file(child_elem.status())) {
            auto child_path = child_elem.path();
            if (child_path.extension() == std::experimental::filesystem::path(".npy")) {
                // extract grasp id from file name
                auto stem = child_path.stem().string();
                try {
                    unsigned int grasp_id = 0;
                    try {
                        grasp_id = std::stoul(stem);
                    } catch (const std::invalid_argument& e) {
                        throw std::runtime_error("Invalid filename - must be the grasp id.");
                    }
                    // check whether we already have an image for this
                    if (unsorted_images.find(grasp_id) != unsorted_images.end()) {
                        throw std::runtime_error("Grasp id " + std::to_string(grasp_id) + " found multiple times");
                    }
                    // load this array
                    cnpy::NpyArray npyarray = cnpy::npy_load(child_path.string());
                    if (npyarray.shape.size() != 2) {
                        throw std::runtime_error("Numpy array of invalid shape: " + std::to_string(npyarray.shape.size()));
                    } else if (npyarray.fortran_order) {
                        throw std::runtime_error("Numpy array must be cstyle");
                    } else if (npyarray.word_size != sizeof(double)) {
                        throw std::runtime_error("Numpy array has invalid word size. Must be " + std::to_string(sizeof(double)));
                    }
                    unsorted_images[grasp_id] = std::make_shared<Image>(npyarray.shape[1], npyarray.shape[0], npyarray.as_vec<double>());
                } catch (const std::runtime_error& e) {
                    std::cerr << "\033[1;31m[Error] Could not load " << child_path << ": " << e.what() << "\033[0m" << std::endl;
                    continue;
                }
            }
        }
    }
    // sort all images and store in member
    _images.resize(unsorted_images.size());
    for (auto& elem : unsorted_images) {
        _images.at(elem.first) = elem.second;
    }
}

unsigned int ImageStateSpace::getNumGrasps() const
{
    return _images.size();
}

bool ImageStateSpace::isValid(const Config& c) const
{
    assert(c.size() == 2);
    unsigned int i = (unsigned int)std::round(c[0]);
    unsigned int j = (unsigned int)std::round(c[1]);
    return _images.at(0)->at(i, j) > 0.0;
}

bool ImageStateSpace::isValid(const Config& c, unsigned int grasp_id, bool only_obj) const
{
    assert(c.size() == 2);
    unsigned int i = (unsigned int)std::round(c[0]);
    unsigned int j = (unsigned int)std::round(c[1]);
    return _images.at(grasp_id)->at(i, j) > 0.0;
}

// state cost
double ImageStateSpace::cost(const Config& a) const
{
    return conditional_cost(a, 0);
}

double ImageStateSpace::conditional_cost(const Config& a, unsigned int grasp_id) const
{
    assert(a.size() == 2);
    unsigned int i = (unsigned int)std::round(a[0]);
    unsigned int j = (unsigned int)std::round(a[1]);
    double val = _images.at(grasp_id)->at(i, j);
    if (val <= 0.0) {
        return std::numeric_limits<float>::infinity();
    }
    return val; // TODO reduce as the distance increases?
}

// distance
double ImageStateSpace::distance(const Config& a, const Config& b) const
{
    assert(a.size() == 2);
    assert(b.size() == 2);
    return std::abs((a[0] - b[0]) + (a[1] - b[1]));
}

// space information
unsigned int ImageStateSpace::getDimension() const
{
    return 2;
}

void ImageStateSpace::getBounds(Config& lower, Config& upper) const
{
    lower.resize(2);
    lower[0] = 0;
    lower[1] = 0;
    upper.resize(2);
    upper[0] = _images.at(0)->width - 0.5;
    upper[1] = _images.at(0)->height - 0.5;
}

void ImageStateSpace::getValidGraspIds(std::vector<unsigned int>& grasp_ids) const
{
    grasp_ids.clear();
    for (unsigned int i = 0; i < getNumGrasps(); ++i) {
        grasp_ids.push_back(i);
    }
}