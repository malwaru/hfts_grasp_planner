#pragma once
#include <experimental/filesystem>
#include <hfts_grasp_planner/placement/mp/mgsearch/MultiGraspRoadmap.h>
#include <unordered_map>

namespace placement {
namespace mp {
    namespace mgsearch {
        class ImageStateSpace : public StateSpace {
        public:
            ImageStateSpace(const std::experimental::filesystem::path& root_path);
            ~ImageStateSpace();
            // grasp query
            unsigned int getNumGrasps() const;
            // State validity
            bool isValid(const Config& c) const override;
            bool isValid(const Config& c, unsigned int grasp_id, bool only_obj = false) const override;
            // state cost
            double cost(const Config& a) const override;
            double conditional_cost(const Config& a, unsigned int grasp_id) const override;
            // distance
            double distance(const Config& a, const Config& b) const override;
            // space information
            unsigned int getDimension() const override;
            void getBounds(Config& lower, Config& upper) const override;

        protected:
            struct Image {
                std::vector<double> data;
                unsigned int width;
                unsigned int height;
                Image()
                    : width(0)
                    , height(0)
                {
                }
                Image(unsigned int w, unsigned int h, const std::vector<double>& d)
                    : width(w)
                    , height(h)
                    , data(d)
                {
                }
                ~Image() = default;

                double& operator()(unsigned int i, unsigned int j)
                {
                    return data.at(i + j * width);
                }

                const double& operator()(unsigned int i, unsigned int j) const
                {
                    return data.at(i + j * width);
                }

                double& at(unsigned int i, unsigned int j)
                {
                    assert(not data.empty());
                    assert(i < width && j < height);
                    return operator()(i, j);
                }

                const double& at(unsigned int i, unsigned int j) const
                {
                    assert(not data.empty());
                    assert(i < width && j < height);
                    return operator()(i, j);
                }
            };
            typedef std::shared_ptr<Image> ImagePtr;

            std::vector<ImagePtr> _images;

            void init(const std::experimental::filesystem::path& root_path);
        };

        typedef std::shared_ptr<ImageStateSpace> ImageStateSpacePtr;
    }
}
}
