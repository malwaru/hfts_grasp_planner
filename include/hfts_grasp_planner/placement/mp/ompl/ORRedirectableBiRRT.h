#pragma once
#include <hfts_grasp_planner/external/ompl/geometric/RedirectableRRTConnect.h>
#include <openrave/openrave.h>

namespace placement {
namespace mp {
    namespace ompl {
        class ORRedirectableBiRRT {
        public:
            ORRedirectableBiRRT(OpenRAVE::EnvironmentBasePtr penv);
            ~ORRedirectableBiRRT();
            bool plan(std::vector<std::vector<float>>& path);
            /**
             *  Return the number of unreached goals.
             */
            unsigned int getNumGoals() const;

            void addGoal(const std::vector<float>& config);
            void addGoals(const std::vector<std::vector<float>>& config);
            void removeGoal(const std::vector<float>& config);
            void removeGoals(const std::vector<std::vector<float>>& config);
        };

        typedef std::shared_ptr<ORRedirectableBiRRT> ORRedirectableBiRRTPtr;
    }
}
}
