#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>

using namespace placement::mp::utils;
ScopedProfiler::FunctionProfile::FunctionProfile()
    : function_name("")
    , num_calls(0)
{
}
// ScopedProfiler::FunctionProfile::~FunctionProfile() = default;

// ScopedProfiler::FunctionProfile::FunctionProfile(const ScopedProfiler::FunctionProfile& other) = default;
double ScopedProfiler::FunctionProfile::avgRuntime() const
{
    if (num_calls > 0) {
        double sum = std::accumulate(runtimes.begin(), runtimes.end(), 0.0);
        return sum / num_calls;
    }
    return 0.0;
}

double ScopedProfiler::FunctionProfile::totalRuntime() const
{
    return std::accumulate(runtimes.begin(), runtimes.end(), 0.0);
}

std::ostream& operator<<(std::ostream& ostr, const ScopedProfiler::FunctionProfile& fp)
{
    ostr << fp.function_name << ": " << fp.num_calls << " ";
    for (auto val : fp.runtimes) {
        ostr << val << " ";
    }
    return ostr;
}

// define profile data
std::map<std::string, ScopedProfiler::FunctionProfile> ScopedProfiler::profile_data;

ScopedProfiler::ScopedProfiler(const std::string& function_name, bool aggregate)
    : _function_name(function_name)
    , _aggregate(aggregate)
    , _start(std::chrono::high_resolution_clock::now())
{
}

ScopedProfiler::~ScopedProfiler()
{
    auto iter = profile_data.find(_function_name);
    if (iter == profile_data.end()) {
        FunctionProfile new_profile;
        profile_data[_function_name] = new_profile;
    }
    if (!_aggregate) {
        profile_data[_function_name].num_calls = 0;
        profile_data[_function_name].runtimes.clear();
    }
    profile_data[_function_name].num_calls += 1;
    using floatingsecs = std::chrono::duration<double>;
    auto duration = std::chrono::duration_cast<floatingsecs>(std::chrono::high_resolution_clock::now() - _start);
    profile_data[_function_name].runtimes.push_back(duration.count());
}

ScopedProfiler::FunctionProfile ScopedProfiler::getProfile(const std::string& function_name)
{
    auto iter = profile_data.find(function_name);
    if (iter == profile_data.end()) {
        FunctionProfile empty_data;
        empty_data.function_name = function_name;
        return empty_data;
    }
    return iter->second;
}

void ScopedProfiler::clearProfiles()
{
    profile_data.clear();
}

void ScopedProfiler::printProfiles(std::ostream& ost, bool summary)
{
    for (auto& profile : profile_data) {
        if (summary) {
            ost << profile.first << ": "
                << "#calls:" << profile.second.num_calls
                << " total:" << profile.second.totalRuntime() << "s"
                << " avg:" << profile.second.avgRuntime() << "s"
                << std::endl;
        } else {
            ost << profile.second << std::endl;
        }
    }
}
