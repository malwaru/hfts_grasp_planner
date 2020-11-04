#include <hfts_grasp_planner/placement/mp/utils/Profiling.h>

using namespace placement::mp::utils;
ScopedProfiler::FunctionProfile::FunctionProfile()
  : function_name(""), num_calls(0), runtimes_sum(0.0), clocktimes_sum(0.0)
{
}
// ScopedProfiler::FunctionProfile::~FunctionProfile() = default;

// ScopedProfiler::FunctionProfile::FunctionProfile(const ScopedProfiler::FunctionProfile& other) = default;
double ScopedProfiler::FunctionProfile::avgRuntime() const
{
  if (num_calls > 0)
  {
    return runtimes_sum / num_calls;
  }
  return 0.0;
}

double ScopedProfiler::FunctionProfile::avgClockTime() const
{
  if (num_calls > 0)
  {
    return clocktimes_sum / num_calls;
  }
  return 0.0;
}

double ScopedProfiler::FunctionProfile::totalRuntime() const
{
  return runtimes_sum;
}

double ScopedProfiler::FunctionProfile::totalClockTime() const
{
  return clocktimes_sum;
}

std::ostream& operator<<(std::ostream& ostr, const ScopedProfiler::FunctionProfile& fp)
{
  return ostr << fp.function_name << ": calls=" << fp.num_calls << " runtimes_sum=" << fp.runtimes_sum
              << " clocktimes=" << fp.clocktimes_sum;
}

// define profile data
std::map<std::string, ScopedProfiler::FunctionProfile> ScopedProfiler::profile_data;

ScopedProfiler::ScopedProfiler(const std::string& function_name, bool aggregate)
  : _function_name(function_name), _aggregate(aggregate), _start(std::chrono::steady_clock::now()), _start_t(clock())
{
}

ScopedProfiler::~ScopedProfiler()
{
  auto iter = profile_data.find(_function_name);
  if (iter == profile_data.end())
  {
    FunctionProfile new_profile;
    profile_data[_function_name] = new_profile;
  }
  if (!_aggregate)
  {
    profile_data[_function_name].num_calls = 0;
    profile_data[_function_name].runtimes_sum = 0.0;
    profile_data[_function_name].clocktimes_sum = 0.0;
  }
  profile_data[_function_name].num_calls += 1;
  using floatingsecs = std::chrono::duration<double>;
  auto duration = std::chrono::duration_cast<floatingsecs>(std::chrono::steady_clock::now() - _start);
  profile_data[_function_name].runtimes_sum += duration.count();
  profile_data[_function_name].clocktimes_sum += double(clock() - _start_t) / CLOCKS_PER_SEC;
}

ScopedProfiler::FunctionProfile ScopedProfiler::getProfile(const std::string& function_name)
{
  auto iter = profile_data.find(function_name);
  if (iter == profile_data.end())
  {
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
  for (auto& profile : profile_data)
  {
    if (summary)
    {
      ost << profile.first << ": "
          << "#calls:" << profile.second.num_calls << " total_runtime:" << profile.second.totalRuntime() << "s"
          << " avg_runtime:" << profile.second.avgRuntime() << "s"
          << " total_cpu_time:" << profile.second.totalClockTime() << " avg_cpu_time:" << profile.second.avgClockTime()
          << std::endl;
    }
    else
    {
      ost << profile.second << std::endl;
    }
  }
}

void ScopedProfiler::dumpProfiles(std::ostream& os)
{
  // dump headers first
  os << "function_name, num_calls, total_runtime, avg_runtime, total_cpu_time, avg_cpu_time\n";
  // dump aggregated profiles
  for (auto& profile : profile_data)
  {
    os << profile.first << ", " << profile.second.num_calls << ", " << profile.second.totalRuntime() << ", "
       << profile.second.avgRuntime() << ", " << profile.second.totalClockTime() << ", "
       << profile.second.avgClockTime() << "\n";
  }
}