#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace placement
{
namespace mp
{
namespace utils
{
/**
 * Keep track of how often a function has been called and
 * how much time each execution took.
 * Usage:
 * Simply instantiate a scoped profiler at the beginning of your function by passing
 * the name of your funtion.
 */
class ScopedProfiler
{
public:
  struct FunctionProfile
  {
    std::string function_name;
    unsigned int num_calls;
    std::vector<double> runtimes;

    FunctionProfile();
    double avgRuntime() const;
    double totalRuntime() const;
  };
  /**
   * Creates a new instance of a profiler. This will start recording the duration
   * the object is in scope and store it in a static map.
   * @param function_name - a globally unique function name under which to log the data
   * @param aggregate - if true, the call data will be aggregated with previous calls, otherwise
   *  any previously generated profile for the same function name is overwritten
   */
  ScopedProfiler(const std::string& function_name, bool aggregate = true);
  ~ScopedProfiler();

  /**
   * Get the function profile for the given function.
   * If there is no profile, the profile will be empty.
   */
  static FunctionProfile getProfile(const std::string& function_name);
  /**
   * Remove all stored profiles.
   */
  static void clearProfiles();

  /**
   * Print all profiles to a stream.
   * @param os - output stream
   * @param summary - if true, print summary, i.e. averages, else print raw data
   */
  static void printProfiles(std::ostream& os, bool summary = true);

private:
  const std::string _function_name;
  const bool _aggregate;
  static std::map<std::string, FunctionProfile> profile_data;
  std::chrono::high_resolution_clock::time_point _start;
};
}  // namespace utils
}  // namespace mp
}  // namespace placement
std::ostream& operator<<(std::ostream& ostr, const placement::mp::utils::ScopedProfiler::FunctionProfile& fp);