#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <time.h>  // only correct on linux
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
    double runtimes_sum;
    double clocktimes_sum;

    FunctionProfile();
    /**
     * Return the average wall clock time.
     */
    double avgRuntime() const;

    /**
     * Return the total wall clock time.
     */
    double totalRuntime() const;

    /**
     * Return the average CPU time.
     */
    double avgClockTime() const;

    /**
     * Return the total CPU time.
     */
    double totalClockTime() const;
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

  /**
   * Dump all profiles in a csv file.
   * @param os - output stream.
   * @param dump_header - if true, first dump a line with header names, else dump data only
   */
  static void dumpProfiles(std::ostream& os, bool dump_header = true);

private:
  const std::string _function_name;
  const bool _aggregate;
  static std::map<std::string, FunctionProfile> profile_data;
  std::chrono::steady_clock::time_point _start;
  clock_t _start_t;
};
}  // namespace utils
}  // namespace mp
}  // namespace placement
std::ostream& operator<<(std::ostream& ostr, const placement::mp::utils::ScopedProfiler::FunctionProfile& fp);