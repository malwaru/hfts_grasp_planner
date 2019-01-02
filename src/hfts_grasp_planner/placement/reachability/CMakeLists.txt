# project(reachability)
find_package(Eigen 3.2 REQUIRED)
find_package(ompl REQUIRED)
find_package(pybind11 REQUIRED)
set(CMAKE_CXX_STANDARD 14)
pybind11_add_module(yumi_reachability pyreachability.cpp reachability.cpp)
set_target_properties(yumi_reachability
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
target_include_directories(yumi_reachability PUBLIC ${ompl_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS})
target_link_libraries(yumi_reachability ompl)
