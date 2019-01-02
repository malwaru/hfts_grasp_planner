cmake_minimum_required(VERSION 3.10)
# project(reachability)
find_package(Boost REQUIRED)
find_package(pybind11 REQUIRED)
# find_package(PythonLibs REQUIRED)
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_COMPILER /usr/bin/clang++)
pybind11_add_module(clearance_utils clearance.cpp)
set_target_properties(clearance_utils
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
# target_include_directories(clearance_utils PUBLIC ${BOOST_INCLUDE_DIRS}})
# target_link_libraries(clearance_utils ${BOOST_LIBRARIES})
