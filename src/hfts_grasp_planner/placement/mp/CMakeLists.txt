find_package(OpenRAVE REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(OMPL)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER clang++)

if (NOT ${OMPL_FOUND})
  find_package(ompl REQUIRED)
  set (OMPL_LIBRARIES ${ompl_LIBRARIES})
endif()
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${OpenRAVE_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

link_directories(${OpenRAVE_LIBRARY_DIRS})

# add_library(ormultigraspmp SHARED
#   ORMultiGraspMPPlugin.cpp
#   MultiGraspMP.cpp
#   MultiGraspBiRRT.cpp
# )
openrave_plugin(ormultigraspmp
  ORMultiGraspMPPlugin.cpp
  MultiGraspMP.cpp
  MultiGraspBiRRT.cpp
  ompl/ORRedirectableBiRRT.cpp
  ../../external/ompl/geometric/RedirectableRRTConnect.cpp
)

set_target_properties(ormultigraspmp PROPERTIES COMPILE_FLAGS "${OpenRAVE_CXX_FLAGS}" LINK_FLAGS "${OpenRAVE_LINK_FLAGS}")
target_link_libraries(ormultigraspmp ${OpenRAVE_LIBRARIES} ${OMPL_LIBRARIES} ${catkin_LIBRARIES})