find_package(OpenRAVE REQUIRED)
find_package(OMPL REQUIRED)
include_directories(
  ${OpenRAVE_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
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
  ../../external/ompl/geometric/RedirectableRRTConnect.cpp
)

set_target_properties(ormultigraspmp PROPERTIES COMPILE_FLAGS "${OpenRAVE_CXX_FLAGS}" LINK_FLAGS "${OpenRAVE_LINK_FLAGS}")
target_link_libraries(ormultigraspmp ${OpenRAVE_LIBRARIES} ${OMPL_LIBRARIES})