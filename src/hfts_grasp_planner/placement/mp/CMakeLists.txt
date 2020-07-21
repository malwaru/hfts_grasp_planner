find_package(OpenRAVE REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread program_options)
find_package(ompl)
set(CMAKE_CXX_STANDARD 17)
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
  mgsearch/MultiGraspRoadmap.cpp
  mgsearch/Graphs.cpp
  mgsearch/LPAStar.cpp
  utils/Profiling.cpp
  mgsearch/MGGraphSearchMP.cpp
  mgsearch/ORStateSpace.cpp
  ORGraphSearch.cpp
  ompl/ORRedirectableBiRRT.cpp
  ../../external/ompl/geometric/RedirectableRRTConnect.cpp
  ../../external/halton/halton.cpp
)

set_target_properties(ormultigraspmp PROPERTIES COMPILE_FLAGS "${OpenRAVE_CXX_FLAGS} -Wall" LINK_FLAGS "${OpenRAVE_LINK_FLAGS}")
target_link_libraries(ormultigraspmp ${OpenRAVE_LIBRARIES} ${OMPL_LIBRARIES} ${catkin_LIBRARIES})

# add_executable(test_graph_search_algorithm
#   utils/Profiling.cpp
#   test/test_mplanner.cpp
# )

# target_link_libraries(test_graph_search_algorithm)

add_executable(test_image_space_algorithm
  # test/test_imagespace.cpp
  test/test_img_planner.cpp
  mgsearch/ImageStateSpace.cpp
  mgsearch/MGGraphSearchMP.cpp
  mgsearch/MultiGraspRoadmap.cpp
  mgsearch/Graphs.cpp
  mgsearch/LPAStar.cpp
  utils/Profiling.cpp
  MultiGraspMP.cpp
  ImgGraphSearch.cpp
  ../../external/halton/halton.cpp
)

set_target_properties(test_image_space_algorithm PROPERTIES COMPILE_FLAGS "${OpenRAVE_CXX_FLAGS} -Wall" LINK_FLAGS "${OpenRAVE_LINK_FLAGS}")
target_link_libraries(test_image_space_algorithm cnpy stdc++fs ${OpenRAVE_LIBRARIES} ${Boost_LIBRARIES} ${OMPL_LIBRARIES} ${catkin_LIBRARIES})