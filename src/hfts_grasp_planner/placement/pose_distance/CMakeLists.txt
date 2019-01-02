project(yumi_reachability C)

# add_definitions("-fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(yumi_pose_distance SHARED yumi_pose_distance.c)