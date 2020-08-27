
cmake_minimum_required (VERSION 2.8)
project (vo_test)

#include opencv
find_package( OpenCV 4 REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#include eigen
include_directories("/usr/include/eigen3/")

#include pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (vo_test ../src/vo_test.cpp)
target_link_libraries (vo_test ${OpenCV_LIBS} ${PCL_LIBRARIES}) 
