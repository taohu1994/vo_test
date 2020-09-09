
cmake_minimum_required (VERSION 2.8)
project (vo_test)

#include opencv
find_package( OpenCV 4 REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#include eigen
include_directories("/usr/include/eigen3/")
include_directories("/home/thomas/Desktop/SLAM/vo_test/src/")
#include pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

#include ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
link_directories(${CERS_INCLUDE_DIRS})
add_definitions(${CERES_DEFINITIONS})

add_executable (vo_test ../src/vo_test.cpp ../src/my_vo_function.cpp)
target_link_libraries (vo_test ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}) 
