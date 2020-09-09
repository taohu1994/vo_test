#ifndef MY_VO_FUNCTION_H_
#define MY_VO_FUNCTION_H_
#include <iostream>
#include <algorithm>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>
#include <boost/format.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

class stereo_camera{
public:
    Eigen::Matrix<double,3,3> fundamental_matrix;
    Eigen::Matrix<double,3,3> instrinct_matrix;
    double focal_length;
    double baseline;
    double pixel_size;
};

bool PixelPair23dPoint(Eigen::Matrix3d K, double focal_length, double baseline,cv::Point2f point_left, cv::Point2f point_right,double pixel_size, Eigen::Vector3d *points_3d)
{
    Eigen::Vector3d point_homo;
    //  double focal_length = camera_instrinct
    // double baseline = camera_instrinct(1,1)
    double diff = double(point_left.x - point_right.x)*pixel_size;
    double depth = focal_length*baseline/diff;
    Eigen::Vector3d point_3d;
    if (depth > 0.1)
    {
        point_homo(0) = point_left.x;
        point_homo(1) = point_left.y;
        point_homo(2) = 1.0;
        *points_3d = depth*K.inverse()*point_homo;
        //points_3d = &point_3d;
        return true;
    }
    else
    {
        return false;
    }
}


/*
*/

#endif
