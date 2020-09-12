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
#include <pcl-1.8/pcl/common/common_headers.h>
#include <pcl-1.8/pcl/features/normal_3d.h>
#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/console/parse.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
/*
class state{
public:
    Eigen::Matrix<double,6,1> state;
    
    
    void init()
    {
    Eigen::Matrix<double,6,1> state_temp;
    for(int i =0; i<6; i++)
    {
        state_temp = 0;
    }
    state = state_temp;
    }
};
*/
/*struct ProjectionError_3d3d {
    ProjectionError_3d3d(const double* curr_points,const double* prev_points)
    : curr_points_(curr_points), prev_points_(prev_points) {}
    
    template <typename T>
    bool operator()(const T* const rotate_R,
                    const T* const transpose_T,
                    T* residuals) const {
                        T rotated_points[3];
                        T prev_poinst_replace[3];
                       // ceres::QuaternionRotatePoint(rotate_R,prev_poinst_replace,rotated_points);
                        rotated_points[0] = rotate_R[0]*T(prev_points_[0])+rotate_R[1]*T(prev_points_[1])+rotate_R[3]*T(prev_points_[2]);
                        rotated_points[1] = rotate_R[3]*T(prev_points_[0])+rotate_R[4]*T(prev_points_[1])+rotate_R[5]*T(prev_points_[2]);
                        rotated_points[2] = rotate_R[6]*T(prev_points_[0])+rotate_R[7]*T(prev_points_[1])+rotate_R[8]*T(prev_points_[2]);
                        rotated_points[0] += (transpose_T[0]);
                        rotated_points[1] += (transpose_T[1]);
                        rotated_points[2] += (transpose_T[2]);
                        
                        T points_diff[3];
                        
                        points_diff[0] = T(curr_points_[0]) - rotated_points[0];
                        points_diff[1] = T(curr_points_[1]) - rotated_points[1];
                        points_diff[2] = T(curr_points_[2]) - rotated_points[2];
                        
                        residuals[0] = points_diff[0]*points_diff[0]+points_diff[1]*points_diff[1]+points_diff[2]*points_diff[2];
                        return true;
                    }
                    // Factory to hide the construction of the CostFunction object from
                    // the client code.
                    static ceres::CostFunction* Create(const double* curr_points,
                                                       const double* prev_points) {
                        return (new ceres::AutoDiffCostFunction<ProjectionError_3d3d, 1, 9, 3>(
                            new ProjectionError_3d3d(curr_points, prev_points)));
                                                       }
private:                                                     
    const double* curr_points_;
    const double* prev_points_;
};
*/
struct ProjectionError_3d3d {
    ProjectionError_3d3d(const double* curr_points,const double* prev_points)
    : curr_points_(curr_points), prev_points_(prev_points) {}
    
    template <typename T>
    bool operator()(const T* const rotate_R,
                    const T* const transpose_T,
                    T* residuals) const {
                        T rotated_points[3];
                        T prev_poinst_replace[3];
                        prev_poinst_replace[0] = T(prev_points_[0]);
                        prev_poinst_replace[1] = T(prev_points_[1]);
                        prev_poinst_replace[2] = T(prev_points_[2]);
                        ceres::QuaternionRotatePoint(rotate_R,prev_poinst_replace,rotated_points);
                        rotated_points[0] += T(transpose_T[0]);
                        rotated_points[1] += T(transpose_T[1]);
                        rotated_points[2] += T(transpose_T[2]);
                        
                        T points_diff[3];
                        
                        points_diff[0] = T(curr_points_[0]) - rotated_points[0];
                        points_diff[1] = T(curr_points_[1]) - rotated_points[1];
                        points_diff[2] = T(curr_points_[2]) - rotated_points[2];
                        
                        residuals[0] = points_diff[0]*points_diff[0]+points_diff[1]*points_diff[1]+points_diff[2]*points_diff[2];
                        return true;
                    }
                    // Factory to hide the construction of the CostFunction object from
                    // the client code.
                    static ceres::CostFunction* Create(const double* curr_points,
                                                       const double* prev_points) {
                        return (new ceres::AutoDiffCostFunction<ProjectionError_3d3d, 1, 4, 3>(
                            new ProjectionError_3d3d(curr_points, prev_points)));
                                                       }
private:                                                     
    const double* curr_points_;
    const double* prev_points_;
};




class stereo_camera{
public:
    cv::Mat fundamental_matrix;
    Eigen::Matrix<double,3,3> instrinct_matrix;
    double focal_length;
    double baseline;
    double pixel_size;
};

bool ImgstringToLRandNUM(cv::String img_name, int *LoR, int *num);

bool Folder2LRimg(std::string folder, std::vector<std::string> *img_left, std::vector<std::string> *img_right, int NUM);
bool PixelPair23dPoint(stereo_camera camera, cv::Point2f point_left, cv::Point2f point_right,Eigen::Vector3d *points_3d);


bool PointXYZ2Vector3d(pcl::PointXYZ *pointxyz, Eigen::Vector4d point4d);

bool TwoFramesImagesToCloudPoints( cv::Mat img_previous_left, cv::Mat img_previous_right, cv::Mat img_current_left, cv::Mat img_current_right,stereo_camera camera,   std::vector<Eigen::Vector4d> *points_prev, Eigen::Matrix<double,4,4> *Curr2Prev);
/*
*/
bool InitialCameraAndPointCloud(cv::Mat img_left, cv::Mat img_right, stereo_camera *camera, std::vector<Eigen::Vector4d> *points_cloud);


#endif
