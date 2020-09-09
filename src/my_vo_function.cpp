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

#include <my_vo_function.h>

using namespace std;
using namespace cv;






bool PixelPair23dPoint(Eigen::Matrix3d K, double focal_length, double baseline,cv:;Point2f point_left, cv::Point2f point_right,double pixel_size, Eigen::Vector3d *points_3d)
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
            point_3d = depth*K.inverse()*point_homo;
            points_3d = &point_3d;
        }
}


/*function TwoFramesImagesToCloudPoints estimates the pose between two frames with stereo images
 * Input: img_previous_left, img_previous_right are the left and right image of previous frame respectivly;
 *        img_current_left, img_current_right are the left and right image of current frame respectivly;
 *        fundamental_matrix is the fundamental matrix of the stereo cameras
 *        Prev2World is the homography matrix transfers the previous frame coordinate to the world coordinate.
 * Output: points_world contains the matched points on the world coordinate
 * 
 * 
 * */
bool TwoFramesImagesToCloudPoints( cv::Mat img_previous_left, cv::Mat img_previous_right, cv::Mat img_current_left, cv::Mat img_current_right,stereo_camera camera, Eigen::Matrix<double,4,4> Prev2World,  std::vector<Eigen::Vector3d> *points_world)
{
    //Feature points detection and matching
    cv::Mat descriptors_previous_left, descriptors_previous_right, descriptors_current_left, descriptors_current_right;
    std::vector<KeyPoint> keypoints_current_left, keypoints_current_right, keypoints_previous_left, keypoints_previous_right;
    std::vector<DMatch> matches_previous, matches_current, matches_pose;
    orb->detect(img_previous_left,keypoints_previous_left,noArray());
    orb->detect(img_previous_right,keypoints_previous_right,noArray());
    orb->detect(img_current_left,keypoints_current_left,noArray());
    orb->detect(img_current_right,keypoints_current_right,noArray());
    orb->compute(img_previous_left,keypoints_previous_left,descriptors_previous_left);
    orb->compute(img_previous_right,keypoints_previous_right,descriptors_previous_right);
    orb->compute(img_current_left,keypoints_current_left,descriptors_current_left);
    orb->compute(img_current_right,keypoints_current_right,descriptors_current_right);
    matcher.match(descriptors_previous_left,descriptors_previous_right,matches_previous);
    matcher.match(descriptors_current_left,descriptors_current_right,matches_current);
    matcher.match(descriptors_previous_left,descriptors_current_left,matches_pose);
    
    vector<Point2d> points_previous_left, points_previous_right, points_current_left, points_current_right;
    vector<Point2d> points_filtered_previous_left, points_filtered_previous_right, points_filtered_current_left, points_filtered_current_right;
    std::vector<KeyPoint> filtered_points_pose_pre, filtered_points_pose_curr;
    std::vector<cv::Point2d> point_left_curr, point_right_curr, point_left_prev, point_right_prev;
    std::vector<cv::Point2d> filtered_point_left_curr, filtered_point_right_curr, filtered_point_left_prev, filtered_point_right_prev;
    Eigen::Vector3d point_3d_curr, point_3d_prev;
    std::vector<Eigen::Vector3d> points_3d_prev, points_3d_curr;
    double pixel_size = camera.pixel_size;
    Eigen::Matrix3d K = camera.instrinct_matrix;
    double focal_length = camera.focal_length;
    double baseline = camera.baseline;
    for( int i = 0; i<(int) matches_pose.size(); i++)
    {
        for(int j = 0; j<(int)matches_previous.size();j++)
            if(matches_previous[j].queryIdx==matches_pose[i].queryIdx)
            {
                for(int k = 0; k < (int)matches_current.size(); k++)
                {
                    if(matches_current[k].queryIdx==matches_pose[i].trainIdx)
                    {
                        if(matches_previous[j].distance < 200 && matches_current[k].distance<200)
                        {
                            point_left_curr.push_back( keypoints_current_left[matches_current[k].queryIdx].pt);
                            point_right_curr.push_back( keypoints_current_right[matches_current[k].trainIdx].pt);
                            point_left_prev.push_back( keypoints_previous_left[matches_previous[j].queryIdx].pt);
                            point_right_prev.push_back(keypoints_previous_right[matches_previous[j].trainIdx].pt);
                            cv::correctMatches(fundamental_matrix,point_left_curr,point_right_curr,filtered_point_left_curr,filtered_point_right_curr);
                            cv::correctMatches(fundamental_matrix,point_left_prev,point_right_prev,filtered_point_left_prev,filtered_point_right_prev);
                            
                            if(  PixelPair23dPoint( K, focal_length, baseline, filtered_point_left_curr[0], filtered_point_right_curr[0], pixel_size, &point_3d_curr) &&  PixelPair23dPoint( K, focal_length, baseline, filtered_point_left_prev[0], filtered_point_right_prev[0], pixel_size, &point_3d_prev))
                            {
                                points_3d_curr.push_back(point_3d_curr);
                                points_3d_prev.push_back(point_3d_prev);
                            }
                        }
                    }
                }
            }
    }
    
    
    //3D-3D pose estimation
    ceres::Problem problem;
    double prev_point[3];
    double curr_point[3];
    double rotate_R[4];
    double transpose_T[3];
    rotate_R[0] = 0.001;
    rotate_R[1] = 0.001;
    rotate_R[2] = 0.001;
    rotate_R[3] = 0.001;
    transpose_T[0] = 0.001;
    transpose_T[1] = 0.001;
    transpose_T[2] = 0.001;
    for(int i=0; i < points_3d_prev.size(); i++)
    {
        prev_point[0] = points_3d_prev[i](0);
        prev_point[1] = points_3d_prev[i](1);
        prev_point[2] = points_3d_prev[i](2);
        curr_point[0] = points_3d_curr[i](0);
        curr_point[1] = points_3d_curr[i](1);
        curr_point[2] = points_3d_curr[i](2);
        ceres::CostFunction* cost_function = ProjectionError_3d3d::Create(curr_point,prev_point);
        problem.AddResidualBlock(cost_function, NULL, rotate_R, transpose_T );
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1000;
    //    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //  cout<<summary.FullReport()<<endl;
    double rotation_matrix[9];
    ceres::QuaternionToRotation(rotate_R, rotation_matrix);
    Eigen::Matrix<double,3,3> pose_rotation;
    Eigen::Vector3d pose_tranpose;
    for(int i=0;i<=2;i++)
    {
        pose_tranpose(i) = transpose_T[i];
        for(int j=0;j<=2;j++)
        {
            pose_rotation(i,j) = rotation_matrix[3*i+j];
        }
    }
    cout<<"The rotation matrix"<<pose_rotation<<endl;
    cout<<"The transpose"<<pose_tranpose<<endl;
    std::vector<Eigen::Vector3d> curr_points_world;
    
    for(int i=0; i < points_3d_curr.size(); i++)
    {
        curr_points_world.push_back(pose_rotation.inverse()*(points_3d_curr[i]-pose_tranpose));
    }
    return true;
}
