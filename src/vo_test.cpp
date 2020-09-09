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
#include <string.h>
#include <ProjectionError_3d3d.h>
#include <my_vo_function.h>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // laod image
    cv::Mat test_left = cv::imread("/home/thomas/Desktop/SLAM/vo_test/test_left.png",0);
    cv::Mat test_right = cv::imread("/home/thomas/Desktop/SLAM/vo_test/test_right.png",0);
    
    // orb detection
    std::vector<KeyPoint> keypoints_left, keypoints_right;
    cv::Mat descriptors_left, descriptors_right;
    Mat print_image;
    std::vector<DMatch> matches;
    
    cv::Ptr<ORB> orb = cv::ORB::create(1000,1.2f,8,31,0,2,ORB::FAST_SCORE,31,20);
    
    orb->detect(test_left,keypoints_left,noArray());
    orb->detect(test_right,keypoints_right,noArray());
    orb->compute(test_left,keypoints_left,descriptors_left);
    orb->compute(test_right,keypoints_right,descriptors_right);
    
    drawKeypoints(test_left,keypoints_left,print_image,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    
    BFMatcher matcher;
    matcher.create(NORM_HAMMING,false);
    matcher.match(descriptors_left,descriptors_right,matches);
    drawMatches(test_left,keypoints_left,test_right,keypoints_right,matches,print_image);
    imshow( "Original matching", print_image);
    // imshow("Right", test_right);
    
    //Matching filter
    std::vector<DMatch> filtered_matches;
    
    double min_dist = 99999;
    double max_dist = 0;
    for(int i=0; i<descriptors_left.rows; i++)
    {
        if(matches[i].distance > max_dist) max_dist = matches[i].distance;
        if(matches[i].distance < min_dist) min_dist = matches[i].distance;
    }
    for(int i=0; i<descriptors_left.rows; i++)
    {
        if(matches[i].distance < std::max(10*min_dist,50.0))
        {
            filtered_matches.push_back(matches[i]);
        }
        
    }
    drawMatches(test_left,keypoints_left,test_right,keypoints_right,filtered_matches,print_image);
    imshow("filtered_matches",print_image);
    
    // calculating fundamental matrix
    vector<Point2d> points_left, points_right;
    vector<Point2d> filtered_points_left, filtered_points_right;
    vector<Point2d> depth_filtered_points_left, depth_filtered_points_right;
    Mat fundamental_matrix;
    for(int j=0; j<(int) filtered_matches.size(); j++)
    {
        
        points_left.push_back(keypoints_left[filtered_matches[j].queryIdx].pt);
        points_right.push_back(keypoints_right[filtered_matches[j].trainIdx].pt);
    }
    fundamental_matrix = cv::findFundamentalMat (points_left,points_right,FM_LMEDS);//2 for 8 points
    
    points_left.clear();
    points_right.clear();
    for(int j=0; j<(int) matches.size(); j++)
    {   
        if(matches[j].distance < 200)
        {
            points_left.push_back(keypoints_left[matches[j].queryIdx].pt);
            points_right.push_back(keypoints_right[matches[j].trainIdx].pt);
        }
    }
    correctMatches(fundamental_matrix,points_left,points_right,filtered_points_left,filtered_points_right);
    
    //2D to 3D
    float focal_length = 487.109*pow(10,-6);
    float baseline = 0.120006;
    float principle_x = 320.788;
    float principle_y = 245.845;
    Eigen::Matrix3d K;
    K << 487.109, 0.0, 320.788, 0.0, 487.109, 245.845, 0.0, 0.0, 1.0;
    Eigen::Vector3d points_2d_homo,points_3d;
    pcl::PointXYZ points_3d_pc;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    cout<<filtered_points_left.size()<<endl;
    for(int i=0; i<=filtered_points_left.size(); i++)
    {
        float diff = (filtered_points_left[i].x - filtered_points_right[i].x)*pow(10,-5);
        float depth = focal_length*baseline/diff;
        if (depth > 0.1)
        {
            //depth_filtered_points_left.push_back(filtered_points_left[i]);
            //depth_filtered_points_right.push_back(filtered_points_right[i]);
            points_2d_homo(0) = filtered_points_left[i].x;
            points_2d_homo(1) = filtered_points_left[i].y;
            points_2d_homo(2) = 1;
            points_3d = depth*K.inverse()*points_2d_homo;
            points_3d_pc.x = points_3d(0);
            points_3d_pc.y = points_3d(1);
            points_3d_pc.z = points_3d(2);
            
            if(points_3d.transpose()*points_3d<1)
            {
                cloud_ptr->points.push_back(points_3d_pc);
            }
        }
        
    }
    
    cout<<"cloud size: "<<cloud_ptr->size()<<endl;
    
    /*
     *    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("point cloud"));
     *    viewer->setBackgroundColor(0,0,0);
     *    viewer->addPointCloud<pcl::PointXYZ>( cloud_ptr,"current points");
     *    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_ptr, 0, 255, 0);
     *    viewer->setPointCloudRenderingProperties
     *    (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "current points");
     *    viewer->addCoordinateSystem(1.0);
     *    viewer->initCameraParameters();
     *    while (!viewer->wasStopped ())
     *    {
     *        viewer->spinOnce (100);
     *        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
}*/
    
    // pose stimation between 3d-3d points
    cv::Mat img_previous_left, img_previous_right, img_current_left, img_current_right;
    cv::Mat descriptors_previous_left, descriptors_previous_right, descriptors_current_left, descriptors_current_right;
    std::vector<KeyPoint> keypoints_current_left, keypoints_current_right, keypoints_previous_left, keypoints_previous_right;
    std::vector<DMatch> matches_previous, matches_current, matches_pose;
    //wean_wide_interesting.left-rectified.00000600.t_001268594688.921905.png
    string img_address = "wean_wide_interesting.left-rectified.";
    img_previous_left = cv::imread("../img/wean_wide_interesting.left-rectified.00000600.t_001268594688.921905.png",IMREAD_GRAYSCALE );
    img_previous_right = cv::imread("../img/wean_wide_interesting.right-rectified.00000600.t_001268594688.921905.png",IMREAD_GRAYSCALE );
    img_current_left = cv::imread("../img/wean_wide_interesting.left-rectified.00000601.t_001268594688.992430.png", IMREAD_GRAYSCALE);
    img_current_right = cv::imread("../img/wean_wide_interesting.right-rectified.00000601.t_001268594688.992430.png", IMREAD_GRAYSCALE);
    
    
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
    cout<<"previous frame matching pair is:"<<matches_previous.size()<<endl;
    vector<Point2d> points_previous_left, points_previous_right, points_current_left, points_current_right;
    vector<Point2d> points_filtered_previous_left, points_filtered_previous_right, points_filtered_current_left, points_filtered_current_right;
    
    //Only both frame matched left
    std::vector<KeyPoint> filtered_points_pose_pre, filtered_points_pose_curr;
    std::vector<cv::Point2d> point_left_curr, point_right_curr, point_left_prev, point_right_prev;
    std::vector<cv::Point2d> filtered_point_left_curr, filtered_point_right_curr, filtered_point_left_prev, filtered_point_right_prev;
    Eigen::Vector3d point_3d_curr, point_3d_prev;
    std::vector<Eigen::Vector3d> points_3d_prev, points_3d_curr;
    double pixel_size = pow(10,-5);
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
    
    cout<<"3D points matched pair size: "<<points_3d_curr.size()<<endl;
    
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
    cout<<"Rotate quaterion:"<<rotate_R[0]<<","<<rotate_R[1]<<","<<rotate_R[2]<<","<<rotate_R[3]<<"."<<endl;
    cout<<"Transpose:"<<transpose_T[0]<<","<<transpose_T[1]<<","<<transpose_T[2]<<"."<<endl;
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
    
    for(int i = 0; i< curr_points_world.size(); i++)
    {
    points_3d_pc.x = curr_points_world[i](0);
    points_3d_pc.y = curr_points_world[i](1);
    points_3d_pc.z = curr_points_world[i](2);
    
    
        cloud_ptr->points.push_back(points_3d_pc);

}




pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("point cloud"));
viewer->setBackgroundColor(0,0,0);
viewer->addPointCloud<pcl::PointXYZ>( cloud_ptr,"current points");
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_ptr, 0, 255, 0);
viewer->setPointCloudRenderingProperties
(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "current points");
viewer->addCoordinateSystem(1.0);
viewer->initCameraParameters();
while (!viewer->wasStopped ())
{
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
}





waitKey(0);
return 0;
}


