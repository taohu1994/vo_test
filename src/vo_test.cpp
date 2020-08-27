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
    vector<Point2f> points_left, points_right;
    vector<Point2f> filtered_points_left, filtered_points_right;
    vector<Point2f> depth_filtered_points_left, depth_filtered_points_right;
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
        points_left.push_back(keypoints_left[matches[j].queryIdx].pt);
        points_right.push_back(keypoints_right[matches[j].trainIdx].pt);
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
    for(int i; i<=filtered_points_left.size(); i++)
    {
        float diff = (filtered_points_left[i].x - filtered_points_right[i].x)*pow(10,-5);
        float depth = focal_length*baseline/diff;
        if (depth > 0.1)
        {
            depth_filtered_points_left.push_back(filtered_points_left[i]);
            depth_filtered_points_right.push_back(filtered_points_right[i]);
            points_2d_homo(0) = filtered_points_left[i].x;
            points_2d_homo(1) = filtered_points_left[i].y;
            points_2d_homo(2) = 1;
            points_3d = depth*K.inverse()*points_2d_homo;
            points_3d_pc.x = points_3d(0);
            points_3d_pc.y = points_3d(1);
            points_3d_pc.x = points_3d(2);
            cloud_ptr->points.push_back(points_3d_pc);
        }
        
    }
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->addPointCloud<pcl::PointXYZ>( cloud_ptr,"point cloud");
/*while (!viewer.wasStopped ())
    {}*/
    
    
    waitKey(0);
    return 0;
}
