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
#include "my_vo_function.h"
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // laod image
    //Initial camera;
    cout<<"Project visual odometry \n"<<"Initialization"<<endl;
    cv::Mat test_left = cv::imread("/home/thomas/Desktop/SLAM/vo_test/test_left.png",0);
    cv::Mat test_right = cv::imread("/home/thomas/Desktop/SLAM/vo_test/test_right.png",0);
    stereo_camera camera;
    double focal_length = 487.109*pow(10,-6);
    double baseline = 0.120006;
    double principle_x = 320.788;
    double principle_y = 245.845;
    Eigen::Matrix3d K;
    K << 487.109, 0.0, 320.788, 0.0, 487.109, 245.845, 0.0, 0.0, 1.0;
    camera.focal_length = focal_length;
    camera.baseline = baseline;
    camera.instrinct_matrix = K;
    camera.pixel_size = pow(10,-5);
    std::vector<Eigen::Vector4d> points_cloud_temp;
    InitialCameraAndPointCloud(test_left, test_right, &camera, &points_cloud_temp);
    // Init state
    std::vector<Eigen::Matrix<double,6,1>> state; // 1-3 rotation angles, 4-6 coordinate
    Eigen::Matrix<double,4,4> Homo2world;
    Eigen::Matrix<double,6,1> state_init;
    std::vector<Eigen::Vector4d> points_cloud; // points cloud
    for (int i=0; i<=3; i++)
    {
        for(int j = 0 ; j<=3; j++)
        {
            Homo2world(i,j) = 0;
            Homo2world(i,i) = 1;
        }
    }
    
    for(int i =0; i<6; i++)
    {
        state_init(i) = 0;
    }
    state.push_back(state_init);
    
    //init point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<DMatch> matches_previous, matches_current, matches_pose;
    cv::Mat img_current_left,img_current_right,img_previous_left,img_previous_right;
    std::vector<Eigen::Vector4d> points_prev;
    Eigen::Matrix<double,4,4> Curr2Prev;
    pcl::PointXYZ pointxyz;
    std::vector<std::string> img_name_left;
    std::vector<std::string> img_name_right;
    Folder2LRimg("/home/thomas/Desktop/SLAM/wean_hall/wean_rectified_images/wean/images/rectified/*.png",&img_name_left,&img_name_right,5);
    for(int i =0; i<img_name_left.size()-1;i++)
    {
    img_previous_left = cv::imread(img_name_left[i],IMREAD_GRAYSCALE );
    img_previous_right = cv::imread(img_name_right[i],IMREAD_GRAYSCALE );
    img_current_left = cv::imread(img_name_left[i+1], IMREAD_GRAYSCALE);
    img_current_right = cv::imread(img_name_right[i+1], IMREAD_GRAYSCALE);
    points_prev.clear();
    
    if(TwoFramesImagesToCloudPoints( img_previous_left, img_previous_right, img_current_left,  img_current_right, camera, &points_prev, &Curr2Prev)==true)
    {
        cout<<"Curr2Prev"<<Curr2Prev<<endl;
        cout<<points_prev[0]<<endl;
        
         // update homography
        for(int j=0; j < points_prev.size(); j++)
        {
            if(PointXYZ2Vector3d(&pointxyz, Homo2world*points_prev[j])==true)
            {
                
               
                cloud_ptr->push_back(pointxyz);
            }
        }
        Homo2world = Homo2world*Curr2Prev;
          cout<<cloud_ptr->size()<<endl;
    }
  
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
