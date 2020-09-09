
#include <my_vo_function.h>

using namespace std;
using namespace cv;




/*function PixelPair23dPoint transfer the matched pairs of left and right pixel points to 3d points
 * 
 * 
 * 
 * 
 * 
 * 
 * */

bool PixelPair23dPoint(stereo_camera camera, cv::Point2d point_left, cv::Point2d point_right,Eigen::Vector3d *points_3d)
{
    Eigen::Matrix3d K = camera.instrinct_matrix;
    double focal_length = camera.focal_length;
    double baseline = camera.baseline;
    double pixel_size = camera.pixel_size; 
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


bool PointXYZ2Vector3d(pcl::PointXYZ *pointxyz, Eigen::Vector4d point4d)
{
    pcl::PointXYZ pointxyz_temp;
    
    pointxyz_temp.x = point4d(0); 
    pointxyz_temp.y = point4d(1); 
    pointxyz_temp.z = point4d(2);
    *pointxyz = pointxyz_temp;
    return true;
}



/*function TwoFramesImagesToCloudPoints estimates the pose between two frames with stereo images
 * Input: img_previous_left, img_previous_right are the left and right image of previous frame respectivly;
 *        img_current_left, img_current_right are the left and right image of current frame respectivly;
 *        fundamental_matrix is the fundamental matrix of the stereo cameras
 *      
 * Output: points_prev contains the matched points on the previous frame coordinate
 *         Curr2Prev is the homograph matrix descripts the projection from current frame to previous frame
 * 
 * 
 * */
bool TwoFramesImagesToCloudPoints( cv::Mat img_previous_left, cv::Mat img_previous_right, cv::Mat img_current_left, cv::Mat img_current_right,stereo_camera camera, std::vector<Eigen::Vector4d> *points_prev, Eigen::Matrix<double,4,4> *Curr2Prev)
{
    //Feature points detection and matching
    
    
    cv::Mat descriptors_previous_left, descriptors_previous_right, descriptors_current_left, descriptors_current_right;
    std::vector<KeyPoint> keypoints_current_left, keypoints_current_right, keypoints_previous_left, keypoints_previous_right;
    cv::Ptr<ORB> orb = cv::ORB::create(1000,1.2f,8,31,0,2,ORB::FAST_SCORE,31,20); //ORB detector
    std::vector<DMatch> matches_previous, matches_current, matches_pose;
    
    orb->detect(img_previous_left,keypoints_previous_left,noArray());
    orb->detect(img_previous_right,keypoints_previous_right,noArray());
    orb->detect(img_current_left,keypoints_current_left,noArray());
    orb->detect(img_current_right,keypoints_current_right,noArray());
    orb->compute(img_previous_left,keypoints_previous_left,descriptors_previous_left);
    orb->compute(img_previous_right,keypoints_previous_right,descriptors_previous_right);
    orb->compute(img_current_left,keypoints_current_left,descriptors_current_left);
    orb->compute(img_current_right,keypoints_current_right,descriptors_current_right);
    BFMatcher matcher;
    matcher.create(NORM_HAMMING,false);
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
    Mat fundamental_matrix = camera.fundamental_matrix;
    int temp = 0;
    // find the match pairs between previous and current frames.
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
                        }
                    }
                }
            }
    }
    cv::correctMatches(fundamental_matrix,point_left_curr,point_right_curr,filtered_point_left_curr,filtered_point_right_curr);
    cv::correctMatches(fundamental_matrix,point_left_prev,point_right_prev,filtered_point_left_prev,filtered_point_right_prev);
    for(int i=0; i<filtered_point_left_curr.size(); i++)
    {
        if(PixelPair23dPoint(camera, filtered_point_left_curr[i], filtered_point_right_curr[i], &point_3d_curr) && PixelPair23dPoint(camera, filtered_point_left_prev[i], filtered_point_right_prev[i], &point_3d_prev))
        {
            points_3d_curr.push_back(point_3d_curr);
            points_3d_prev.push_back(point_3d_prev);
        }
    }
    if( points_3d_prev.size()<=10)
    {
        cout<<"Too few matched points, terminal the pose estimation"<<endl;
        return false;
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
    std::vector<Eigen::Vector3d> points_previous_frame;
    std::vector<Eigen::Vector4d> points_prev_homo;
    Eigen::Vector3d prev_point_3d;
    Eigen::Vector4d prev_point_4d;
    for(int i=0; i < points_3d_curr.size(); i++)
    {
        prev_point_3d = pose_rotation.inverse()*(points_3d_curr[i]-pose_tranpose);
        prev_point_4d.block(0,0,3,1) = prev_point_3d;
        prev_point_4d(3) = 1;
        points_prev_homo.push_back(prev_point_4d);
    }    
    Eigen::Matrix<double,4,4> HomoCurr2Prev;
    for (int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            HomoCurr2Prev(i,j) = 0;
        }
    }
    HomoCurr2Prev.block(0,0,3,3) = pose_rotation.inverse();
    HomoCurr2Prev.block(0,3,3,1) = pose_tranpose*(-1.0);
    *Curr2Prev = HomoCurr2Prev;
    *points_prev =  points_prev_homo;
    return true;
}

bool InitialCameraAndPointCloud(cv::Mat img_left, cv::Mat img_right, stereo_camera *camera, std::vector<Eigen::Vector4d> *points_cloud)
{
    std::vector<KeyPoint> keypoints_left, keypoints_right;
    cv::Mat descriptors_left, descriptors_right;
    std::vector<DMatch> matches;
    cv::Ptr<ORB> orb = cv::ORB::create(1000,1.2f,8,31,0,2,ORB::FAST_SCORE,31,20);
    orb->detect(img_left,keypoints_left,noArray());
    orb->detect(img_right,keypoints_right,noArray());
    orb->compute(img_left,keypoints_left,descriptors_left);
    orb->compute(img_right,keypoints_right,descriptors_right);
    BFMatcher matcher;
    matcher.create(NORM_HAMMING,false);
    matcher.match(descriptors_left,descriptors_right,matches);
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
    Eigen::Vector3d point_3d;
    Eigen::Vector4d point_homo;
    camera->fundamental_matrix = fundamental_matrix;
    
    for(int i =0; i< filtered_points_left.size(); i++ )
    {       
        PixelPair23dPoint( *camera, filtered_points_left[i], filtered_points_right[i],  &point_3d);
        point_homo.block(0,0,3,1) = point_3d;
        point_homo(3) = 1.0;
        points_cloud->push_back(point_homo); 
    }
    
    
    return true;
}


