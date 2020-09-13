
#include <my_vo_function.h>

using namespace std;
using namespace cv;



//bool RANSAC_3D3D(cv::)
bool RANSAC_3D3D(vector<Eigen::Vector3d> *points_3d_prev, vector<Eigen::Vector3d> *points_3d_curr, Eigen::Matrix<double,3,3> *pose_rotation, Eigen::Vector3d *pose_transpose,double *projection_error_return, int *inliner_number_return)
{
    vector<Eigen::Vector3d> prev_points = *points_3d_prev;
    vector<Eigen::Vector3d> curr_points = *points_3d_curr;
    vector<Eigen::Vector3d> prev_points_inliner;
    vector<Eigen::Vector3d> curr_points_inliner;
    vector<Eigen::Vector3d> prev_points_inliner_final;
    vector<Eigen::Vector3d> curr_points_inliner_final;
    vector<Eigen::Vector3d> prev_point_temp, curr_point_temp;
    vector<Eigen::Matrix<double,3,3>> R;
    vector<Eigen::Vector3d> T;
    Eigen::Matrix<double,3,3> rotation ;
    Eigen::Vector3d transpose;
    Eigen::Matrix<double,3,3> R_temp;
    Eigen::Vector3d T_temp;
    Eigen::Vector3d project_error_vector;
    int num = prev_points.size();         
    int K = 5;//iteration num
    int inliner[K];
    double threshold = 0.1;
    int minimum_points_num = 10;
    int N[minimum_points_num];
    int projection_error[K];
    double inliner_number_min = 0;
    for(int k =0; k<K; k++)
    {   
        //Randomly select 7 points
        prev_point_temp.clear();
        curr_point_temp.clear();
        for(int nn = 0; nn<minimum_points_num;nn++)
        {
            N[nn] = int(rand() % num);  
            prev_point_temp.push_back(prev_points[N[nn]]);
            curr_point_temp.push_back(curr_points[N[nn]]);
        }
        //Model fitting with 7 points

        if(!ceres_modelfitting(prev_point_temp, curr_point_temp, &R_temp, &T_temp ))
        {
            return false;
        }
        //outliner detection
        
        inliner[k] = 0;
        prev_points_inliner.clear();
        curr_points_inliner.clear();
        for(int i=0; i<num; i++)
        {
            project_error_vector = R_temp*prev_points[i] + T_temp-curr_points[i];
            if(project_error_vector.norm()<threshold)
            {
                inliner[k]++;
                prev_points_inliner.push_back(prev_points[i]);
                curr_points_inliner.push_back(curr_points[i]);
            }
        }
        //inliner refitting
        projection_error[k] = 0;
        prev_points_inliner_final.clear();
        curr_points_inliner_final.clear();
        if(prev_points_inliner.size()>=minimum_points_num)
            if(!ceres_modelfitting(prev_points_inliner,curr_points_inliner,&R_temp,&T_temp))
            {
                return false;
            }
            R.push_back(R_temp);
            T.push_back(T_temp);
            for(int i=0; i<prev_points_inliner.size(); i++)
            {
                project_error_vector = R_temp*prev_points_inliner[i] + T_temp-curr_points_inliner[i];
                if(project_error_vector.norm()<threshold)
                {
                    projection_error[k] += project_error_vector.norm();
                    prev_points_inliner_final.push_back(prev_points_inliner[i]);
                    curr_points_inliner_final.push_back(curr_points_inliner[i]);
                }
            }
            if(inliner[k]>inliner_number_min)
            {
                inliner_number_min = inliner[k];
                *points_3d_prev = prev_points_inliner_final;
                *points_3d_curr = curr_points_inliner_final;
                *pose_rotation = R_temp;
                *pose_transpose = T_temp;
                *projection_error_return = projection_error[k];
                *inliner_number_return = prev_points_inliner_final.size();
            }
    }
    return true;
}
bool ceres_modelfitting(vector<Eigen::Vector3d> points_3d_prev, vector<Eigen::Vector3d> points_3d_curr, Eigen::Matrix<double,3,3> *pose_rotation_return, Eigen::Vector3d *pose_transpose_return)
{
    ceres::Problem problem;
    double prev_point[3];
    double curr_point[3];
    
    double transpose_T[3];
    double rotate_R[4];
    rotate_R[0] = 1.0;
    rotate_R[1] = 0.000000;
    rotate_R[2] = 0.000000;
    rotate_R[3] = 0.000000;;
    transpose_T[0] = 0.00000;
    transpose_T[1] = 0.00000;
    transpose_T[2] = 0.00000;
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
    options.max_num_iterations = 10000;
    // options.gradient_tolerance = 1e-20;
    //options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout<<summary.BriefReport()<<endl;
    if(!summary.IsSolutionUsable())
    {
    cout<<"UNCONVERGENCE!"<<endl;
        return false;
    }
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
    *pose_transpose_return = pose_tranpose;
    *pose_rotation_return = pose_rotation;
    return true;
}

/*function PixelPair23dPoint transfer the matched pairs of left and right pixel points to 3d points
 * 
 * 
 * 
 * 
 * 
 * 
 * */
bool ImgstringToLRandNUM(cv::String img_name, int *LoR, int *num)
{
    std::vector<int> dot_place;
    std::string img_num;
    for(int j=0; j< img_name.length(); j++)
    {
        
        if(img_name[j]=='.')
        {
            dot_place.push_back(j);
        }
    }
    if(dot_place.empty())
    {
        cout<<"WARNING:"<<img_name<<" has 0 dot"<<endl;
        return false;
    }
    if(img_name[dot_place[0]+1] == 'l')
    {*LoR = 1;}
    if(img_name[dot_place[0]+1] == 'r')
    {*LoR = 0;}
    for(int j =0; j<(dot_place[2]-dot_place[1]-1); j++)
    {
        img_num[j] = img_name[dot_place[1]+1+j];
    }
    *num  = std::stoi (img_num,nullptr);
    
    return true;
}

bool Folder2LRimg(std::string folder, std::vector<std::string> *img_left, std::vector<std::string> *img_right, int NUM)
{
    vector<cv::String> img_string;
    vector<int> left_or_right; //left = 0; right = 1;
    vector<int> img_number_left,img_number_right;
    std::vector<cv::String> img_left_temp(NUM);
    std::vector<cv::String> img_right_temp(NUM);
    cv::glob(folder, img_string, false);
    cv::String img_name;
    vector<int> dot_palce;
    int LoR, img_num;
    
    for (int i=0; i<img_string.size(); i++)
    {
        
        {
            img_name = img_string[i];
            ImgstringToLRandNUM(img_name, &LoR, &img_num);
            img_num = img_num - 600;
            if(LoR == 1)//left
            {
                if(img_num<NUM)
                {
                    img_left_temp[img_num] = img_name;
                }
            }
            else
            {
                if(img_num<NUM)
                {
                    img_right_temp[img_num] = (img_name);
                }
            }
            
        }
    }
    for(int i =0; i< NUM; i++)
    {
        if( img_left_temp[i].empty() || img_right_temp[i].empty())
        {
            img_left_temp.erase(img_left_temp.begin()+i);
            img_right_temp.erase(img_left_temp.begin()+i);
            i=i-1;
        }
        else
        {
            img_left->push_back( img_left_temp[i]);
            img_right->push_back(img_right_temp[i]);         
        }
    }
    
}


bool PixelPair23dPoint(stereo_camera camera, cv::Point2d point_left, cv::Point2d point_right,Eigen::Vector3d *points_3d)
{
    Eigen::Matrix3d K = camera.instrinct_matrix;
    double focal_length = camera.focal_length;
    double baseline = camera.baseline;
    double pixel_size = camera.pixel_size; 
    double Cx = camera.instrinct_matrix(0,2);
    double Cy = camera.instrinct_matrix(1,2);
    Eigen::Vector3d point_homo;
    //  double focal_length = camera_instrinct
    // double baseline = camera_instrinct(1,1)
    double diff = double(point_left.x - point_right.x);
    double depth = focal_length*baseline/(diff*camera.pixel_size);
    Eigen::Vector3d point_3d;
    
    point_homo(0) = point_left.x;
    point_homo(1) = point_left.y;
    point_homo(2) = 1.0;
    point_3d(0) = (point_homo(0)*depth-Cx*depth)*pixel_size/focal_length;
    point_3d(1) = (point_homo(1)*depth-Cy*depth)*pixel_size/focal_length;
    point_3d(2) = depth;
    *points_3d = point_3d;
    //points_3d = &point_3d;
    return true;
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
    
    
    Eigen::Matrix<double,3,3> pose_rotation;
    Eigen::Vector3d pose_transpose;
    double projection_error_ransac;
    int inliner_number;
    //RANSAC outliner filtering
    if(!RANSAC_3D3D(&points_3d_prev, &points_3d_curr, &pose_rotation, &pose_transpose,&projection_error_ransac, &inliner_number))
    {
        return false;
    }
    
    if(inliner_number<20)
    {
        return false;
    }
    
    std::vector<Eigen::Vector4d> points_prev_homo;
    Eigen::Vector3d prev_point_3d;
    Eigen::Vector4d prev_point_4d;
    Eigen::Vector3d Project_error;
    int project_error_sum = 0;
    
    for(int i=0; i < points_3d_prev.size(); i++)
    {
        // prev_point_3d = pose_rotation.inverse()*(points_3d_curr[i]-pose_tranpose);
        prev_point_3d = points_3d_prev[i];
        Project_error = (pose_rotation*points_3d_prev[i]+pose_transpose-points_3d_curr[i]);
        project_error_sum += Project_error.norm();
       // cout<<"projection_error +T:"<<Project_error.norm()<<endl;
        prev_point_4d(0) = prev_point_3d(0);
        prev_point_4d(1) = prev_point_3d(1);
        prev_point_4d(2) = prev_point_3d(2);
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
    HomoCurr2Prev(3,3) = 1;
    HomoCurr2Prev.block(0,0,3,3) = pose_rotation;
    HomoCurr2Prev.block(0,3,3,1) = pose_transpose;
    *Curr2Prev = HomoCurr2Prev.inverse();
    for(int i =0; i< points_prev_homo.size(); i++)
    {
        points_prev->push_back(  points_prev_homo[i]);
    }
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


