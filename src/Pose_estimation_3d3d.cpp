#include <cmath>
#include <cstdio>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>


class ProjectionError_3d3d {
private:
    const Eigen::Vector3d pre_points_;
    const Eigen::Vector3d curr_points_;
    
public:
    ProjectionError_3d3d(const Eigen::Vector3d& pre_points,
                         const Eigen::Vector3d& curr_points):
                         pre_points_(pre_points), curr_points_(curr_points)
                         {}
                         
                         template  <typename  T>
                         bool operator() (const T* const rotate_R,
                                          const T* const trans_T,
                                          T* residual) const {
                                              T curr_point_T[3] = {T(curr_points_(0)),T(curr_points_(1)),T(curr_points_(2))};
                                              T pre_point_T[3] = {T(pre_points_(0)),T(pre_points_(1)),T(pre_points_(2))};
                                              
                                              
                                              
                                              R[0] = Rotate_R
                                              
                                              residual = pre_points_
                                              return true;
                                              
                                              
                                              T l_pt_L[3] = {T(laser_point_(0)), T(laser_point_(1)), T(laser_point_(2))};
                                              T n_C[3] = {T(normal_to_plane_(0)), T(normal_to_plane_(1)), T(normal_to_plane_(2))};
                                              T l_pt_C[3];
                                              ceres::AngleAxisRotatePoint(R_t, l_pt_L, l_pt_C);
                                              l_pt_C[0] += R_t[3];
                                              l_pt_C[1] += R_t[4];
                                              l_pt_C[2] += R_t[5];
                                              Eigen::Matrix<T, 3, 1> laser_point_C(l_pt_C);
                                              Eigen::Matrix<T, 3, 1> laser_point_L(l_pt_L);
                                              Eigen::Matrix<T, 3, 1> normal_C(n_C);
                                              residual[0] = normal_C.normalized().dot(laser_point_C) - normal_C.norm();
                                              return true;
                                          }
};

struct ProjectionError_3d3d {
    ProjectionError_3d3d(double observed_x, double observed_y)
    : observed_x(observed_x), observed_y(observed_y) {}
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
                        // camera[0,1,2] are the angle-axis rotation.
                        T p[3];
                        ceres::AngleAxisRotatePoint(camera, point, p);
                        // camera[3,4,5] are the translation.
                        p[0] += camera[3];
                        p[1] += camera[4];
                        p[2] += camera[5];
                        // Compute the center of distortion. The sign change comes from
                        // the camera model that Noah Snavely's Bundler assumes, whereby
                        // the camera coordinate system has a negative z axis.
                        T xp = - p[0] / p[2];
                        T yp = - p[1] / p[2];
                        // Apply second and fourth order radial distortion.
                        const T& l1 = camera[7];
                        const T& l2 = camera[8];
                        T r2 = xp*xp + yp*yp;
                        T distortion = 1.0 + r2  * (l1 + l2  * r2);
                        // Compute final projected point position.
                        const T& focal = camera[6];
                        T predicted_x = focal * distortion * xp;
                        T predicted_y = focal * distortion * yp;
                        // The error is the difference between the predicted and observed position.
                        residuals[0] = predicted_x - observed_x;
                        residuals[1] = predicted_y - observed_y;
                        return true;
                    }
                    // Factory to hide the construction of the CostFunction object from
                    // the client code.
                    static ceres::CostFunction* Create(const double observed_x,
                                                       const double observed_y) {
                        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                            new SnavelyReprojectionError(observed_x, observed_y)));
                                                       }
                                                       double observed_x;
                                                       double observed_y;
};


// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
public:
    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }
    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_points()           { return parameters_  + 9 * num_cameras_; }
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
    }
    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];
        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }
        return true;
    }
private:
    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }
    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;
};
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).


int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 2) {
        std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
        return 1;
    }
    BALProblem bal_problem;
    if (!bal_problem.LoadFile(argv[1])) {
        std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
        return 1;
    }
    const double* observations = bal_problem.observations();
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observations[2 * i + 0],
                                         observations[2 * i + 1]);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 bal_problem.mutable_camera_for_observation(i),
                                 bal_problem.mutable_point_for_observation(i));
    }
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    return 0;
}


/// Start Optimization here

/// Step 1: Initialization
Eigen::Matrix3d Rotn;
Rotn(0, 0) = 1; Rotn(0, 1) = 0; Rotn(0, 2) = 0;
Rotn(1, 0) = 0; Rotn(1, 1) = 1; Rotn(1, 2) = 0;
Rotn(2, 0) = 0; Rotn(2, 1) = 0; Rotn(2, 2) = 1;
Eigen::Vector3d axis_angle;
ceres::RotationMatrixToAngleAxis(Rotn.data(), axis_angle.data());

Eigen::Vector3d Translation = Eigen::Vector3d(0, 0, 0);
Eigen::VectorXd R_t(6);
R_t(0) = axis_angle(0);
R_t(1) = axis_angle(1);
R_t(2) = axis_angle(2);
R_t(3) = Translation(0);
R_t(4) = Translation(1);
R_t(5) = Translation(2);
/// Step2: Defining the Loss function (Can be NULL)
//                    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
//                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
ceres::LossFunction *loss_function = NULL;

/// Step 3: Form the Optimization Problem
ceres::Problem problem;
problem.AddParameterBlock(R_t.data(), 6);
for (int i = 0; i < all_normals.size(); i++) {
    Eigen::Vector3d normal_i = all_normals[i];
    std::vector<Eigen::Vector3d> lidar_points_i
    = all_lidar_points[i];
    for (int j = 0; j < lidar_points_i.size(); j++) {
        Eigen::Vector3d lidar_point = lidar_points_i[j];
        ceres::CostFunction *cost_function = new
        ceres::AutoDiffCostFunction<CalibrationErrorTerm, 1, 6>
        (new CalibrationErrorTerm(lidar_point, normal_i));
        problem.AddResidualBlock(cost_function, loss_function, R_t.data());
    }
}

/// Step 4: Solve it
ceres::Solver::Options options;
options.max_num_iterations = 200;
options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
options.minimizer_progress_to_stdout = true;
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
std::cout << summary.FullReport() << '\n';
