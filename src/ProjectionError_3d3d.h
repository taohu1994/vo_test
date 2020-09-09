#ifndef PROJECTIONERROR_3D3D_H_
#define PROJECTIONERROR_3D3D_H_
#include <cmath>
#include <cstdio>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>




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

#endif 
