// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "eigen3/Eigen/Dense"
#include <sophus/se3.hpp>
#include "sophus/ceres_manifold.hpp"

struct SpecialOrthogonalGroup
{
    template <typename T>
    bool Plus(const T *x, const T *delta, T *x_plus_delta) const
    {
        // x is a point on the manifold and its dimension is equal to the ambient size
        // delta is the point in the tangent space and its dimension is equal to the tangent size.
        // In the case of Special Orthogonal Lie Group, its tangent size is equal to e.
        // x_plus_delta is the perturbed point on the manifold.

        // x is a unit quaternion in the ordering: (w,x,y,z)
        T w = *x;
        T i = *(x + 1);
        T j = *(x + 2);
        T k = *(x + 3);
        // quaternion array with new ordering
        T quaternion[4] = {i, j, k, w};

        // Creating an Sophus manifold object.
        Eigen::Map<Sophus::SO3<T>> SO3_x(quaternion);

        // Creating an Eigen::Vector3d object from delta
        T delta_x = *(delta + 0);
        T delta_y = *(delta + 1);
        T delta_z = *(delta + 2);
        Eigen::Matrix<T, 3, 1> so3_delta;
        so3_delta << delta_x, delta_y, delta_z;

        // x plus delta operation
        Sophus::SO3<T> SO3_x_plus_delta = Sophus::SO3<T>::exp(so3_delta) * SO3_x;

        *(x_plus_delta + 0) = SO3_x_plus_delta.unit_quaternion().w();
        *(x_plus_delta + 1) = SO3_x_plus_delta.unit_quaternion().x();
        *(x_plus_delta + 2) = SO3_x_plus_delta.unit_quaternion().y();
        *(x_plus_delta + 3) = SO3_x_plus_delta.unit_quaternion().z();

        return true;
    }

    template <typename T>
    bool Minus(const T *y, const T *x, T *y_minus_x) const
    {
        // x is a point on the manifold and its dimension is equal to the ambient size
        // y is also a point on the manifold and its dimension is equal to the ambient size
        // y_minus_x is the point in the tangent space and its dimension is equal to the tangent size.

        // x and y are unit quaternions in the ordering: (w,x,y,z)
        T x_w = *(x + 0);
        T x_i = *(x + 1);
        T x_j = *(x + 2);
        T x_k = *(x + 3);
        // inverted x quaternion array with new ordering
        T minus_x_quaternion[4] = {-x_i, -x_j, -x_k, x_w};

        T y_w = *(y + 0);
        T y_i = *(y + 1);
        T y_j = *(y + 2);
        T y_k = *(y + 3);
        // y quaternion array with new ordering
        T y_quaternion[4] = {y_i, y_j, y_k, y_w};

        // Creating Sophus objects on the manifold
        Eigen::Map<Sophus::SO3<T>> SO3_minus_x(minus_x_quaternion);
        Eigen::Map<Sophus::SO3<T>> SO3_y(y_quaternion);

        Sophus::SO3<T> SO3_y_minus_x = SO3_y * SO3_minus_x;
        Eigen::Matrix<T, 3, 1> y_minus_x_vector = SO3_y_minus_x.log();

        *(y_minus_x + 0) = y_minus_x_vector[0];
        *(y_minus_x + 1) = y_minus_x_vector[1];
        *(y_minus_x + 2) = y_minus_x_vector[2];

        return true;
    }
};

// Read a Bundle Adjustment in the Large dataset.
class BALProblem
{
public:
    ~BALProblem()
    {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
        delete[] camera_intrinsics_;
        delete[] camera_positions_; 
        delete[] camera_rotations_; 
    }

    int num_observations() const { return num_observations_; }
    const double *observations() const { return observations_; }
    double *mutable_cameras() { return parameters_; }
    double *mutable_points() { return parameters_ + 9 * num_cameras_; }

    double *mutable_camera_for_observation(int i)
    {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double *mutable_point_for_observation(int i)
    {
        return mutable_points() + point_index_[i] * 3;
    }

    double* camera_rotation(int i) {
        return camera_rotations_ + camera_index_[i] * 4; 
    }

    double* camera_translation(int i) {
        return camera_positions_ + camera_index_[i] * 3; 
    }

    double* camera_intrinsic(int i) {
        return camera_intrinsics_ + camera_index_[i] * 3; 
    }

    void unit_quaternion(double* quat, double* unit_quat) {
        double norm = sqrt(pow(*quat, 2) + pow(*(quat + 1), 2) + pow(*(quat + 2), 2) + pow(*(quat + 3), 2));
        
        unit_quat[0] = *(quat + 0) / norm;
        unit_quat[1] = *(quat + 1) / norm;
        unit_quat[2] = *(quat + 2) / norm;
        unit_quat[3] = *(quat + 3) / norm;
    }

    bool LoadFile(const char *filename)
    {
        FILE *fptr = fopen(filename, "r");
        if (fptr == nullptr)
        {
            return false;
        };

        // Extract the first three numbers from the data file given as input
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);

        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];

        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
        // parameters_ is simply a really large array containing parameters for the
        // camera + 3D points which need to be estimated
        parameters_ = new double[num_parameters_];

        for (int i = 0; i < num_observations_; ++i)
        {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j)
            {
                FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
            }
        }

        for (int i = 0; i < num_parameters_; ++i)
        {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }

        // Assigning the sizes of camera arrays
        camera_rotations_ = new double[4*num_cameras_]; 
        camera_positions_ = new double[3*num_cameras_]; 
        camera_intrinsics_ = new double[3*num_cameras_]; 

        // Extracting Camera information and storing it in different arrays
        for (int i = 0; i < num_cameras_; ++i) {
            // ------------- Extracting Rotation ------------- // 
            double angleAxis[3] = {*(parameters_ + 9*i), *(parameters_ + 9*i + 1), *(parameters_ + 9*i + 2)};
            double quat[4];
            // Converting Angle Axis to Rotation Matrix
            ceres::AngleAxisToQuaternion(angleAxis, quat); // Given quaternion from this function has the format: w, x, y, z
            double norm = sqrt(pow(*quat, 2) + pow(*(quat + 1), 2) + pow(*(quat + 2), 2) + pow(*(quat + 3), 2));
            double unit_quat[4];
            this->unit_quaternion(quat, unit_quat); 
            // Reordering the quaterion from (w,x,y,z) -> (x,y,z,w). This is done because Eigen::Map<Sophus::SO3d> obj(quat)
            // reads quaternion in the following ordering: x,y,z,w.
            double reordered_unit_quat[4] = {unit_quat[1], unit_quat[2], unit_quat[3], unit_quat[0]};
            // Copying rotation matrix coefficients into camera_rotations_ array.
            std::copy(unit_quat, unit_quat + 4, camera_rotations_ + 4 * i);

            // ------------- Extracting Translation ------------- //
            std::copy(parameters_ + 9*i + 3, parameters_ + 9*i + 6, camera_positions_ + 3*i);

            // ------------- Extracting Camera Intrinsics ------------- //
            std::copy(parameters_ + 9*i + 6, parameters_ + 9*i + 9, camera_intrinsics_ + 3*i); 
        }
        return true;
    }

private:
    template <typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value)
    {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1)
        {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;

    int *point_index_;
    int *camera_index_;
    double *observations_;
    double *parameters_;
    double *camera_rotations_;   // Quaternions
    double *camera_positions_;   // 3D vector
    double *camera_intrinsics_;  // Includes focal length, f, and distortion parameters - k1 and k2. 
};


struct ReprojectionErrorFunctor
{
    ReprojectionErrorFunctor(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T *const camera_translation,  // Translation component of camera
                    const T* const camera_rotation,     // Rotation component in quaterion (x,y,z,w) 
                    const T* const camera_intrinsics,   // Focal length, distortion parameters k1 and k2
                    const T *const point,               // Location of 3D point
                    T *residuals) const
    {

        // Reordering quaternion
        // double ceres_quat[4];
        // ceres_quat[0] = *(camera_rotation+3); 
        // ceres_quat[1] = *camera_rotation; 
        // ceres_quat[2] = *(camera_rotation+1);
        // ceres_quat[3] = *(camera_rotation+2);  

        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::QuaternionRotatePoint(camera_rotation, point, p); 

        // Translating the point
        p[0] += camera_translation[0];
        p[1] += camera_translation[1];
        p[2] += camera_translation[2];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T &l1 = camera_intrinsics[1];
        const T &l2 = camera_intrinsics[2];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + r2 * (l1 + l2 * r2);

        // Compute final projected point position.
        const T &focal = camera_intrinsics[0];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorFunctor, 2, 3, 4, 3, 3>(
            new ReprojectionErrorFunctor(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
};

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    if (argc != 2)
    {
        std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
        return 1;
    }

    BALProblem bal_problem;
    if (!bal_problem.LoadFile(argv[1]))
    {
        std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
        return 1;
    }

    const double *observations = bal_problem.observations();

    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;

    // ceres::QuaternionManifold manifold = ceres::QuaternionManifold();
    ceres::Manifold *manifold = new ceres::AutoDiffManifold<SpecialOrthogonalGroup, 4, 3>;

    for (int i = 0; i < bal_problem.num_observations(); ++i)
    {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.

        ceres::CostFunction *cost_function = ReprojectionErrorFunctor::Create(
            observations[2 * i + 0], observations[2 * i + 1]);

        problem.AddParameterBlock(bal_problem.camera_rotation(i), 4, manifold); 
        problem.AddResidualBlock(cost_function,
                                 nullptr /* squared loss */,
                                 bal_problem.camera_translation(i),
                                 bal_problem.camera_rotation(i),
                                 bal_problem.camera_intrinsic(i),
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

    // delete manifold; 

    return 0;
}
