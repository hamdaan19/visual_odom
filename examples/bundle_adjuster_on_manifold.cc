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

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "eigen3/Eigen/Dense"

// Manifold Functor
// struct SpecialOrthogonalFunctor {
//     template <typename T>
//     bool Plus(const T* x, const T* delta, T* x_plus_delta) const {

//     }

//     template <typename T> 
//     bool Minus(const T* y, const T* x, T* y_minus_x) {

//     }
// };

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
        return camera_rotations_ + camera_index_[i] * 9; 
    }

    double* camera_translation(int i) {
        return camera_positions_ + camera_index_[i] * 3; 
    }

    double* camera_intrinsic(int i) {
        return camera_intrinsics_ + camera_index_[i] * 3; 
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
        camera_rotations_ = new double[9*num_cameras_]; 
        camera_positions_ = new double[3*num_cameras_]; 
        camera_intrinsics_ = new double[3*num_cameras_]; 

        // Extracting Camera information and storing it in different arrays
        for (int i = 0; i < num_cameras_; ++i) {
            // ------------- Extracting Rotation ------------- // 
            double angleAxis[3] = {*(parameters_ + 9*i), *(parameters_ + 9*i + 1), *(parameters_ + 9*i + 2)};
            double rot[9];
            // Converting Angle Axis to Rotation Matrix
            ceres::AngleAxisToRotationMatrix(angleAxis, rot);
            // Copying rotation matrix coefficients into camera_rotations_ array. 
            std::copy(rot, rot+9, camera_rotations_ + 9*i);

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
    double *camera_rotations_;   // In rotation matrix form 
    double *camera_positions_;   // 3D vector
    double *camera_intrinsics_;  // Includes focal length, f, and distortion parameters - k1 and k2. 
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError
{
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {
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
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T &l1 = camera[7];
        const T &l2 = camera[8];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + r2 * (l1 + l2 * r2);

        // Compute final projected point position.
        const T &focal = camera[6];
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
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
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
    for (int i = 0; i < bal_problem.num_observations(); ++i)
    {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.

        ceres::CostFunction *cost_function = SnavelyReprojectionError::Create(
            observations[2 * i + 0], observations[2 * i + 1]);
        problem.AddResidualBlock(cost_function,
                                 nullptr /* squared loss */,
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

    // Playing around with code

    // for (int i = 3; i < 8; i++) {
    //     double *myArray = bal_problem.mutable_camera_for_observation(i);

    //     // Converting Angle-Axis rotation to rotation matrix
    //     double angleAxis[3] = {*myArray, *(myArray + 1), *(myArray + 2)};
    //     double Rot[9];

    //     ceres::AngleAxisToRotationMatrix(angleAxis, Rot);

    //     for (int j = 0; j < 9; j++)
    //     {
    //         std::cout << *(myArray + j) << " ";
    //     }
    //     std::cout << "Rotation Matrix:\n"; 
    //     for (int j = 0; j < 9; j++)
    //     {
    //         std::cout << Rot[j] << " ";
    //     }

    //     Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R; 
    //     R << 
    //     Rot[0], Rot[3], Rot[6], 
    //     Rot[1], Rot[4], Rot[7], 
    //     Rot[2], Rot[5], Rot[8]; 

    //     std::cout << "\nR: \n" << R; 

    //     std::cout << "\n\n";
    // }

    for (int i = 5; i < 10; ++i) {
        double* param = bal_problem.mutable_camera_for_observation(i);
        std::cout << "Angle Axis: " << *param << " " << *(param+1) << " " << *(param+2) << "\n"; 

        double* rot = bal_problem.camera_rotation(i);
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R;
        R <<
        rot[0], rot[3], rot[6],
        rot[1], rot[4], rot[7],
        rot[2], rot[5], rot[8];

        std::cout << "R:\n" << R << "\n"; 

        double* t = bal_problem.camera_translation(i); 
        Eigen::Map<Eigen::Vector3d> t_(t); 
        std::cout << "t:\n" << t_ << "\n"; 



        double* intrinsic_params = bal_problem.camera_intrinsic(i); 
        double f = *intrinsic_params; 
        double k1 = *(intrinsic_params+1);
        double k2 = *(intrinsic_params + 2);

        std::cout << "f: " << f << " k1: " << k1 << " k2: " << k2 << std::endl; 
    }



    return 0;
}
