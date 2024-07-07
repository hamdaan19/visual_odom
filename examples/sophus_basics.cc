#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <sophus/se3.hpp>

int main(int argc, char** argv) {

    // Rotation Matrix
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    // The equivalent quaternion
    Eigen::Quaterniond q(R);

    // Special Orthogonal Group - SO3

    // Sophus::SO3d can be constructed from rotation matrix
    Sophus::SO3d SO3_R(R);
    // Sophus::SO3d can be constructed from a quaternion (Eigen::Quaterniond) as well
    Sophus::SO3d SO3_q(q);

    // Also not that Sophus::SO3d::matrix() function returns an Eigen::Matrix3d type 
    std::cout << "SO(3) from matrix:\n" << SO3_R.matrix() << std::endl; 
    std::cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << std::endl;

    // Use the log operator to map a point from the manifold to the tangent space, aka the Lie Algebra
    // There is also the exponential operator, Sophus::SO3d::exp() which is the inverse operator
    // of Sophus::SO3d::log
    Eigen::Vector3d so3 = SO3_R.log(); 
    std::cout << "so3 (transposed): " << so3.transpose() << std::endl; 

    // The 'hat' operator transforms a vector into its skew-symmetric form
    std::cout << "hat(so3):\n" << Sophus::SO3d::hat(so3) << std::endl; 
    // The 'vee' operator transforms a skew-symmetric matrix into its vector form
    std::cout << "vee(hat(so3)): " << Sophus::SO3d::vee( Sophus::SO3d::hat(so3) ).transpose() << "\n"; 

    // Applying a perturbation
    Eigen::Vector3d perturbation_so3(2, 1e-2, 0); // A small perturbation 
    Sophus::SO3d SO3_perturbed = Sophus::SO3d::exp(perturbation_so3) * SO3_R; 
    std::cout << "Perturbed rotation:\n" << SO3_perturbed.matrix() << "\n";

    Sophus::SO3d delta = SO3_perturbed * SO3_R.inverse();
    Eigen::Vector3d delta_vector = delta.log(); 

    std::cout << "SO3 delta: " << delta_vector.transpose() << "\n"; 

    // Special Euclidean Group - SE3
    Eigen::Vector3d t(1, 0, 0);     // Translation 1 along X axis
    Sophus::SE3d SE3_Rt(R, t); 
    Sophus::SE3d SE3_qt(q, t); 

    std::cout << "SE3 from R and t:\n" << SE3_Rt.matrix() << std::endl; 
    std::cout << "SE3 from q and t:\n" << SE3_qt.matrix() << std::endl; 

    // The Lie Algebra for SE3 is a 6 dimensional vector
    typedef Eigen::Matrix<double, 6, 1> Vector6d; 
    // Applying the log operator
    Vector6d se3 = SE3_Rt.log(); 
    std::cout << "se3 (transposed): " << se3.transpose() << "\n"; 

    // Vee and Hat operator
    std::cout << "hat(se3):\n" << Sophus::SE3d::hat(se3) << "\n";
    std::cout << "vee(hat(se3))" << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << "\n"; 

    // Update (on the manifold) 
    Vector6d update_se3;
    update_se3.setZero(); 
    update_se3(0,0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    std::cout << "SE3 updated:\n" << SE3_updated.matrix() << "\n";

    // Some other functions
    std::cout << "Number of parameters in SO(3): " << Sophus::SO2d::num_parameters << "\n";
    std::cout << "Number of parameters in so(3): " << Sophus::SO2d::DoF << "\n";  

    // Playing around
    double quat[4] = {0, 0, 0, 1}; // Quaternion format: x, y, z, w
    Eigen::Map<Sophus::SO3d> rot(quat); // Mapping a quaternion rotation from double* type to Sophus::SO3d type

    std::cout << "Matrix:\n" << rot.matrix() << "\n"; 

}