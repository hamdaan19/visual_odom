#include <iostream>
#include <Eigen/Dense> 
#include <Eigen/SVD>
#include <opencv2/core/types.hpp>
#include <cstdlib>
#include <ctime> 
#include <cmath> 


double baseline(Eigen::MatrixXd proj_mtx){
    double X = proj_mtx(0,3);
    double Z = proj_mtx(2,3);
    double c_x = proj_mtx(0,2);
    double f = proj_mtx(0,0);

    double t_x = (X - c_x*Z)/f; // baseline in meters

    return t_x; 
}

Eigen::MatrixXd triangulate(double baseline, double fl, double c_x, double c_y, Eigen::MatrixXd* L, Eigen::MatrixXd* R){
    // std::cout << "Camera Projection Matrix: \n" << proj_mtx << std::endl; 
    
    Eigen::MatrixXd tri_mtx(4, 4); // triangulation matrix
    tri_mtx << baseline, 0, 0, 0, 0, baseline, 0, 0, 0, 0, baseline*fl, 0, 0, 0, 0, 1; 

    // std::cout << "Triangulation Matrix: \n" << tri_mtx << std::endl; 
    // L->at(i).pt.x = L->row(i)[0]

    double u, v_avg, x, y_avg, disp; 
    Eigen::VectorXd ptImg(4), pt3D(4), pt(4);
    Eigen::MatrixXd vecImg(4, 0), vecImgRaw(4,0);
    for (int i = 0; i < L->rows(); i++){
        
        u = L->row(i)[0];
        v_avg = (L->row(i)[1]+R->row(i)[1])/(double)2; 
        x = c_x - u;
        y_avg = c_y - v_avg;  
        disp = R->row(i)[0] - L->row(i)[0]; 

        ptImg << x, y_avg, 1, disp; 
        // pt3D = tri_mtx*ptImg;
        vecImg.conservativeResize(vecImg.rows(), vecImg.cols()+1);
        vecImg.col(vecImg.cols()-1) = ptImg/disp;  

    }   

    auto vec3D = tri_mtx*vecImg; 

    // std::cout << "Points in 3D: \n" << vec3D << std::endl;

    return vec3D;
}

Eigen::Vector3d centroid(Eigen::MatrixXd* set){
    auto set_ = set->block(0, 0, set->rows()-1, set->cols());
    auto ctd = set_.rowwise().sum()/set_.cols(); 
    return ctd; 
}

Eigen::Matrix4d computeTransform3Dto3D(Eigen::MatrixXd* set0, Eigen::MatrixXd* set1){

    Eigen::Vector3d ctd0 = centroid(set0);
    Eigen::Vector3d ctd1 = centroid(set1);

    int n = set0->cols();
    Eigen::Matrix3d X = Eigen::Matrix3d::Zero(); 
    Eigen::Matrix3d X_i; 

    for (int i=0; i < n; i++){
        auto err0 = set0->col(i).block(0,0,3,1) - ctd0;
        auto err1 = set1->col(i).block(0,0,3,1) - ctd1; 
        X_i = err0 * err1.transpose();
        X = X + X_i; 
    }

    // std::cout << "X: " << X << std::endl; 

    Eigen::JacobiSVD<Eigen::Matrix3d> svd;
    svd.compute(X, Eigen::ComputeFullV | Eigen::ComputeFullU); 
    // std::cout << "Singular Values: \n" << svd.singularValues() << std::endl;
    // std::cout << "U: \n" << svd.matrixU() << std::endl; 
    // std::cout << "V: \n" << svd.matrixV() << std::endl; 

    Eigen::Matrix3d Rot = svd.matrixV()*svd.matrixU().transpose(); 
    Eigen::Vector3d t = ctd1 - Rot*ctd0;

    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block(0,0,3,3) = Rot; 
    transformation.block(0,3,3,1) = t; 

    return transformation; 
}

Eigen::MatrixXd computeEssential8point(Eigen::Matrix<double, 8, 3>* points0, Eigen::Matrix<double, 8, 3>* points1){ 
    // points0 and points1 are in homogeneous coordinates i.e., [u, v, 1]

    // Step 1: Normalize the points. Mean is 0 and Variance is 1. // 

    // Computing the Mean
    Eigen::Vector3d mean0 = points0->transpose().rowwise().mean(); 
    Eigen::Vector3d mean1 = points1->transpose().rowwise().mean(); 

    // Computing the Variance
    Eigen::Vector3d var0 = Eigen::Vector3d::Zero(); 
    Eigen::Vector3d var1 = Eigen::Vector3d::Zero(); 

    for (int i = 0; i < 8; i++){
        Eigen::Vector3d arr0 = (points0->transpose().col(i) - mean0).array().square(); 
        Eigen::Vector3d arr1 = (points1->transpose().col(i) - mean1).array().square(); 

        var0 += arr0/8; 
        var1 += arr1/8; 
    }

    double s0_x, s0_y, s1_x, s1_y; // Scale values for normalization
    s0_x = 1/std::sqrt(var0[0]);
    s0_y = 1/std::sqrt(var0[1]);
    s1_x = 1/std::sqrt(var1[0]);
    s1_y = 1/std::sqrt(var1[1]); 

    Eigen::Matrix3d T0, T1;
    T0 << s0_x, 0, -1*s0_x*mean0[0], 0, s0_y, -1*s0_y*mean0[1], 0, 0, 1; 
    T1 << s1_x, 0, -1*s1_x*mean1[0], 0, s1_y, -1*s1_y*mean1[1], 0, 0, 1;

    auto normPoints0 = T0 * points0->transpose(); 
    auto normPoints1 = T1 * points1->transpose();

    // Step 2: Construct the coeffient matrix (A, 8x9). Ae = 0. // 
    double u0, v0, u1, v1;
    Eigen::Matrix<double, 8, 9> A; 
    for (int i = 0; i < 8; i++){ 
        u0 = normPoints0.col(i)[0];
        v0 = normPoints0.col(i)[1];
        u1 = normPoints1.col(i)[0];
        v1 = normPoints1.col(i)[1];

        A.row(i) = Eigen::Matrix<double, 1, 9>(u0*u1, u1*v0, u1, u0*v1, v0*v1, v1, u0, v0, 1);
    }

    // Step 3: Perform SVD of A. Last singular vector is an approximation e. // 
    Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> svd;
    svd.compute(A, Eigen::ComputeFullV | Eigen::ComputeFullU); 
    auto e_norm = svd.matrixV().col(8); 
    Eigen::Matrix3d E_norm = e_norm.reshaped(3,3).transpose(); 

    // Step 4: Compute E from E_norm. Szeliski page 706 // 
    Eigen::Matrix3d E = T1.transpose()*E_norm*T0; 

    // Step 5: Perform SVD of E and recompute E by changing the last singular value to 0 // 
    // This ensures that the rank of E is 2 // 
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> svd_E;
    svd_E.compute(E, Eigen::ComputeFullV | Eigen::ComputeFullU); 
    auto S = svd_E.singularValues(); S[2] = 0; // Changing the last singular value to 0
    E = svd_E.matrixU()*S.asDiagonal()*svd_E.matrixV().transpose(); 
    // std::cout << "Essential Matrix:\n " << E << std::endl;  

    return E; 
}

std::vector<int> RANSAC(Eigen::MatrixXd* points0, Eigen::MatrixXd* points1, double threshold){
    std::srand(std::time(NULL)); 
    std::cout << "Random numbers:\n"; 

    int iters = 25; 
    int score = 0, bestScore = 0; 
    Eigen::Matrix3d bestE; 
    std::vector<int> idx, bestIdx; 

    double u_0, v_0, u_1, v_1; 
    Eigen::Matrix<double, 8, 3> pts0, pts1; 

    for (int n = 0; n < iters; n++){

        // Select random points
        for (int i = 0; i <= 7; i++){
            int random = 0 + (std::rand() % points0->rows());
            u_0 = points0->row(random)[0];
            v_0 = points0->row(random)[1];
            u_1 = points1->row(random)[0];
            v_1 = points1->row(random)[1]; 

            pts0.row(i) = Eigen::RowVector3d(u_0, v_0, 1);
            pts1.row(i) = Eigen::RowVector3d(u_1, v_1, 1);

        }

        Eigen::Matrix3d E = computeEssential8point(&pts0, &pts1); 

        for (int m = 0; m < points0->rows(); m++){
            u_0 = points0->row(m)[0];
            v_0 = points0->row(m)[1];
            u_1 = points1->row(m)[0];
            v_1 = points1->row(m)[1]; 

            double val = Eigen::RowVector3d(u_1, v_1, 1)*E*Eigen::Vector3d(u_0, v_0, 1);
            if (std::abs(val) < threshold){
                score += 1; 
                idx.push_back(m);
            }
        }

        if (score > bestScore){
            bestScore = score;
            bestE = E; 
            bestIdx = idx; 
        }

        score = 0; // Reset score
        idx.clear(); // Clear all indices
    }

    std::cout << "Score: " << bestScore << std::endl; 
    std::cout << "Total Points: " << points0->rows() << std::endl; 

    return bestIdx; 

}


void RANSAC(Eigen::MatrixXd* pointsL0, 
                        Eigen::MatrixXd* pointsR0, 
                        Eigen::MatrixXd* pointsL1, 
                        Eigen::MatrixXd* pointsR1, 
                        double threshold){
    // Stereo RANSAC Implementation 
        
    std::srand(std::time(NULL)); // Seeding 

    int iters = 50; 
    int score = 0, bestScore = 0; 
    Eigen::Matrix3d bestE; 
    std::vector<int> idx, bestIdx; 

    double u_0, v_0, u_1, v_1; 
    Eigen::Matrix<double, 8, 3> pts0, pts1; 

    // Detect outliers from L0 R0 pair of keypoints. 
    for (int n = 0; n < iters; n++){

        // Select random points
        for (int i = 0; i <= 7; i++){
            int random = 0 + (std::rand() % pointsL0->rows());
            u_0 = pointsL0->row(random)[0];
            v_0 = pointsL0->row(random)[1];
            u_1 = pointsR0->row(random)[0];
            v_1 = pointsR0->row(random)[1]; 

            pts0.row(i) = Eigen::RowVector3d(u_0, v_0, 1);
            pts1.row(i) = Eigen::RowVector3d(u_1, v_1, 1);

        }

        Eigen::Matrix3d E = computeEssential8point(&pts0, &pts1); 

        for (int m = 0; m < pointsL0->rows(); m++){
            u_0 = pointsL0->row(m)[0];
            v_0 = pointsL0->row(m)[1];
            u_1 = pointsR0->row(m)[0];
            v_1 = pointsR0->row(m)[1]; 

            double val = Eigen::RowVector3d(u_1, v_1, 1)*E*Eigen::Vector3d(u_0, v_0, 1);
            if (std::abs(val) < threshold){
                score += 1; 
                idx.push_back(m);
            }
        }

        if (score > bestScore){
            bestScore = score;
            bestE = E; 
            bestIdx.clear();
            bestIdx = idx; 
        }

        score = 0; // Reset score
        idx.clear(); // Clear all indices
    }

    Eigen::MatrixXd inlierPointsL0(bestIdx.size(),2), inlierPointsR0(bestIdx.size(),2), inlierPointsL1(bestIdx.size(),2), inlierPointsR1(bestIdx.size(),2); 
    int row = 0; 
    for (int idx : bestIdx){
        inlierPointsL0.row(row) = pointsL0->row(idx);
        inlierPointsR0.row(row) = pointsR0->row(idx);
        inlierPointsL1.row(row) = pointsL1->row(idx);
        inlierPointsR1.row(row) = pointsR1->row(idx);
        row += 1; 
    }

    *pointsL0 = inlierPointsL0; 
    *pointsR0 = inlierPointsR0; 
    *pointsL1 = inlierPointsL1; 
    *pointsR1 = inlierPointsR1; 

}
