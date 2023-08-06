#include <iostream>
#include <chrono> 

#include <opencv2/core/core.hpp> 
#include <opencv2/core/types.hpp> 
#include <opencv2/features2d/features2d.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/calib3d.hpp> 

/*
THIS IS FILE CONTAINS CODE TO DETECT AND MATCH ORB FEATURES ACROSS THREE CAMERA FRAMES.
THE CODE IS NOT OPTIMIZED. IT CAN BE USED AS REFERENCE ONLY. 
*/

void simple_matcher(
    cv::Mat query_descriptors, 
    cv::Mat train_descriptors, 
    std::vector<cv::DMatch>* good_matches_ptr,
    std::vector<int>* query_idx_ptr,
    std::vector<int>* train_idx_ptr ) {

    // Creating Matcher Object
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMINGLUT); 

    // Start matching
    std::vector<cv::DMatch> matches; 
    matcher->match(query_descriptors, train_descriptors, matches); 

    // Clearing Arrays for safety
    good_matches_ptr->clear();
    query_idx_ptr->clear(); 
    train_idx_ptr->clear(); 

    // Filter Good Matches
    // good matches based on Hamming distance
    for (int i = 0; i < matches.size(); i++){
        if (matches[i].distance <= 20) {
            good_matches_ptr->push_back(matches[i]);           // Appending good matches
            query_idx_ptr->push_back(matches[i].queryIdx);     // Appending indices of good matches from image 1 
            train_idx_ptr->push_back(matches[i].trainIdx);     // Appending indices of good matches from image 2
        }
    } 


}


int main(int argc, char** argv) {
    if (argc != 4){
        std::cout << "usage: requires three arguments for paths to images\n";
        return 1; 
    }

    // Reading images 
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR); 
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR); 
    cv::Mat img_3 = cv::imread(argv[3], cv::IMREAD_COLOR); 

    // Defining extractors and matchers 
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2, keypoints_3; 
    cv::Mat descriptors_1, descriptors_2, descriptors_3; 
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(1000); 
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMINGLUT); 

    // Detecting ORB features
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(); 
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2); 
    detector->detect(img_3, keypoints_3); 

    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2); 
    descriptor->compute(img_3, keypoints_3, descriptors_3); 

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(); 

    std::chrono::duration<double> time_taken = t1-t0; 
    std::cout << "ORB time cost: " << time_taken.count() << " s"<< std::endl; 

    // Descriptor Matching 
    // Finding Matches in images 1 and 2
    // Nameing conventions: good_idx_<img-number>_<image-par>, good_matches_<image-pair>
    std::vector<cv::DMatch> good_matches_12; 
    std::vector<int> good_idx_1_12, good_idx_2_12; 

    simple_matcher(descriptors_1, descriptors_2, &good_matches_12, &good_idx_1_12, &good_idx_2_12); 

    // Finding Matches in images 1 and 3
    // Nameing conventions: good_idx_<img-number>_<image-par>, good_matches_<image-pair>
    std::vector<cv::DMatch> good_matches_13; 
    std::vector<int> good_idx_1_13, good_idx_3_13; 
    simple_matcher(descriptors_1, descriptors_3, &good_matches_13, &good_idx_1_13, &good_idx_3_13); 

    // Common Matches
    std::vector<int> common_indices_img1, common_indices_img2, common_indices_img3; 
    std::vector<cv::DMatch> common_matches_img12, common_matches_img13;

    for (int i = 0; i < good_idx_1_13.size(); i++){
        int q_idx = good_idx_1_13[i]; 
        for (int j = 0; j < good_idx_1_12.size(); j++) {
            int o_idx = good_idx_1_12[j]; 
            if (q_idx == o_idx) {
                // Keypoint indices
                common_indices_img1.push_back(q_idx); 
                common_indices_img2.push_back(good_idx_2_12[j]);
                common_indices_img3.push_back(good_idx_3_13[i]);

                // Matches
                common_matches_img13.push_back(good_matches_13.at(i));
                common_matches_img12.push_back(good_matches_12.at(j));
            }
        }
    }

    // Computing Essential Matrix

    // DEFINING CAMERA INTRINSIC MATRIX

    // cv::Mat K = cv::Mat::zeros(3, 3, CV_64F); 
    // K.at<double>(1,2) = 1.0; 
   
    cv::Mat K = (cv::Mat_<double>(3,3) << 7.18856e+02, 0.0, 6.071928e+02, 0, 7.18856e+02, 1.852157e+02, 0, 0, 1); 
    std::cout << "Camera Intrinsic Matrix:\n" << K << std::endl;

    /*
    
    _InputArray is a class that can be constructed from Mat, Mat_<T>, Matx<T, m, n>, std::vector<T>, 
    std::vector<std::vector<T> > or std::vector<Mat>. It can also be constructed from a matrix expression.
    
    */

    std::vector<cv::Point2f> points_1, points_2, points_3; 
    cv::KeyPoint::convert(keypoints_1, points_1, common_indices_img1);   // Converting vector<KeyPoints> --> vector<Point2f>
    cv::KeyPoint::convert(keypoints_2, points_2, common_indices_img2);   
    cv::KeyPoint::convert(keypoints_3, points_3, common_indices_img3);   // 2D Pixel coordinates
    
    cv::Mat E = cv::findEssentialMat(points_1, points_2, K, cv::LMEDS); // Computing the essential matrix

    std::cout << "Essential Matrix:\n" << E << std::endl; 

    // Recovering the pose from the Essential Matrix
    cv::Mat R, t;       // Rotation and translation
    cv::recoverPose(E, points_1, points_2, K, R, t); 

    std::cout << "R:\n" << R << "\nt:\n" << t << std::endl;  

    // TRIANGULATING POINTS

    // reshape(1) - specifying one channel, t() - transposes
    cv::Mat mat_1 = cv::Mat(points_1).reshape(1).t(); 
    cv::Mat mat_2 = cv::Mat(points_2).reshape(1).t();

    std::cout <<" size (r x c): " << mat_1.size << std::endl; 
    std::cout <<" c: " << mat_1.cols << std::endl; 

    // creating prjection matrices
    cv::Mat projMat_1 = K*cv::Mat::eye(cv::Size(4,3), CV_64F); // cv::Size(cols, rows)

    cv::Mat transformMat_2 = cv::Mat::eye(cv::Size(4,3), CV_64F); 
    // copying R and t into sectins of tranformMat_2
    // sourceMat(rowRange, colRange).copyTo( destinationMat(rowRange, colRange) )
    R(cv::Range(0,3),cv::Range(0,3)).copyTo( transformMat_2(cv::Range(0,3),cv::Range(0,3)) ); 
    t(cv::Range(0,3),cv::Range(0,1)).copyTo( transformMat_2(cv::Range(0,3),cv::Range(3,4)) ); 
    cv::Mat projMat_2 = K*transformMat_2; 

    // Triangulation
    cv::Mat points3D_homo; 
    cv::triangulatePoints(projMat_1, projMat_2, mat_1, mat_2, points3D_homo); 

    // std::cout << "Triangulated Points: " << std::endl; 
    // std::cout << points3D_homo(cv::Range(0,4), cv::Range(0,10)).t() << std::endl; 

    // Storing Triangulated Points
    std::vector<cv::Point3d> points3D;
    for (int i = 0; i < points3D_homo.cols; i++){

        cv::Mat p_mat = points3D_homo.col(i);
        float s = p_mat.at<float>(3,0); 

        // scaling p_mat
        p_mat = p_mat / s; 

        cv::Point3f p; 

        p.x = p_mat.at<float>(0,0);
        p.y = p_mat.at<float>(1,0);
        p.z = p_mat.at<float>(2,0); 

    }

    // Draw Matches
    cv::Mat img_match13; 
    cv::drawMatches(img_1, keypoints_1, img_3, keypoints_3, common_matches_img13, img_match13); 
    cv::imshow("matches13", img_match13);

    cv::Mat img_match12; 
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, common_matches_img12, img_match12); 
    cv::imshow("matches12", img_match12);
    
    cv::waitKey(0); 




}