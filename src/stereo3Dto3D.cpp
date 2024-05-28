#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <filesystem>
#include <iomanip>

double baseline(Eigen::MatrixXd proj_mtx); 
Eigen::MatrixXd triangulate(double baseline, double fl, double c_x, double c_y, Eigen::MatrixXd* L, Eigen::MatrixXd* R);
Eigen::Matrix4d computeTransform3Dto3D(Eigen::MatrixXd* set0, Eigen::MatrixXd* set1);
void RANSAC(Eigen::MatrixXd* pointsL0, Eigen::MatrixXd* pointsR0, Eigen::MatrixXd* pointsL1, Eigen::MatrixXd* pointsR1, double threshold); 

int NUM_IMAGES = 4540; 

int main(){

    std::string dataset_path_L = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_0";
    std::string dataset_path_R = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_1";
    std::vector<std::string> img_paths_L, img_paths_R;

    std::string filenameL, filenameR, str; 
    for (int img = 0; img <= NUM_IMAGES; img++){
        
        str = std::to_string(img);
        size_t n = 6;
    
        str = std::string(n - str.size(), '0').append(str);
    
        filenameL = dataset_path_L + "/" + str + ".png";
        filenameR = dataset_path_R + "/" + str + ".png";

        img_paths_L.push_back(filenameL);
        img_paths_R.push_back(filenameR); 

    }

    std::string img_path_L0 = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_0/000000.png";
    std::string img_path_R0 = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_1/000000.png";
    std::string img_path_L1 = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_0/000002.png";
    std::string img_path_R1 = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_1/000002.png";

    std::vector<std::vector<std::string>> paths;
    std::vector<std::string> pair0{img_path_L0, img_path_R0};
    std::vector<std::string> pair1{img_path_L1, img_path_R1};
    paths.push_back(pair0);
    paths.push_back(pair1);

    // Filter matches using the Lowe's ratio test
    const double ratio_thresh = 0.80; 

    int minHessian = 300; 
    cv::Ptr<cv::xfeatures2d::SURF> SURFDetector = cv::xfeatures2d::SURF::create(minHessian);

    int nfeatures = 0;
    int nOctaveLayers = 5;
    double contrastThreshold = 0.03;
    double edgeThreshold = 10;
    double sigma = 1; 

    cv::Ptr<cv::SIFT> SIFTDetector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

    // Variables and Containers
    std::vector<cv::KeyPoint> keyPointsL0, keyPointsR0, keyPointsL1, keyPointsR1; 

    Eigen::MatrixXd proj_mtx(3, 4);
        proj_mtx << 
            7.188560000000e+02, 
            0.000000000000e+00,
            6.071928000000e+02,
            -3.861448000000e+02,
            0.000000000000e+00,
            7.188560000000e+02,
            1.852157000000e+02,
            0.000000000000e+00,
            0.000000000000e+00,
            0.000000000000e+00,
            1.000000000000e+00,
            0.000000000000e+00; 

    Eigen::Matrix<double, 4, 4> state = Eigen::Matrix<double, 4, 4>::Identity(); 

    for (int i = 0; i < NUM_IMAGES-1; i++) {

        // Reading the two stereo pairs of images
        cv::Mat imgL0 = cv::imread(img_paths_L.at(i), cv::IMREAD_GRAYSCALE);
        cv::Mat imgR0 = cv::imread(img_paths_R.at(i), cv::IMREAD_GRAYSCALE);
        cv::Mat imgL1 = cv::imread(img_paths_L.at(i+1), cv::IMREAD_GRAYSCALE);
        cv::Mat imgR1 = cv::imread(img_paths_R.at(i+1), cv::IMREAD_GRAYSCALE);

        // Clear Variables
        keyPointsL0.clear(); keyPointsR0.clear(); keyPointsL1.clear(); keyPointsR1.clear();
        
        cv::Mat descriptorsL0, descriptorsR0, descriptorsL1, descriptorsR1; 

        // Find the features and descriptors in the images
        SIFTDetector->detectAndCompute(imgL0, cv::noArray(), keyPointsL0, descriptorsL0);
        SIFTDetector->detectAndCompute(imgR0, cv::noArray(), keyPointsR0, descriptorsR0); 
        SIFTDetector->detectAndCompute(imgL1, cv::noArray(), keyPointsL1, descriptorsL1);
        SIFTDetector->detectAndCompute(imgR1, cv::noArray(), keyPointsR1, descriptorsR1);

        // We consider imgL0 as the query image
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knnMatches;

        std::vector<cv::Mat> descriptorArr{descriptorsR0, descriptorsL1, descriptorsR1};
        matcher->add(descriptorArr);
        matcher->train();
        matcher->knnMatch(descriptorsL0, knnMatches, 4);

        std::vector<cv::DMatch> goodMatchesL0R0, goodMatchesL0L1, goodMatchesL0R1; 
        std::vector<cv::DMatch> commonMatchesL0R0, commonMatchesL0L1, commonMatchesL0R1; 
        std::vector<cv::DMatch> inlierMatchesL0R0, inlierMatchesL0L1, inlierMatchesL0R1; 
        std::vector<cv::KeyPoint> commonKeyPointsL0, commonKeyPointsR0, commonKeyPointsL1, commonKeyPointsR1; 

        int num_points = 0;
        Eigen::MatrixXd commonPointMatL0(num_points,2), commonPointMatR0(num_points,2), commonPointMatL1(num_points,2), commonPointMatR1(num_points,2); 

        for (size_t i = 0; i < knnMatches.size(); i++){
            int count = 0; 
            if ((knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance)){
                count += 1; 
                switch(knnMatches[i][0].imgIdx){
                    case 0:
                        goodMatchesL0R0.push_back(knnMatches[i][0]);
                        break;
                    case 1:
                        goodMatchesL0L1.push_back(knnMatches[i][0]);
                        break;
                    case 2:
                        goodMatchesL0R1.push_back(knnMatches[i][0]);
                        break; 
                }
      
            }
            if ((knnMatches[i][1].distance < ratio_thresh * knnMatches[i][2].distance)){
                count += 1; 
                switch(knnMatches[i][1].imgIdx){
                    case 0:
                        goodMatchesL0R0.push_back(knnMatches[i][1]);
                        break;
                    case 1:
                        goodMatchesL0L1.push_back(knnMatches[i][1]);
                        break;
                    case 2:
                        goodMatchesL0R1.push_back(knnMatches[i][1]);
                        break; 
                }
            }
            if ((knnMatches[i][2].distance < ratio_thresh * knnMatches[i][3].distance)){
                count += 1; 
                switch(knnMatches[i][2].imgIdx){
                    case 0:
                        goodMatchesL0R0.push_back(knnMatches[i][2]);
                        break;
                    case 1:
                        goodMatchesL0L1.push_back(knnMatches[i][2]);
                        break;
                    case 2:
                        goodMatchesL0R1.push_back(knnMatches[i][2]);
                        break; 
                }
            }
            if ((knnMatches[i][0].imgIdx != knnMatches[i][1].imgIdx) && 
                (knnMatches[i][1].imgIdx != knnMatches[i][2].imgIdx) && 
                (knnMatches[i][2].imgIdx != knnMatches[i][0].imgIdx) && (count == 3)){
                
                num_points += 1; 

                commonKeyPointsL0.push_back( keyPointsL0.at(knnMatches[i][0].queryIdx) );
                commonKeyPointsR0.push_back( keyPointsR0.at(goodMatchesL0R0.back().trainIdx) );
                commonKeyPointsL1.push_back( keyPointsL1.at(goodMatchesL0L1.back().trainIdx) );
                commonKeyPointsR1.push_back( keyPointsR1.at(goodMatchesL0R1.back().trainIdx) );

                commonMatchesL0R0.push_back(goodMatchesL0R0.back());
                commonMatchesL0L1.push_back(goodMatchesL0L1.back());
                commonMatchesL0R1.push_back(goodMatchesL0R1.back());

                commonPointMatL0.conservativeResize(num_points, Eigen::NoChange_t::NoChange);
                commonPointMatR0.conservativeResize(num_points, Eigen::NoChange_t::NoChange);
                commonPointMatL1.conservativeResize(num_points, Eigen::NoChange_t::NoChange);
                commonPointMatR1.conservativeResize(num_points, Eigen::NoChange_t::NoChange);

                commonPointMatL0.row(num_points-1) = Eigen::RowVector2d(keyPointsL0.at(knnMatches[i][0].queryIdx).pt.x, keyPointsL0.at(knnMatches[i][0].queryIdx).pt.y); 
                commonPointMatR0.row(num_points-1) = Eigen::RowVector2d(keyPointsR0.at(goodMatchesL0R0.back().trainIdx).pt.x, keyPointsR0.at(goodMatchesL0R0.back().trainIdx).pt.y); 
                commonPointMatL1.row(num_points-1) = Eigen::RowVector2d(keyPointsL1.at(goodMatchesL0L1.back().trainIdx).pt.x, keyPointsL1.at(goodMatchesL0L1.back().trainIdx).pt.y); 
                commonPointMatR1.row(num_points-1) = Eigen::RowVector2d(keyPointsR1.at(goodMatchesL0R1.back().trainIdx).pt.x, keyPointsR1.at(goodMatchesL0R1.back().trainIdx).pt.y); 

            }

        }

        std::cout << "goodMatchesL0R0: " << goodMatchesL0R0.size() << std::endl; 
        std::cout << "goodMatchesL0L1: " << goodMatchesL0L1.size() << std::endl; 
        std::cout << "goodMatchesL0R1: " << goodMatchesL0R1.size() << std::endl; 
        std::cout << "common KeyPoints: " << commonKeyPointsL0.size() << std::endl; 


        double bl = baseline(proj_mtx);
        double focal_len = proj_mtx(0,0);
        double c_x = proj_mtx(0,2);
        double c_y = proj_mtx(1,2);

        std::cout << commonPointMatL0.rows() << " " << commonPointMatL0.cols() << std::endl; 
        RANSAC(&commonPointMatL0, &commonPointMatR0, &commonPointMatL1, &commonPointMatR1, 0.006);
        RANSAC(&commonPointMatL1, &commonPointMatR1, &commonPointMatL0, &commonPointMatR0, 0.006);
        std::cout << commonPointMatL0.rows() << " " << commonPointMatL0.cols() << std::endl; 

        // for (int idx : index){
        //     inlierMatchesL0R0.push_back(commonMatchesL0R0.at(idx));
        //     inlierMatchesL0L1.push_back(commonMatchesL0L1.at(idx));
        //     inlierMatchesL0R1.push_back(commonMatchesL0R1.at(idx));
        // }

        // cv::Mat flann_img;
        // cv::drawMatches(imgL0, keyPointsL0, imgR0, keyPointsR0, commonMatchesL0R0, flann_img, 
        //     cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // cv::imshow("Good Matches", flann_img); 

        Eigen::MatrixXd set0 = triangulate(bl, focal_len, c_x, c_y, &commonPointMatL0, &commonPointMatR0);
        Eigen::MatrixXd set1 = triangulate(bl, focal_len, c_x, c_y, &commonPointMatL1, &commonPointMatR1);

        // std::cout << "3D: " << set0.transpose() << std::endl; 
        // std::cout << "3D1: " << set1.transpose() << std::endl; 

        auto T = computeTransform3Dto3D(&set0, &set1); 
        state = T*state; 
        std::cout << "\nTransformation: \n" << state << "\n\n";

        std::cout << "Img: " << i << "/" << NUM_IMAGES << std::endl; 

        cv::imshow("L", imgL0);
        cv::waitKey(1);

    }

    // int k = cv::waitKey(0);
    // while (k != 'q'){
    //     k = cv::waitKey(0);
    // }

}