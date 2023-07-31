#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <algorithm> 
#include <time.h>

#include <iostream>

int main() {
    std::cout << "This is a \'feature detection and matching\' script." << std::endl; 

    // std::string image_path = cv::samples::findFile("car.jpg");
    std::string img_path_1 = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_0/000080.png";
    std::string img_path_2 = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_1/000080.png";
    std::string img_path_3 = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_0/000081.png";
    std::string img_path_4 = "/home/hamdaan/Datasets/data_odometry_gray/dataset/sequences/00/image_1/000081.png";


    // cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat img1 = cv::imread(img_path_1, cv::IMREAD_GRAYSCALE); 
    cv::Mat img2 = cv::imread(img_path_2, cv::IMREAD_GRAYSCALE); 
    cv::Mat img3 = cv::imread(img_path_3, cv::IMREAD_GRAYSCALE);
    cv::Mat img4 = cv::imread(img_path_4, cv::IMREAD_GRAYSCALE);

    // if(img.empty()) {
    //     std::cout << "Could not read the image: " << image_path << std::endl; 
    // }

    int minHessian = 200; 

    int nfeatures = 0;
    int nOctaveLayers = 5;
    double contrastThreshold = 0.03;
    double edgeThreshold = 10;
    double sigma = 1; 

    cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(); 
    cv::Ptr<cv::SIFT> SIFTDetector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    cv::Ptr<cv::xfeatures2d::SURF> SURFDetector = cv::xfeatures2d::SURF::create(minHessian);
    cv::Ptr<cv::ORB> ORBDetector = cv::ORB::create(1000); 
    std::vector<cv::KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4;
    cv::Mat descriptors1, descriptors2, descriptors3, descriptors4;

    time_t start = clock();

    // fastDetector->detect(img1, keypoints2);
    // SIFTDetector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1); 
    SIFTDetector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    SIFTDetector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2); 
    SIFTDetector->detectAndCompute(img3, cv::noArray(), keypoints3, descriptors3);
    SIFTDetector->detectAndCompute(img4, cv::noArray(), keypoints4, descriptors4);

    double elapsed = ( clock() - (double)start )/CLOCKS_PER_SEC;
    std::cout << "Elapsed 1: " << elapsed << std::endl; 

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<cv::DMatch> matches; 
    std::vector<std::vector<cv::DMatch>> knnMatches; 

    std::vector<cv::Mat> des{descriptors2, descriptors3, descriptors4};
    matcher->add(des);
    matcher->train();
    matcher->knnMatch(descriptors1, knnMatches, 6); // query, train

    elapsed = ( clock() - (double)start )/CLOCKS_PER_SEC;
    std::cout << "Elapsed 2: " << elapsed << std::endl; 

    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    cv::imshow("Matches", img_matches);

    std::cout << "Size of descriptors: " << descriptors1.rows << " " << descriptors1.cols << std::endl;
    std::cout << "Size of descriptors: " << descriptors2.rows << " " << descriptors2.cols << std::endl;

    int count = 0, zeros = 0, ones = 0, twos = 0; 
    // Filter matches using the Lowe's ratio test
    const double ratio_thresh = 0.80; 
    std::vector<cv::DMatch> goodMatches; 
    for (size_t i = 0; i < knnMatches.size(); i++){

        switch (knnMatches[i][0].imgIdx) {
            case 0:
                zeros += 1;
                break;
            case 1: 
                ones += 1;
                break;
            case 2:
                twos += 1;
                break; 
            default:
                break;
        }
        if ((knnMatches[i][3].distance < ratio_thresh * knnMatches[i][4].distance)  && (knnMatches[i][1].imgIdx == 1)){
            goodMatches.push_back(knnMatches[i][1]);
            std::cout << knnMatches[i][0].imgIdx << " " << knnMatches[i][1].imgIdx << " " << knnMatches[i][2].imgIdx << " " << knnMatches[i][3].imgIdx << knnMatches[i][4].imgIdx << knnMatches[i][5].imgIdx << std::endl; 
        }
    }

    cv::Mat flann_img;
    cv::drawMatches(img1, keypoints1, img3, keypoints3, goodMatches, flann_img, 
        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Good Matches", flann_img); 

    // for (cv::DMatch m : goodMatches){
    //     std::cout << m.queryIdx << " " << m.trainIdx << std::endl;
    //     cv::Point2f pointQ = keypoints1.at(m.queryIdx).pt; 
    //     cv::Point2f pointT = keypoints2.at(m.trainIdx).pt; 

    //     std::cout << "Query: " << pointQ << std::endl;
    //     std::cout << "Train: " << pointT << std::endl;
    // }

    // Store good matched keypoints and descriptors
    std::vector<cv::KeyPoint> matchedKeyPoints1, matchedKeyPoints2;
    cv::Mat matchedDescriptors1(goodMatches.size(), 64, CV_32F), matchedDescriptors2(goodMatches.size(), 64, CV_32F);
    int r = 0; 
    for (cv::DMatch m : goodMatches){
    //     descriptors1.row(m.queryIdx).copyTo(matchedDescriptors1.row(r));
    //     descriptors2.row(m.trainIdx).copyTo(matchedDescriptors2.row(r));
    //     r++;`
    }

    std::cout << "\nNo. of matches: " << knnMatches.size() << std::endl;
    std::cout << "No. of good matches: " << goodMatches.size() << std::endl; 
    std::cout << "No. of keypoints (query): " << keypoints1.size() << std::endl;
    std::cout << "No. of keypoints (train): " << keypoints2.size() << std::endl; 
    std::cout << descriptors1.rows << " " << descriptors1.cols << std::endl; 
    std::cout << descriptors2.rows << " " << descriptors2.cols << std::endl; 
    std::cout << knnMatches[0].size() << std::endl;
    std::cout << "Count: " << count << std::endl; 
    std::cout << "Zeros: " << zeros << std::endl; 
    std::cout << "Ones: " << ones << std::endl; 
    std::cout << "Twos: " << twos << std::endl; 


     
    cv::waitKey(0);
}