////
//// Created by sicong on 08/11/18.
////
//
//#include <iostream>
//#include <fstream>
//#include <list>
//#include <vector>
//#include <chrono>
//using namespace std;
//
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/video/tracking.hpp>
//
//using namespace cv;
//int main( int argc, char** argv )
//{
//
//    if ( argc != 3 )
//    {
//        cout<<"usage: feature_extraction img1 img2"<<endl;
//        return 1;
//    }
//    //-- Read two images
//    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
//    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
//
//    list< cv::Point2f > keypoints;
//    vector<cv::KeyPoint> kps;
//
//    std::string detectorType = "Feature2D.BRISK";
//    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", 100);
//
//
//    detector->detect( img_1, kps );
//    for ( auto kp:kps )
//        keypoints.push_back( kp.pt );
//
//    vector<cv::Point2f> next_keypoints;
//    vector<cv::Point2f> prev_keypoints;
//    for ( auto kp:keypoints )
//        prev_keypoints.push_back(kp);
//    vector<unsigned char> status;
//    vector<float> error;
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
//    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
//
//    // visualize all  keypoints
//    hconcat(img_1,img_2,img_1);
//    for ( int i=0; i< prev_keypoints.size() ;i++)
//    {
//        cout<<(int)status[i]<<endl;
//        if(status[i] == 1)
//        {
//            Point pt;
//            pt.x =  next_keypoints[i].x + img_2.size[1];
//            pt.y =  next_keypoints[i].y;
//
//            line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
//        }
//    }
//
//    cv::imshow("klt tracker", img_1);
//    cv::waitKey(0);
//
//    return 0;
//}


//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

cv::Mat ComputeE(cv::Matx33d F, cv::Mat K){
    cv::Mat F_ = cv::Mat::zeros(3, 3, CV_32F);
    for (int i = 0; i < 9; i++){
        F_.at<float>(i/3, i%3) = F(i/3, i%3);
    }
    cv::Mat E = K.t() * F_ * K;
    return E;
}



cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset,vector<cv::Point2f> next_subset, vector<int> img_size){
    // cv::Matx33d F;
    cv::Mat F;
    //fill the blank
    cv::Mat Normalize = cv::Mat::eye(3, 3, CV_32F);
    Normalize.at<float>(0, 0) = 2.0 / img_size[0];
    Normalize.at<float>(1, 1) = 2.0 / img_size[1];
    Normalize.at<float>(0, 2) = -1;
    Normalize.at<float>(1, 2) = -1;
    
    cv::Mat W = cv::Mat::ones(prev_subset.size(), 9, CV_32F);
    
    for (int i=0; i<W.rows ;i++){
		cv::Mat X1 = cv::Mat::ones(1, 3, CV_32F);
		X1.at<float>(0, 0) = prev_subset[i].x;
		X1.at<float>(0, 1) = prev_subset[i].y;
		cv::Mat X2 = cv::Mat::ones(1, 3, CV_32F);
		X2.at<float>(0, 0) = next_subset[i].x;
		X2.at<float>(0, 1) = next_subset[i].y;
		X1 = X1.t();
		X2 = X2.t();
		cv::Mat NX1 = cv::Mat::ones(X1.rows, X1.cols, CV_32F);
		NX1 = Normalize * X1;
		double u1 = NX1.at<float>(0, 0);
		double v1 = NX1.at<float>(1, 0);
		cv::Mat NX2 = cv::Mat::ones(X2.rows, X2.cols, CV_32F);
		NX2 = Normalize * X2;	// (3, 1) u=(0,0) v=(1,0)
		double u2 = NX2.at<float>(0, 0);
		double v2 = NX2.at<float>(1, 0);
		W.at<float>(i, 0) = u1 * u2;
        W.at<float>(i, 1) = u1 * v2;
        W.at<float>(i, 2) = u1;
        W.at<float>(i, 3) = v1 * u2;
        W.at<float>(i, 4) = v1 * v2;
        W.at<float>(i, 5) = v1;
        W.at<float>(i, 6) = u2;
        W.at<float>(i, 7) = v2;
	}
	
    cv::SVD svd(W);
    cv::Mat E = cv::Mat::ones(3, 3, CV_32F);
    for (int i = 0; i < 9; i++){
        E.at<float>(i/3, i%3) = svd.vt.at<float>(svd.vt.rows - 1, i);
    }
    cv::SVD svd1(E);
    
    cv::Mat w_mat = cv::Mat::eye(3, 3, CV_32F);
    // cout <<"w"<< svd1.w.rows <<" "<<svd1.w.cols << endl;
    // cout <<"w.size"<< svd1.w.size()<< endl;
    w_mat.at<float>(0, 0) = svd1.w.at<float>(0, 0);
    w_mat.at<float>(1, 1) = svd1.w.at<float>(1, 0);
    w_mat.at<float>(2, 2) = 0;
    F = svd1.u * w_mat * svd1.vt;
    
    F = Normalize.t() * F * Normalize;

    cv::Matx33d F_ = cv::Matx33d(F);
    
    return F_;
}
bool checkinlier(cv::Point2f prev_keypoint,cv::Point2f next_keypoint,cv::Matx33d Fcandidate,double d){
    //fill the blank
    cv::Matx33d t_F = Fcandidate.t();
    float u = prev_keypoint.x;
    float v = prev_keypoint.y;
    float a = t_F(0, 0) * u + t_F(0, 1) * v + t_F(0, 2);
    float b = t_F(1, 0) * u + t_F(1, 1) * v + t_F(1, 2);
    float c = t_F(2, 0) * u + t_F(2, 1) * v + t_F(2, 2);
    u = next_keypoint.x;
    v = next_keypoint.y;
    float dd = fabs(a*u+b*v+c)/sqrt(a*a+b*b);
    if (dd < d){
        return true;
	}
    return false;
}

vector<int> FindImageSize(Mat img){
    // img_size[0]:width, img_size[1]:height
    vector<int> img_size;
    img_size.push_back(img.size().width);
    img_size.push_back(img.size().height);
    return img_size;
}

int main( int argc, char** argv )
{

    srand ( time(NULL) );

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
	vector<int> img_size = FindImageSize(img_1);
    list< cv::Point2f > keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
    detector->set("thres", 100);


    detector->detect( img_1, kps );
    for ( auto kp:kps )
        keypoints.push_back( kp.pt );

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    vector<cv::Point2f> kps_prev,kps_next;
    kps_prev.clear();
    kps_next.clear();
    for(size_t i=0;i<prev_keypoints.size();i++)
    {
        if(status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
        }
    }


    // p Probability that at least one valid set of inliers is chosen
    // d Tolerated distance from the model for inliers
    // e Assumed outlier percent in data set.
    double p = 0.99;
    double d = 1.5f;
    double e = 0.2;

    int niter = static_cast<int>(std::ceil(std::log(1.0-p)/std::log(1.0-std::pow(1.0-e,8))));
    Mat Fundamental;
    cv::Matx33d F,Fcandidate;
    int bestinliers = -1;
    vector<cv::Point2f> prev_subset,next_subset;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();

    for(int i=0;i<niter;i++){
        // step1: randomly sample 8 matches for 8pt algorithm
        unordered_set<int> rand_util;
        while(rand_util.size()<8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        vector<int> random_indices (rand_util.begin(),rand_util.end());
        for(size_t j = 0;j<rand_util.size();j++){
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }
        // step2: perform 8pt algorithm, get candidate F
        
        Fcandidate = Findfundamental(prev_subset,next_subset,img_size);
        // step3: Evaluate inliers, decide if we need to update the best solution
        int inliers = 0;
        for(size_t j=0;j<prev_keypoints.size();j++){
            if(checkinlier(prev_keypoints[j],next_keypoints[j],Fcandidate,d))
                inliers++;
        }
        if(inliers > bestinliers)
        {
            F = Fcandidate;
            bestinliers = inliers;
        }
        prev_subset.clear();
        next_subset.clear();
    }
	
    // step4: After we finish all the iterations, use the inliers of the best model to compute Fundamental matrix again.

    // for(size_t j=0;j<prev_keypoints.size();j++){
    
    for (size_t j=0; j<kps_prev.size();j++){
        if(checkinlier(kps_prev[j],kps_next[j],F,d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }
    }
    cout << prev_subset.size() << endl;
    F = Findfundamental(prev_subset,next_subset,img_size);
    cout<<"Fundamental matrix is \n"<<F<<endl;

    // -----------------lab 9-----------------------
    // compute E from F
    /* K
    camera.fx: 517.3
    camera.fy: 516.5
    camera.cx: 318.643040
    camera.cy: 255.313989
    */
    cv::Mat K = cv::Mat::zeros(3, 3, CV_32F);
    K.at<float>(0, 0) = 517.3;
    K.at<float>(1, 1) = 516.5;
    K.at<float>(2, 2) = 1.0;
    K.at<float>(0, 2) = 318.643040;
    K.at<float>(1, 2) = 255.313989;
    cv::Mat E = ComputeE(F, K);

    // initialize W and Z to compute S and R
    cv::Mat W = cv::Mat::zeros(3, 3, CV_32F);
    W.at<float>(0, 1) = -1;
    W.at<float>(1, 0) = 1;
    W.at<float>(2, 2) = 1;
    cv::Mat Z = cv::Mat::zeros(3, 3, CV_32F);
    Z.at<float>(0, 1) = 1;
    Z.at<float>(1, 0) = -1;

    // compute S and R
    cv::SVD svd_SR(E);
    cv::Mat S1 = (-1 * svd_SR.u) * Z * svd_SR.u.t();
    cv::Mat U1 = svd_SR.u * W.t() * svd_SR.vt;
    cv::Mat S2 = svd_SR.u * Z * svd_SR.u.t();
    cv::Mat U2 = svd_SR.u * W * svd_SR.vt;
	cout << svd_SR.w.size() << endl;
    cout << svd_SR.w.at<float>(0, 0) << " " << svd_SR.w.at<float>(1, 0) << " " << svd_SR.w.at<float>(2, 0) << endl;
    return 0;
}
