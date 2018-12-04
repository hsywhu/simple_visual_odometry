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
//    vector<double> error;
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
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

cv::Mat ComputeE(cv::Matx33d F, cv::Mat K){
    cv::Mat F_ = cv::Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < 9; i++){
        F_.at<double>(i/3, i%3) = F(i/3, i%3);
    }
    // cv::Mat F_ = Mat(F);
    cv::Mat E = K.t() * F_ * K;
    return E;
}

cv::Mat ComputeRT(cv::Mat R, cv::Mat t){
    cv::Mat Rt = cv::Mat::zeros(4, 4, CV_64F);
    for (int i = 0; i < 9; i++){
        Rt.at<double>(i/3, i%3) = R.at<double>(i/3, i%3);
    }
    for (int i = 0; i < 3; i++){
        Rt.at<double>(i, 3) = t.at<double>(i, 0);
        Rt.at<double>(3, i) = 0;
    }
    Rt.at<double>(3, 3) = 1.0;
    return Rt;
}

bool Triangulation(cv::Mat P1, cv::Mat P2, cv::Point2f pt1, cv::Point2f pt2, cv::Mat RT){
    cv::Mat A = cv::Mat::zeros(4, 4, CV_64F);
    
    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);
    // cout << A << endl;

    // A.row(0) = pt1.y * P1.row(2) - P1.row(1);
    // A.row(1) = P1.row(0) - pt1.x * P1.row(2);
    // A.row(2) = pt2.y * P2.row(2) - P2.row(1);
    // A.row(3) = P2.row(0) - pt2.x * P2.row(2);
    cv::SVD svd_A(A);
    // cout << "vt" << svd_A.vt.rows << " " << svd_A.vt.cols << endl;
    cv::Mat X1 = svd_A.vt.row(svd_A.vt.rows - 1);
    X1 = X1.t();
    cv::Mat X2 = RT * X1;
    X1 = X1 / X1.at<double>(3, 0);
    X2 = X2 / X2.at<double>(3, 0);
    if (X1.at<double>(2, 0) > 0 && X2.at<double>(2, 0) > 0)
        return true;
    return false;
}


cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset,vector<cv::Point2f> next_subset, vector<int> img_size){
    // cv::Matx33d F;
    cv::Mat F;
    //fill the blank
    cv::Mat Normalize = cv::Mat::eye(3, 3, CV_64F);
    Normalize.at<double>(0, 0) = 2.0 / img_size[0];
    Normalize.at<double>(1, 1) = 2.0 / img_size[1];
    Normalize.at<double>(0, 2) = -1;
    Normalize.at<double>(1, 2) = -1;
    
    cv::Mat W = cv::Mat::ones(prev_subset.size(), 9, CV_64F);
    
    for (int i=0; i<W.rows ;i++){
		cv::Mat X1 = cv::Mat::ones(1, 3, CV_64F);
		X1.at<double>(0, 0) = prev_subset[i].x;
		X1.at<double>(0, 1) = prev_subset[i].y;
		cv::Mat X2 = cv::Mat::ones(1, 3, CV_64F);
		X2.at<double>(0, 0) = next_subset[i].x;
		X2.at<double>(0, 1) = next_subset[i].y;
		X1 = X1.t();
		X2 = X2.t();
		cv::Mat NX1 = cv::Mat::ones(X1.rows, X1.cols, CV_64F);
		NX1 = Normalize * X1;
		double u1 = NX1.at<double>(0, 0);
		double v1 = NX1.at<double>(1, 0);
		cv::Mat NX2 = cv::Mat::ones(X2.rows, X2.cols, CV_64F);
		NX2 = Normalize * X2;	// (3, 1) u=(0,0) v=(1,0)
		double u2 = NX2.at<double>(0, 0);
		double v2 = NX2.at<double>(1, 0);

		W.at<double>(i, 0) = u1 * u2;
        W.at<double>(i, 1) = u1 * v2;
        W.at<double>(i, 2) = u1;
        W.at<double>(i, 3) = v1 * u2;
        W.at<double>(i, 4) = v1 * v2;
        W.at<double>(i, 5) = v1;
        W.at<double>(i, 6) = u2;
        W.at<double>(i, 7) = v2;

        // W.at<double>(i, 0) = u1 * u2;
        // W.at<double>(i, 1) = u2 * v1;
        // W.at<double>(i, 2) = u2;
        // W.at<double>(i, 3) = v2 * u1;
        // W.at<double>(i, 4) = v1 * v2;
        // W.at<double>(i, 5) = v2;
        // W.at<double>(i, 6) = u1;
        // W.at<double>(i, 7) = v1;
	}
	
    cv::SVD svd(W);
    cv::Mat E = cv::Mat::ones(3, 3, CV_64F);
    for (int i = 0; i < 9; i++){
        E.at<double>(i/3, i%3) = svd.vt.at<double>(svd.vt.rows - 1, i);
    }
    cv::SVD svd1(E);
    
    cv::Mat w_mat = cv::Mat::eye(3, 3, CV_64F);
    // cout <<"w"<< svd1.w.rows <<" "<<svd1.w.cols << endl;
    // cout <<"w.size"<< svd1.w.size()<< endl;
    w_mat.at<double>(0, 0) = svd1.w.at<double>(0, 0);
    w_mat.at<double>(1, 1) = svd1.w.at<double>(1, 0);
    w_mat.at<double>(2, 2) = 0;
    F = svd1.u * w_mat * svd1.vt;
    
    F = Normalize.t() * F * Normalize;

    cv::Matx33d F_ = cv::Matx33d(F);
    
    return F_;
}

bool checkinlier(cv::Point2f prev_keypoint,cv::Point2f next_keypoint,cv::Matx33d Fcandidate,double d){
    //fill the blank
    cv::Matx33d t_F = Fcandidate.t();
    double u = prev_keypoint.x;
    double v = prev_keypoint.y;
    double a = t_F(0, 0) * u + t_F(0, 1) * v + t_F(0, 2);
    double b = t_F(1, 0) * u + t_F(1, 1) * v + t_F(1, 2);
    double c = t_F(2, 0) * u + t_F(2, 1) * v + t_F(2, 2);
    u = next_keypoint.x;
    v = next_keypoint.y;
    double dd = fabs(a*u+b*v+c)/sqrt(a*a+b*b);
    if (dd < d){
        return true;
	}
    return false;
}

void DrawEpipolar(vector<cv::Point2f> prev_subset, vector<cv::Point2f>next_subset, Mat img_1_origin, Mat img_2_origin, vector<int> img_size, cv::Matx33d F){
    Mat img_1 = img_1_origin.clone();
    Mat img_2 = img_2_origin.clone();
    for ( int i=0; i < prev_subset.size() ;i++)
    {

        Point pt1;
        Point pt2;

        double xc = prev_subset[i].x;
        double yc = prev_subset[i].y;
        int sx = img_size[0];
        int sy = img_size[1];

        cv::Mat F_ = Mat(F);
        cv::Mat v = cv::Mat::zeros(1, 3, CV_64F);
        v.at<double>(0, 0) = xc;
        v.at<double>(0, 1) = yc;
        v.at<double>(0, 2) = 1;
        cv::Mat l = F_ * v.t();

        double l1_ = l.at<double>(0,0);
        double l2_ = l.at<double>(1,0);
    
        
        double s = sqrt(pow(l1_,2) + pow(l2_,2));

        cv::Mat lt = l/s;
        double l1 = lt.at<double>(0,0);
        double l2 = lt.at<double>(1,0);
        double l3 = lt.at<double>(2,0);
        double y1;
        double y2;
        double x1;
        double x2;
        if (l1 != 0){
            y1 = sy;
            y2 = 1;
            x1 = -(l2 * y1 + l3/l1);
            x2 = -(l2 * y2 + l3/l1);
        }else{
            x1 = sx;
            x2 = 1;
            y1 = -(l1 * x1 + l3/l2);
            y2 = -(l1 * x2 + l3/l2);
        }
        pt1.x = x1;
        pt1.y = y1;
        pt2.x = x2;
        pt2.y = y2;

        line(img_1, pt1, pt2, cv::Scalar(0,255,255));

        Point pt3;
        Point pt4;
        xc = next_subset[i].x;
        yc = next_subset[i].y;
        sx = img_size[0];
        sy = img_size[1];
        v.at<double>(0, 0) = xc;
        v.at<double>(0, 1) = yc;
        v.at<double>(0, 2) = 1;
        l = F_ * v.t();

        l1_ = l.at<double>(0,0);
        l2_ = l.at<double>(1,0);
        
        s = sqrt(pow(l1_,2) + pow(l2_,2));

        lt = l/s;
        l1 = lt.at<double>(0,0);
        l2 = lt.at<double>(1,0);
        l3 = lt.at<double>(2,0);
        if (l1 != 0){
            y1 = sy;
            y2 = 1;
            x1 = -(l2 * y1 + l3/l1);
            x2 = -(l2 * y2 + l3/l1);
        }else{
            x1 = sx;
            x2 = 1;
            y1 = -(l1 * x1 + l3/l2);
            y2 = -(l1 * x2 + l3/l2);
        }
        pt3.x = x1;
        pt3.y = y1;
        pt4.x = x2;
        pt4.y = y2;

        line(img_2, pt3, pt4, cv::Scalar(0,255,255));
    }
    hconcat(img_1,img_2,img_1);
    cv::imshow("epipolar", img_1);
    cv::waitKey(0);
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
    double d = 1.5;
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
    cout << "number of inlier: " << prev_subset.size() << endl;
    F = Findfundamental(prev_subset,next_subset,img_size);
    cv::Mat F_temp = Mat(F);
    F_temp = F_temp / F_temp.at<double>(2, 2);
    F = Matx33d(F_temp);
    cout << "F" << endl;
    cout << F << endl;
    // void DrawEpipolar(vector<cv::Point2f> prev_subset, vector<cv::Point2f>next_subset, Mat img_1, Mat img_2, vector<int> img_size, cv::Matx33d F){
    DrawEpipolar(prev_subset, next_subset, img_1, img_2, img_size, F);
    cv::Matx33d F_cv = (Matx33d)cv::findFundamentalMat(prev_subset, next_subset, cv::FM_8POINT, 1.5f);
    DrawEpipolar(prev_subset, next_subset, img_1, img_2, img_size, F_cv);
    cout << "F_cv" << endl;
    cout << F_cv << endl;
    // cout<<"Fundamental matrix is \n"<<F<<endl;

    // -----------------lab 9-----------------------
    // compute E from F
    // K:
    // camera.fx: 517.3
    // camera.fy: 516.5
    // camera.cx: 318.643040
    // camera.cy: 255.313989
/*    
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    K.at<double>(0, 0) = 517.3;
    K.at<double>(1, 1) = 516.5;
    K.at<double>(2, 2) = 1.0;
    K.at<double>(0, 2) = 318.643040;
    K.at<double>(1, 2) = 255.313989;
    cv::Mat E = K.t() * Mat(F) * K;

    // initialize W and Z to compute S and R
    cv::Mat W = cv::Mat::zeros(3, 3, CV_64F);
    W.at<double>(0, 1) = -1;
    W.at<double>(1, 0) = 1;
    W.at<double>(2, 2) = 1;
    cv::Mat Z = cv::Mat::zeros(3, 3, CV_64F);
    Z.at<double>(0, 1) = 1;
    Z.at<double>(1, 0) = -1;

    // compute S and R
    cv::SVD svd_SR(E);

    cv::Mat S1 = (-1 * svd_SR.u) * Z * svd_SR.u.t();
    cv::Mat R1 = svd_SR.u * W.t() * svd_SR.vt;
    if(cv::determinant(R1) < 0)
        R1 = -R1;
    // cv::Mat S2 = svd_SR.u * Z * svd_SR.u.t();
    cv::Mat R2 = svd_SR.u * W * svd_SR.vt;
    if(cv::determinant(R2) < 0)
        R2 = -R2;
    // cout << "R1\n" << R1 << endl;
    // cout << "R2\n" << R2 << endl;
    // cout << "E.w" << svd_SR.w << endl;

    cv::SVD svd_S1(S1);
    cv::Mat T1 = svd_S1.vt.row(svd_S1.vt.rows - 1).t();
    cv::Mat T2 = -1 * T1;
    // cv::SVD svd_S2(S2);
    // cv::Mat T2 = svd_S2.vt.row(svd_S2.vt.rows - 1);

    vector<int> counter (4, 0);
    cv::Mat K_3x4 = cv::Mat::zeros(3, 4, CV_64F);
    for (int i = 0; i < 9; i++){
        K_3x4.at<double>(i/3, i%3) = K.at<double>(i/3, i%3);
    }

    vector<cv::Mat> RT;
    vector<cv::Mat> P;

    RT.push_back(ComputeRT(R1, T1));
    P.push_back(K_3x4 * RT[0]);

    RT.push_back(ComputeRT(R1, T2));
    P.push_back(K_3x4 * RT[1]);

    RT.push_back(ComputeRT(R2, T1));
    P.push_back(K_3x4 * RT[2]);

    RT.push_back(ComputeRT(R2, T2));
    P.push_back(K_3x4 * RT[3]);

    cv::Mat camera_RT = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat camera_P = K_3x4 * camera_RT;

    for (int i = 0; i < prev_subset.size(); i++){
        for (int j = 0; j < RT.size(); j++){
            if (Triangulation(camera_P, P[j], prev_subset[i], next_subset[i], RT[j]))
                counter[j]++;
        }
    }

    int chosen_idx = 0;
    for (int j = 0; j < counter.size(); j++){
        if (counter[chosen_idx] < counter[j]){
            chosen_idx = j;
        }
        cout << j << " " << counter[j] << endl;
    }
    
    cout << "chosen P idx: " << chosen_idx << endl;
    cout << P[chosen_idx] << endl;

	cout << svd_SR.w.size() << endl;
    cout << svd_SR.w.at<double>(0, 0) << " " << svd_SR.w.at<double>(1, 0) << " " << svd_SR.w.at<double>(2, 0) << endl;
*/

    // test ------------------------------------------------------
   	// Create a random 3D scene
	cv::Mat points3D(1, 16, CV_64FC4);
	cv::randu(points3D, cv::Scalar(-5.0, -5.0, 1.0, 1.0), cv::Scalar(5.0, 5.0, 10.0, 1.0 ));


	// Compute 2 camera matrices
	cv::Matx34d C1 = cv::Matx34d::eye();
	cv::Matx34d C2 = cv::Matx34d::eye();
	cv::Mat K = cv::Mat::eye(3, 3, CV_64F);

	C2(2, 3) = 1;
    // C2(0, 0) = 2;
    // C2(1, 2) = 1.5;
    // C2(0, 3) = 1;

	// Compute points projection
	std::vector<cv::Point2f> points1;
	std::vector<cv::Point2f> points2;

	for(size_t i = 0; i < points3D.cols; i++)
	{
		cv::Vec3d hpt1 = C1*points3D.at<cv::Vec4d>(0, i);
		cv::Vec3d hpt2 = C2*points3D.at<cv::Vec4d>(0, i);

		hpt1 /= hpt1[2];
		hpt2 /= hpt2[2];

		cv::Point2f p1;
        p1.x = hpt1[0];
        p1.y = hpt1[1];
		cv::Point2f p2;
        p2.x = hpt2[0];
        p2.y = hpt2[1];
		points1.push_back(p1);
		points2.push_back(p2);
	}


	// Print
	// std::cout << C1 << std::endl;
	std::cout << C2 << std::endl;
	// std::cout << points3D << std::endl;
    
    // F = Findfundamental(points1,points2,img_size);
    // F = (Matx33d)cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 1.5f);
    // cv::Mat E = K.t() * Mat(F) * K;
    // initialize W and Z to compute S and R
    cv::Mat C2_ = Mat(C2);
    cv::Mat tt = C2_.col(3);
    cv::Mat tx = cv::Mat::zeros(3, 3, CV_64F);
    tx.at<double>(0, 1) = -1 * tt.at<double>(2, 0);
    tx.at<double>(0, 2) = tt.at<double>(1, 0);
    tx.at<double>(1, 0) = tt.at<double>(2, 0);
    tx.at<double>(1, 2) = -1 * tt.at<double>(0, 0);
    tx.at<double>(2, 0) = -1 * tt.at<double>(1, 0);
    tx.at<double>(2, 1) = tt.at<double>(0, 0);
    cv::Mat RR = cv::Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < 9; i++){
        RR.at<double>(i/3, i%3) = C2_.at<double>(i/3, i%3);
    }
    cv::Mat E = tx * RR;
    img_size[0] = 1;
    img_size[1] = 1;
    F = Findfundamental(points1,points2,img_size);
    E = K.t() * Mat(F) * K;

    cv::Mat W = cv::Mat::zeros(3, 3, CV_64F);
    W.at<double>(0, 1) = -1;
    W.at<double>(1, 0) = 1;
    W.at<double>(2, 2) = 1;
    cv::Mat Z = cv::Mat::zeros(3, 3, CV_64F);
    Z.at<double>(0, 1) = 1;
    Z.at<double>(1, 0) = -1;

    // compute S and R
    cv::SVD svd_SR(E);

    cv::Mat S1 = (-1 * svd_SR.u) * Z * svd_SR.u.t();
    cv::Mat R1 = svd_SR.u * W.t() * svd_SR.vt;
    if(cv::determinant(R1) < 0)
        R1 = -R1;
    // cv::Mat S2 = svd_SR.u * Z * svd_SR.u.t();
    cv::Mat R2 = svd_SR.u * W * svd_SR.vt;
    if(cv::determinant(R2) < 0)
        R2 = -R2;
    // cout << "R1\n" << R1 << endl;
    // cout << "R2\n" << R2 << endl;
    // cout << "E.w" << svd_SR.w << endl;

    cv::SVD svd_S1(S1);
    cv::Mat T1 = svd_S1.vt.row(svd_S1.vt.rows - 1).t();
    cv::Mat T2 = -1 * T1;
    // cv::SVD svd_S2(S2);
    // cv::Mat T2 = svd_S2.vt.row(svd_S2.vt.rows - 1);

    vector<int> counter (4, 0);
    cv::Mat K_3x4 = cv::Mat::zeros(3, 4, CV_64F);
    for (int i = 0; i < 9; i++){
        K_3x4.at<double>(i/3, i%3) = K.at<double>(i/3, i%3);
    }

    vector<cv::Mat> RT;
    vector<cv::Mat> P;

    RT.push_back(ComputeRT(R1, T1));
    P.push_back(K_3x4 * RT[0]);

    RT.push_back(ComputeRT(R1, T2));
    P.push_back(K_3x4 * RT[1]);

    RT.push_back(ComputeRT(R2, T1));
    P.push_back(K_3x4 * RT[2]);

    RT.push_back(ComputeRT(R2, T2));
    P.push_back(K_3x4 * RT[3]);

    // cv::Mat RT1 = ComputeRT(R1, T1);
    // cv::Mat P1 = K_3x4 * RT1;

    // cv::Mat RT2 = ComputeRT(R1, T2);
    // cv::Mat P2 = K_3x4 * RT2;

    // cv::Mat RT3 = ComputeRT(R2, T1);
    // cv::Mat P3 = K_3x4 * RT3;

    // cv::Mat RT4 = ComputeRT(R2, T2);
    // cv::Mat P4 = K_3x4 * RT4;

    cv::Mat camera_RT = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat camera_P = K_3x4 * camera_RT;

    for (int i = 0; i < points1.size(); i++){
        for (int j = 0; j < RT.size(); j++){
            if (Triangulation(camera_P, P[j], points1[i], points2[i], RT[j]))
                counter[j]++;
        }
    }

    for (int j = 0; j < counter.size(); j++){
        cout << j << " " << counter[j] << endl;
        cout << j << " " << RT[j] << endl;
    }

    return 0;
}
