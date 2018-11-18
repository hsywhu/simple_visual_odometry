//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
int main( int argc, char** argv )
{
	int num_grid_height = 10;
	int num_grid_width = 10;
	float distance_thres = 1.0;

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    list< cv::Point2f > keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
	detector->set("thres", 100);


    detector->detect( img_1, kps );
    bool grid[num_grid_height][num_grid_width];
    float img_height = img_1.size().height;
    float img_width = img_1.size().width;

    for ( auto kp:kps ){
    	int width_idx = kp.pt.x / (img_width / num_grid_width);
    	int height_idx = kp.pt.y / (img_height / num_grid_height);

    	if ( grid[height_idx][width_idx] == false){
    		keypoints.push_back( kp.pt );
    		grid[height_idx][width_idx] = true;
    	}
    }

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    vector<cv::Point2f> back_keypoints;
    vector<cv::Point2f> final_keypoints_1;
    vector<cv::Point2f> final_keypoints_2;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status_forward;
    vector<float> error_forward;
    vector<unsigned char> status_backward;
    vector<float> error_backward;
    vector<unsigned char> status_final;
    vector<float> error_final;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status_forward, error_forward );
    cv::calcOpticalFlowPyrLK( img_2, img_1, next_keypoints, back_keypoints, status_backward, error_backward );
    
    for (int i = 0; i < prev_keypoints.size(); i++){
    	float square_distance = pow(prev_keypoints[i].x - back_keypoints[i].x, 2.0) + pow(prev_keypoints[i].y - back_keypoints[i].y, 2.0);
    	if (square_distance < distance_thres){
    		final_keypoints_1.push_back(prev_keypoints[i]);
    		final_keypoints_2.push_back(next_keypoints[i]);
    		status_final.push_back(status_forward[i]);
    		error_final.push_back(error_forward[i]);
    	}
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    // vector<cv::Point2f> next_keypoints;
    // vector<cv::Point2f> prev_keypoints;
    // vector<cv::Point2f> back_keypoints;
    // vector<cv::Point2f> final_keypoints_1;
    // vector<cv::Point2f> final_keypoints_2;
    // for ( auto kp:keypoints )
    //     prev_keypoints.push_back(kp);
    // vector<unsigned char> status;
    // vector<float> error;
    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    // cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;


    // visualize all  keypoints
    hconcat(img_1,img_2,img_1);
    for ( int i=0; i< final_keypoints_1.size() ;i++)
    {
        cout<<(int)status_final[i]<<endl;
        if(status_final[i] == 1)
        {
            Point pt;
            pt.x =  final_keypoints_2[i].x + img_2.size[1];
            pt.y =  final_keypoints_2[i].y;

            line(img_1, final_keypoints_1[i], pt, cv::Scalar(0,255,255));
        }
    }


    cv::imshow("klt tracker", img_1);
    cv::waitKey(0);

    return 0;
}
