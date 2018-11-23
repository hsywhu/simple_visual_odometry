//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std;

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>


using namespace cv;
using namespace cv::gpu;

void downloadpts(const GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

void downloadmask(const GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

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
    // vector<cv::KeyPoint> kps;
    vector<cv::Point2f> kps;

	// CPU version
    // std::string detectorType = "Feature2D.BRISK";
    // Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
	// detector->set("thres", 100);
    // detector->detect( img_1, kps );
    
    // GPU version
    GoodFeaturesToTrackDetector_GPU gpu_detector;
    gpu_detector = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);
    cv::gpu::GpuMat img_1_gpu, d_pts;
    img_1_gpu.upload(img_1);
    gpu_detector(img_1_gpu, d_pts);
    downloadpts(d_pts, kps);
    
    bool grid[num_grid_height][num_grid_width];
    float img_height = img_1.size().height;
    float img_width = img_1.size().width;

    for ( auto kp:kps ){
    	int width_idx = kp.x / (img_width / num_grid_width);
    	int height_idx = kp.y / (img_height / num_grid_height);

    	if ( grid[height_idx][width_idx] == false){
    		keypoints.push_back( kp );
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
    
    // cpu version
    // cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status_forward, error_forward );
    // cv::calcOpticalFlowPyrLK( img_2, img_1, next_keypoints, back_keypoints, status_backward, error_backward );
    
    // gpu version
    PyrLKOpticalFlow d_pyrLK;
    d_pyrLK.winSize.width = 21;
    d_pyrLK.winSize.height = 21;
    d_pyrLK.maxLevel = 3;
    d_pyrLK.iters = 30;
    cv::gpu::GpuMat img_2_gpu, prev_keypoints_gpu, next_keypoints_gpu, back_keypoints_gpu, status_forward_gpu, status_backward_gpu;
    img_1_gpu.upload(img_1);
    img_2_gpu.upload(img_2);
    cv::Mat prev_keypoints_cv(1, (int) prev_keypoints.size(), CV_32FC2, (void*) &prev_keypoints[0]);
	prev_keypoints_gpu.upload(prev_keypoints_cv);
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    d_pyrLK.sparse(img_1_gpu, img_2_gpu, prev_keypoints_gpu, next_keypoints_gpu, status_forward_gpu);
    d_pyrLK.sparse(img_2_gpu, img_1_gpu, next_keypoints_gpu, back_keypoints_gpu, status_backward_gpu);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time with forward backward in GPU："<<time_used.count()<<" seconds."<<endl;
    downloadpts(next_keypoints_gpu, next_keypoints);
    downloadpts(back_keypoints_gpu, back_keypoints);
    downloadmask(status_forward_gpu, status_forward);
    
    int outliers_forward_backward = 0;
    for (int i = 0; i < prev_keypoints.size(); i++){
    	float square_distance = pow(pow(prev_keypoints[i].x - back_keypoints[i].x, 2.0) + pow(prev_keypoints[i].y - back_keypoints[i].y, 2.0), 0.5);
    	if (square_distance < distance_thres && status_forward[i]){
    		final_keypoints_1.push_back(prev_keypoints[i]);
    		final_keypoints_2.push_back(next_keypoints[i]);
    		status_final.push_back(status_forward[i]);
    	}else{
			outliers_forward_backward++;
			}
    }
    cout<<"outliers_forward_backward: "<<outliers_forward_backward<<endl;
	

    // visualize all keypoints
    hconcat(img_1,img_2,img_1);
    for ( int i=0; i< final_keypoints_1.size() ;i++)
    {
        // cout<<(int)status_final[i]<<endl;
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
