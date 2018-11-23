#include "frontend/VisualFrontend.h"
#include <chrono>
#include <fstream>
using namespace std;
using namespace cv;


void VisualFrontend::downloadpts(const GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

void VisualFrontend::downloadmask(const GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

VisualFrontend::VisualFrontend()
{

	newId = 0;

}

void VisualFrontend::trackAndExtract(cv::Mat& im_gray, Features2D& trackedPoints, Features2D& newPoints)
{
	grid.setImageSize1(im_gray.size().width, im_gray.size().height);
	double klt_time;
	double new_feature_time;
	if (oldPoints.size() > 0)
	{
        //Track prevoius points with optical flow
		auto festart = chrono::steady_clock::now();
		// track(im_gray, trackedPoints);
		track1(im_gray, trackedPoints);
		auto feend = chrono::steady_clock::now();
		cout << "klt running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;
		klt_time =  chrono::duration <double, milli> (feend-festart).count();
		//Save tracked points
		oldPoints = trackedPoints;
	}

	//Extract new points
	auto festart = chrono::steady_clock::now();
	// extract(im_gray, newPoints);
	extract1(im_gray, newPoints);
	auto feend = chrono::steady_clock::now();
	cout << "new feature time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;
	new_feature_time =  chrono::duration <double, milli> (feend-festart).count();
	//save old image
	im_prev = im_gray;
	
	
	// save running time to file
	std::ofstream myfile;
	myfile.open("result_klt.csv", std::ofstream::out | std::ofstream::app);
	myfile << klt_time <<"\n";
	myfile.close();
	
	// Reset Occupancy Grid
	grid.resetGrid1();
}

void VisualFrontend::extract1(Mat& im_gray, Features2D& newPoints)
{
    // fill the blank
	
	//Initialise detector
	std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
	detector->set("thres", 100);
	double new_feature_time;
	auto festart = chrono::steady_clock::now();
    vector<cv::KeyPoint> kps;
    vector<cv::Point2f> keypoints;
	detector->detect(im_gray, kps);
	for ( auto kp:kps ){
    		keypoints.push_back( kp.pt );
    }

	// TODO Grid
	for (int i = 0; i < keypoints.size(); i++)
	{
		if (grid.isNewFeature1(keypoints[i])){
			oldPoints.addPoint(keypoints[i], newId);
			newPoints.addPoint(keypoints[i], newId);
			newId++;
		}
	}
	auto feend = chrono::steady_clock::now();
	// save running time to file
	std::ofstream myfile;
	myfile.open("result_extract.csv", std::ofstream::out | std::ofstream::app);
	myfile << new_feature_time <<"\n";
	myfile.close();
}

void VisualFrontend::track1(Mat& im_gray, Features2D& trackedPoints)
{
    // fill the blank
	double distance_thres = 0.5;
	
	vector<cv::Point2f> oldPoints_vec;
	vector<cv::Point2f> nextPoints_vec;
	vector<cv::Point2f> backPoints_vec;
	vector<float> error_forward;
	vector<float> error_backward;
	std::vector<unsigned char> forward_status;
	std::vector<unsigned char> backward_status;
	oldPoints_vec = oldPoints.getPoints();

	// Forward
	cv::calcOpticalFlowPyrLK( im_prev, im_gray, oldPoints_vec, nextPoints_vec, forward_status, error_forward );
	
	// Backward
	cv::calcOpticalFlowPyrLK( im_gray, im_prev, nextPoints_vec, backPoints_vec, backward_status, error_backward );
		
	for (int i = 0; i < forward_status.size(); i++){
    	float square_distance = pow(oldPoints_vec[i].x - backPoints_vec[i].x, 2.0) + pow(oldPoints_vec[i].y - backPoints_vec[i].y, 2.0);
		forward_status[i] = (square_distance < distance_thres) && forward_status[i];
    }
	
	trackedPoints = Features2D(oldPoints, nextPoints_vec, forward_status);
	// trackedPoints = Features2D(oldPoints, nextPoints_vec);
	
	// Grid
	for (int i = 0; i < trackedPoints.size(); i++)
		grid.addPoint1(trackedPoints[i]);

	cout << "trackedPoints' size" << nextPoints_vec.size() << endl;
}
