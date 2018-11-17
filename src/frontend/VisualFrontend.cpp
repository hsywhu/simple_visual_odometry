#include "frontend/VisualFrontend.h"
#include <chrono>
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
	//Initialise detector
	// std::string detectorType = "Feature2D.BRISK";

	// detector = Algorithm::create<FeatureDetector>(detectorType);
	// detector->set("thres", thresholdExtraction);
	//Initialize ID
    gpu_detector = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);

    d_pyrLK.winSize.width = 21;
    d_pyrLK.winSize.height = 21;
    d_pyrLK.maxLevel = 3;
    d_pyrLK.iters = 30;

	newId = 0;

}

void VisualFrontend::trackAndExtract(cv::Mat& im_gray, Features2D& trackedPoints, Features2D& newPoints)
{
	grid.setImageSize1(im_gray.size().width, im_gray.size().height);
	if (oldPoints.size() > 0)
	{
        //Track prevoius points with optical flow
		auto festart = chrono::steady_clock::now();
		// track(im_gray, trackedPoints);
		track1(im_gray, trackedPoints);
		auto feend = chrono::steady_clock::now();
		cout << "klt running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;

		//Save tracked points
		oldPoints = trackedPoints;
	}

	//Extract new points
	auto festart = chrono::steady_clock::now();
	// extract(im_gray, newPoints);
	extract1(im_gray, newPoints);
	auto feend = chrono::steady_clock::now();
	cout << "new feature time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;
	
	//save old image
	im_prev = im_gray;

	// Reset Occupancy Grid
	grid.resetGrid1();
}

void VisualFrontend::extract1(Mat& im_gray, Features2D& newPoints)
{
    // fill the blank
	cv::gpu::GpuMat d_src, d_pts;
	d_src.upload(im_gray);
	gpu_detector(d_src, d_pts);
	vector<cv::Point2f> keypoints;
	VisualFrontend::downloadpts(d_pts, keypoints);

	// TODO Grid
	for (int i = 0; i < keypoints.size(); i++)
	{
		if (grid.isNewFeature1(keypoints[i])){
			oldPoints.addPoint(keypoints[i], newId);
			newPoints.addPoint(keypoints[i], newId);
			newId++;
		}
	}
}

void VisualFrontend::track1(Mat& im_gray, Features2D& trackedPoints)
{
    // fill the blank
	double distance_thres = 0.5;
	cv::gpu::GpuMat prevImg, nextImg, prevPts, nextPts, backPts, status;
	prevImg.upload(im_prev);
	nextImg.upload(im_gray);
	vector<cv::Point2f> oldPoints_vec;
	oldPoints_vec = oldPoints.getPoints();
	cv::Mat oldPoints_cv(1, (int) oldPoints_vec.size(), CV_32FC2, (void*) &oldPoints_vec[0]);
	prevPts.upload(oldPoints_cv);
	
	// Forward
	d_pyrLK.sparse(prevImg, nextImg, prevPts, nextPts, status);
	vector<cv::Point2f> nextPoints_vec;
	VisualFrontend::downloadpts(nextPts, nextPoints_vec);
	std::vector<unsigned char> forward_status;
	VisualFrontend::downloadmask(status, forward_status);
	
	// Backward
	d_pyrLK.sparse(nextImg, prevImg, nextPts, backPts, status);
	vector<cv::Point2f> backPoints_vec;
	VisualFrontend::downloadpts(backPts, backPoints_vec);
	
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
