#if !defined(ConnectedComponents_H__)
#define ConnectedComponents_H__

#include "opencv2/opencv.hpp"

#include <algorithm>
#include <numeric>
using namespace std;

#include "Logger.h"
static Logger m_logger(LOG_GET_LOGGER("ConnectedComponents"));

class ConnectedComponents 
{
public:	

	static void GetComponentLocations(cv::Mat &img, float thresh, std::vector< cv::Point2f > &myCentroids);
	static void GetComponentLocations(cv::Mat &prob, cv::Mat &mask, std::vector< cv::Point3f > &myCentroids);
	
	static int Label(cv::Mat &label_image);
	static int Centroids(cv::Mat &label_image, vector< cv::Point2f > &myCentroids);
	static int Centroids(cv::Mat &label_image, cv::Mat &conf_image, std::vector< cv::Point3f > &myCentroids);

	static void ComputeGradientBins(cv::Mat& img, cv::Mat& grad, cv::Mat& qangle, cv::Size paddingTL, cv::Size paddingBR, int nbins);
	ConnectedComponents();
	virtual ~ConnectedComponents();

};

#endif 
