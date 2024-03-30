#if !defined(SegmentTrain_H__)
#define SegmentTrain_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "SegmentWS.h"
#include "Logger.h"
#include "SetElement.h"
#include <boost/tuple/tuple.hpp>

class SegmentTrain 
{
	public:
		
		static double FindBestParameter(cv::Mat &img, cv::Mat &seg, SegmentParameter &param);
		static double FindBestThreshold(cv::Mat &img, cv::Mat &seg, SegmentParameter &param);

		SegmentTrain();
		virtual ~SegmentTrain();
	private:				

		Logger m_logger;				
}; 

#endif 

