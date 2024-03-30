#if !defined(Stump_H__)
#define Stump_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"

#include "Logger.h"

class Stump 
{
	public:							
		Logger m_logger; 

		static double RandMSTTrain(double &thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos = 0.0, double extraNeg = 0.0);		
		static double RandMSTError( double thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos = 0.0, double extraNeg = 0.0, double *totalPos = 0, double *totalNeg = 0); 
		static void TrainThreshold(double &thresh, cv::Mat &result, cv::Mat &labels, cv::Mat &weights);
		static double EvalThreshold(double &thresh, cv::Mat &result, cv::Mat &labels, cv::Mat &weights, double &dr, double &far);

		Stump();
		virtual ~Stump();
}; 


#endif 

