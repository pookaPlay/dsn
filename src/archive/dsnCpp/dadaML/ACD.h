#if !defined(ACD_H__)
#define ACD_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"

#include "Logger.h"

class ACD 
{
	public:					
		 void Train(cv::Mat &x, cv::Mat &y);
		 void Apply(cv::Mat &x, cv::Mat &y, cv::Mat &result);
		 
		ACD();
		virtual ~ACD();
private:
		Logger m_logger;
		cv::Mat mxy, Q; 
}; 


#endif 

