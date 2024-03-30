#if !defined(ErrorEval_H__)
#define ErrorEval_H__

#include "opencv2/opencv.hpp"

#include <vector>
using namespace std;

class ErrorEval 
{
	public:					
		
		static double RandIndexMat(cv::Mat &seg1, cv::Mat &seg2, int verbose = 0);
		static double RandIndexMatSampled(cv::Mat &gtImg, cv::Mat &myImg, int numIter = 100000, int verbose = 0);

		ErrorEval();
		virtual ~ErrorEval();
}; 


#endif 