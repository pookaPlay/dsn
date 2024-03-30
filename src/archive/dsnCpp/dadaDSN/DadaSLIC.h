#if !defined(DadaSLIC_H__)
#define DadaSLIC_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"


class DadaSLIC
{
	public:		

		static void Run(cv::Mat &img, cv::Mat &bseg, int desiredK, double compactness = 10.0);

		DadaSLIC();
		virtual ~DadaSLIC();
	
	protected:			
		static Logger m_logger;								
}; 

#endif 

