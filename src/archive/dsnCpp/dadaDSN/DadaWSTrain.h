#if !defined(DadaWSTrain_H__)
#define DadaWSTrain_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaWS.h"


class DadaWSTrain
{
	public:		

		
		cv::Mat & GetBasins() { 
			return(m_basins); 
		}; 
		
		cv::Mat & GetGroundTruth() {
			return(m_groundTruth);
		}; 

		std::vector< cv::Mat > & GetInput() {
			return(m_input);
		}; 

										
		void Init(std::vector< cv::Mat > &origImg, cv::Mat &origSeg, cv::Mat &editSeg, cv::Mat &basinSeg);

		DadaWSTrain(std::vector< cv::Mat > &origImg, cv::Mat &origSeg, cv::Mat &editSeg, cv::Mat &basinSeg);

		DadaWSTrain();
		virtual ~DadaWSTrain();
	
	protected:				
		cv::Mat m_basins; 
		cv::Mat m_groundTruth;
		std::vector< cv::Mat > m_input;
		cv::Range m_rowRange; 
		cv::Range m_colRange;
		
		Logger m_logger;								
}; 

#endif 

