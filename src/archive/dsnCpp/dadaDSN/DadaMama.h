#if !defined(DadaMama_H__)
#define DadaMama_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaWS.h"
#include "DadaWSTrain.h"


class DadaMama
{
	public:		

		void Train(cv::Mat &img, cv::Mat &bseg, cv::Mat &oseg, cv::Mat &mseg);
		void Apply(cv::Mat &img, cv::Mat &bseg);
		void Update(double threshold);
		
		void GetThreshold(double &thresh, double &threshMin, double &threshMax)
		{
			m_dada->GetOutputThresh(threshMin, threshMax, thresh); 
		};

		cv::Mat & GetOutputSeg() {
			return(m_outputSeg);
		};

		DadaMama();
		virtual ~DadaMama();
	
	protected:			
		cv::Mat m_outputSeg;

		DadaWSTrain m_trainData; 

		std::unique_ptr<DadaWS> m_dada;
		std::shared_ptr<DadaParam> m_param;
		Logger m_logger;								
}; 

#endif 

