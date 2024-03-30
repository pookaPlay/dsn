#if !defined(ThreshSegment_Int_H__)
#define ThreshSegment_Int_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"

#include "Logger.h"
#include "SegmentWS.h"
#include "SegmentParameter.h"

/** 
* This provides a light weight interface to test the interactive threshold used in segmentation
*
*/

class ThreshSegment_Int 
{
	public:
		SegmentWS segment;
		//SegmentACD segment;
		SegmentParameter param;

		int baseMode; 
		cv::Mat img, seg, picy;

		void Init(cv::Mat &imgf, int runSeg = 1);
		void Update(float threshold); 
		void Render();
		void Run();
		void RenderSeg(cv::Mat &img, cv::Mat &seg, cv::Mat &picy);

		ThreshSegment_Int();
		virtual ~ThreshSegment_Int();
	private:				

		Logger m_logger;				
}; 


#endif 

