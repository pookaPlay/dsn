#if !defined(VizMat_H__)
#define VizMat_H__

#include "opencv2/opencv.hpp"
#include <vector>
#include <boost/assign/list_of.hpp> 
#include "Logger.h"

class VizMat {
public:	

	static void DisplayEdgeSeg(cv::Mat &img, cv::Mat &seg, std::string id, int waitVal, float mag);

	static void DisplaySeg(cv::Mat &in, std::string id, int waitVal, float mag); 

	static void DisplayFloat(cv::Mat &in, std::string id, int waitVal, float mag); 	
		
	static void DisplayByte(cv::Mat &in, std::string id, int waitVal, float mag);

	static void DisplayFloat(cv::Mat &in, std::vector< cv::Point2f > &pts, std::string id, int waitVal, float mag); 

	static void DisplayFloat(cv::Mat &in, std::vector< cv::Point3f > &pts, std::string id, int waitVal, float mag);
	
	static void FloatToByte(cv::Mat &in, cv::Mat &out, float resize = 1.0f);
	
	static void FloatTo8UC3(cv::Mat &in, cv::Mat &out, float maxVal = 255.0);

	static void Mosaic(std::vector< cv::Mat > &in, int nw, int nh, cv::Mat &out); 

	static int DisplayColorSeg(cv::Mat &in, std::string id, int waitVal, float mag);

	static void GenerateEdgeMaskFromSeg(cv::Mat &seg, cv::Mat &msk);  

	VizMat();
	virtual ~VizMat();

	static Logger m_logger;	
};


#endif
