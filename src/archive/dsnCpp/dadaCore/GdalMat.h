#if !defined(GdalMat_H__)
#define GdalMat_H__

#include "opencv2/opencv.hpp"
#include <string>

#include "Logger.h"

class GdalMat {
public:

	static void ReadHeader(std::string fname, int &w, int &h, int &numbands);
	static void Read2DWindowAsFloat(std::string fname, cv::Mat &img, int x0, int y0, int x1, int y1);
	static void Write2DTiffFloat(std::string fname, cv::Mat &img);
	static void Read2DAsFloat(std::string fname, cv::Mat &img);
  	static void ReadColorAsFloat(std::string fname, std::vector< cv::Mat > &imgs, int numBands);
  	static void ReadColorAs8UC3(std::string fname, cv::Mat &img);  	

	GdalMat();
	virtual ~GdalMat();	

	static Logger m_logger;
};


#endif
