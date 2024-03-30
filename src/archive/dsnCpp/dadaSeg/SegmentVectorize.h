#if !defined(SegmentVectorize_H__)
#define SegmentVectorize_H__

#include <map>
#include <vector>
#include "opencv2/opencv.hpp"

#include "Logger.h"

class SegmentVectorize
{
  public:		  	  
  
      void Convert(cv::Mat &seg, float scale=1.0);
      void Export(std::string fname);
      void Clear();

	  SegmentVectorize();
	  virtual ~SegmentVectorize();

  private:
      float m_scale; 
	  std::map<int, int> m_minx, m_maxx, m_miny, m_maxy, m_count; 
	  std::map<int, cv::Mat> m_msk;
      std::map<int, std::vector< cv::Point > > m_contours;
	  int m_imgw, m_imgh; 
	  Logger m_logger;
};


#endif 
