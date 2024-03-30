#if !defined(SegmentParameter_H__)
#define SegmentParameter_H__

#include<vector>
#include<string>
using namespace std;

#include "Logger.h"
#include "opencv2/opencv.hpp"

class SegmentParameter;

/**
* Parameters that affect the segmentation are passed through this object
* 
*/
class SegmentParameter 
{
  public:
	  double scaleFactor;
	  int preType, preSize;
	  int gradType, gradSize;
	  int postType, postSize;
	  double threshold;
	  int absoluteThreshold; 
	  int waterfall;
	  
	  double & GetScale() { return(this->scaleFactor); }; 
	  int & GetPreType() { return(this->preType); };
	  int & GetPreSize() { return(this->preSize); };
	  int & GetGradType() { return(this->gradType); };
	  int & GetGradSize() { return(this->gradSize); };
	  int & GetPostType() { return(this->postType); };
	  int & GetPostSize() { return(this->postSize); };
	  double & GetThreshold() { return(this->threshold); };

	  void Print();

	  void Save(string fname);
	  void Load(string fname);
	  void Save(cv::FileStorage &fs) const;
	  void Load(cv::FileStorage &fs);
	  void Load(cv::FileNodeIterator &fs);

	  //friend cv::FileStorage& operator<< (cv::FileStorage &fs, SegmentParameter &me);
	  //friend cv::FileStorage& operator>> (cv::FileStorage &fs, SegmentParameter &me);

	  Logger m_logger;

	  SegmentParameter(double aScaleType, int aPreType, int aPostType);
	  SegmentParameter(double aScaleType, int aPreType, int aPreSize, int aPostType, int aPostSize);

	  SegmentParameter();	  
	  virtual ~SegmentParameter();

} ;

#endif 
