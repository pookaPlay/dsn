#if !defined(SegmentParameter_H__)
#define SegmentParameter_H__

#include<vector>
#include<string>
using namespace std;

#include "Logger.h"
#include "Storable.h"
#include "opencv2/opencv.hpp"

class SegmentParameter;

/**
* An object to encapsulate parameters that affect the segmentation
* These parameters are stored as configurations and can be adjusted
* by the user through the GUI.
* 
**/
class SegmentParameter : public Storable
{
  public:	  
	  /**
	  * scaleFactor: Images are resized by scaleFactor before the segmentation pipeline is applied
	  * and then scaled back to the original size before returning e.g.  
	  *		scaleFactor = 1.0	keeps the original image scale (default)
	  *		scaleFactor = 0.5	reduces by the image by a half (in x and y dimensions) 
	  **/
	  double scaleFactor;
	  
	  /** 
	  * preType: Applies a smoothing (noise reduction) filter before calculating the gradient.  
	  *		preType = 0		No pre-filter (default) 
	  *		preType = 1		Use bilateral filter
	  *		preType = 2		Use opening (get rid of dark spots) by reconstruction filter (edge preserving)
	  *		preType = 3		Use closing (get rid of light spots) by reconstruction filter (edge preserving)
	  **/
	  int preType;
	  
	  /**
	  * preSize: Window size of filter for preType ne 0.
	  **/
	  int preSize;
	  
	  /**
	  * gradType: Applies a smoothing (noise reduction) filter before calculating the gradient.
	  *		gradType = 0		No gradient (unusual!)
	  *		gradType = 1		Use Scharr 1st order derivatives (default)
	  *		gradType = 2		Morphological gradient
	  *		gradType = 3		Laplacian (2nd order derivatives)
	  **/
	  int gradType; 
	  
	  /**
	  * gradSize: Window size of gradient filter (only applies to Morphological gradient).
	  **/
	  int gradSize;

	  /**
	  * postType: Applies post gradient smoothing filter.
	  *		postType = 0		No post-filter 
	  *		postType = 1		Apply Gaussian smoothing (default)
	  *		postType = 2		Apply morphological opening 
	  *		postType = 3		Apply morphological opening by reconstruction
	  **/
	  int postType; 

	  /**
	  * postSize: Window size of filter for postType ne 0.
	  **/
	  int postSize;

	  /**
	  * threshold: Level in the segmentation hierarchy. 
	  *            Typically adjusted by the user with a slider control  
	  *		threshold = 0.0		Lowest level (most segments)
	  *		threshold = 1.0		highest level (1 segment)
	  **/
	  double threshold;

	  /**
	  * absoluteThreshold: Flag to indicate that threshold should be treated as an absolute value (and is not necessaryily between 0 and 1). 
	  *		absoluteThreshold = 0		threshold is 0->1 and will be scaled to data range (default)
	  *		absoluteThreshold = 1		threshold can be any value and will not be scaled. 
	  **/	
	  int absoluteThreshold;

	  /**
	  * waterfall: Flag to indicate we are using waterfall approach instead of connected component approach.
	  *		waterfall = 0		Use horizontal cuts of the merge tree (default)
	  *		waterfall = 1		Apply a second watershed on top of the first watershed.
	  **/
	  int waterfall;

	  double & GetScale() { return(this->scaleFactor); }; 
	  int & GetPreType() { return(this->preType); };
	  int & GetPreSize() { return(this->preSize); };
	  int & GetGradType() { return(this->gradType); };
	  int & GetGradSize() { return(this->gradSize); };
	  int & GetPostType() { return(this->postType); };
	  int & GetPostSize() { return(this->postSize); };
	  int & GetWaterfall() { return(this->waterfall); };
	  double & GetThreshold() { return(this->threshold); };

	  void Print();

	  void Save(string fname);
	  void Load(string fname);
	  void Save(cv::FileStorage &fs) const;
	  void Load(cv::FileStorage &fs);
	  void Load(cv::FileNodeIterator &fs);

	  /**
	   * Used for testing
	   **/
	  int CompareTo(SegmentParameter &p2);

	  SegmentParameter(double aScaleType, int aPreType, int aPostType);
	  SegmentParameter(double aScaleType, int aPreType, int aPreSize, int aPostType, int aPostSize);

	  SegmentParameter();	  
	  virtual ~SegmentParameter();

  private:
	  Logger m_logger;

} ;

#endif 
