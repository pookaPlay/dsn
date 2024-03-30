#if !defined(SegmentWSMarker_H__)
#define SegmentWSMarker_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"

#include "SegmentWS.h"

class SegmentWSMarker : public SegmentWS
{
public:	

	void MarkerSegmentation(cv::Mat &img, cv::Mat &mark, SegmentParameter &param);

	void InitMarkerGraph(cv::Mat &img, cv::Mat &mark);

	void ProcessMarkerGraph(SegmentParameter &param);
	
	void PropagateMarkers(SegmentParameter &param);

	void RecurseOwnership(MamaVId starti, int owned, double thresh);

	SegmentWSMarker();
	virtual ~SegmentWSMarker();

private:
	map<MamaVId, int> m_pixelMarkers;
	map<int, int> m_markers;
	int m_maxMarkers;

};

#endif 


