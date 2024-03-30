#if !defined(SegmentHACD_H__)
#define SegmentHACD_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "SegmentACD.h"
#include <boost/graph/adjacency_list.hpp>
using namespace boost;

#include "SegmentParameter.h"


class SegmentHACD
{
public:
	/**
	* Most of the segmentation work gets done here
	* This runs a two-layer segmentation method similar to a hierarchical watershed
	* The first layer is a standard watershed and identifies basins in the input image
	*/
	void Init(cv::Mat &img, SegmentParameter &param, int numLayers);


	/**
	* Updates labels based on thresholding the second layer merge graph
	* @param param contains threshold which is expected to be in range [0..1]
	* @returns the number of unique labels in the segmentation
	*/
	void UpdateThreshold(SegmentParameter &param);

	void UpdateBaseThreshold(SegmentParameter &param);
	
	int LabelBaseVertices(ACDVId id, int label, double localThresh);

	void ACDInitEdge();
	void WSCInitEdge();

	/**
	* Get segmentation labels as a float image
	* Image segmentation labels have range starting at 1 (compared to internal format starting at 0)
	* @param segf contains Mat that gets initialized and filled.
	*/
	void GetLabels(cv::Mat &segf);

	/**
	* Get segmentation labels as a float image and return basic facts about segmentation
	* Image segmentation labels have range starting at 1 (compared to internal format starting at 0)
	* @param segf contains Mat that gets initialized and filled.
	*/
	void GetLabelsWithStats(cv::Mat &segf, int &segCount, int &segMin, int &segMax);

	/**
	* Get basin labels as a float image
	* Image segmentation labels have range starting at 1 (compared to internal format starting at 0)
	* @param segf contains Mat that gets initialized and filled.
	*/
	void GetBasinLabels(cv::Mat &segf);

	/**
	* Provides access to basin segmentation graph	
	*/
	ACDGraph & GetBasinGraph() { return(this->myM); };	

	/**
	* Provides access to final segmentation graph
	*/
	ACDGraph & GetGraph() { return(this->myM2); };

	/**
	* Clears all internal data
	*/
	void Clear();

	Logger m_logger;

	SegmentHACD();
	virtual ~SegmentHACD();

private:
	int myW, myH;
	vector<ACDGraph> myM; 

	ACD myACD;

	int myThresholdIndex;
	vector< pair<double, ACDEId> > mySortedMergeDepth;
	vector< pair<double, ACDEId> > mySortedBaseEdge;

	/**
	* Estimates the gradient  
	*/
	void Preprocess(cv::Mat &img, cv::Mat &segInput, SegmentParameter &param);

	/**
	* Constructs a ACD boost graph from an image
	*/
	void InitGraph(cv::Mat &img);
	void InitGraph(std::vector< cv::Mat > &imgs);
	/**
	* Used to label components
	*/
	int LabelVertices(ACDVId id, int label, double localThresh);

	/**
	* Most of the segmentation work gets done here
	* This runs a two-layer segmentation method similar to a hierarchical watershed
	* The first layer is a standard watershed and identifies basins in the input image
	* The second...	
	*/
	void ProcessMST();


};

#endif 


