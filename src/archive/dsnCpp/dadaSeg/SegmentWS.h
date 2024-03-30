#if !defined(SegmentWS_H__)
#define SegmentWS_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"

#include <boost/graph/adjacency_list.hpp>
using namespace boost;

#include "SegmentParameter.h"


/**
* Attributes associated with vertices 
*/
typedef struct aVertexProp
{
	int x, y;
	int label;	
	int ownedBy;
	double weight;
} MamaVertexProp;

/**
* Attributes associated with edges
*/
typedef struct aEdgeProp
{
	double weight;
	double wasWeight;
	int connected;
} MamaEdgeProp;

/**
* The key datastructure for segmentation is based on Boost graph
*/
typedef adjacency_list<setS, vecS, undirectedS, MamaVertexProp, MamaEdgeProp> MamaGraph;
typedef graph_traits<MamaGraph>::vertex_descriptor MamaVId;
typedef graph_traits<MamaGraph>::edge_descriptor MamaEId;
typedef MamaGraph::adjacency_iterator MamaNeighborIt;
typedef MamaGraph::out_edge_iterator MamaNeighborEdgeIt;
typedef MamaGraph::edge_iterator MamaEdgeIt;
typedef MamaGraph::vertex_iterator MamaNodeIt;

/**
* The core segmentation routine is a type of hierarchical watershed.
* We use the watershed cuts approach which formulates watersheds on edge weighted graphs:
* Jean Cousty, Gilles Bertrand, Laurent Najman, Michel Couprie, "Watershed Cuts: Minimum Spanning Forests and the Drop of Water Principle," 
* IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, no. 8, pp. 1362-1374, August, 2009 
*/

class SegmentWS
{
public:
	/**
	* Most of the segmentation work gets done here
	* This runs a two-layer segmentation method similar to a hierarchical watershed
	* The first layer is a standard watershed and identifies basins in the input image
	*/
	void Init(cv::Mat &img, SegmentParameter &param);

	/**
	* Most of the segmentation work gets done here
	* This runs a two-layer segmentation method similar to a hierarchical watershed
	* The first layer is a standard watershed and identifies basins in the input image
	*/
	void InitNoPre(cv::Mat &img, SegmentParameter &param);

	/**
	* Updates labels based on thresholding the second layer merge graph
	* @param param contains threshold which is expected to be in range [0..1]
	* @returns the number of unique labels in the segmentation
	*/
	void UpdateThreshold(SegmentParameter &param);

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
	MamaGraph & GetBasinGraph() { return(this->myM); };	

	/**
	* Provides access to final segmentation graph
	*/
	MamaGraph & GetGraph() { return(this->myM2); };


	/**
	* Top level call for marker based segmentation
	*/
	void MarkerSegmentation(cv::Mat &img, cv::Mat &mark, SegmentParameter &param);

	/**
	* Provides access to final marker segmentation 
	*/
	void GetMarkerLabels(cv::Mat &segf);
	
	/**
	* Clears all internal data
	*/
	void Clear();	

	SegmentWS();
	virtual ~SegmentWS();

private:
	Logger m_logger;

	map<int, int> myMarkers; 
	int myMaxMarker; 
	int myW, myH;
	MamaGraph myM, myM2; 

	int myThresholdIndex;
	vector< pair<double, MamaEId> > mySortedMergeDepth;

	/**
	* Estimates the gradient  
	*/
	void Preprocess(cv::Mat &img, cv::Mat &segInput, SegmentParameter &param);

	/**
	* Constructs a Mama boost graph from an image
	*/
	void InitGraph(cv::Mat &img);

	/**
	* Constructs a Mama boost graph from an image and marker image
	*/
	void InitMarkerGraph(cv::Mat &img, cv::Mat &mark);

	void ProcessMarkerGraph(SegmentParameter &param);

	void RecurseOwnership(MamaVId starti, int owned, double thresh);
	void PropagateMarkers(SegmentParameter &param);
	/**
	* Used to label components
	*/
	int LabelVertices(MamaVId id, int label, double localThresh);

	/**
	* Most of the segmentation work gets done here
	* This runs a two-layer segmentation method similar to a hierarchical watershed
	* The first layer is a standard watershed and identifies basins in the input image
	* The second...	
	*/
	void ProcessMST(SegmentParameter &param);


};

#endif 


