#if !defined(SegmentACD_H__)
#define SegmentACD_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "ACD.h"
#include <boost/graph/adjacency_list.hpp>
using namespace boost;

#include "SegmentParameter.h"


/**
* Attributes associated with vertices 
*/
typedef struct aVertexProp2
{
	int x, y;
	int label;	
	double weight;	
	vector<double> val;
} ACDVertexProp;

/**
* Attributes associated with edges
*/
typedef struct aEdgeProp2
{
	double weight;
	double wasWeight;
	int connected;
} ACDEdgeProp;

/**
* The key datastructure for segmentation is based on Boost graph
*/
typedef adjacency_list<setS, vecS, undirectedS, ACDVertexProp, ACDEdgeProp> ACDGraph;
typedef graph_traits<ACDGraph>::vertex_descriptor ACDVId;
typedef graph_traits<ACDGraph>::edge_descriptor ACDEId;
typedef ACDGraph::adjacency_iterator ACDNeighborIt;
typedef ACDGraph::out_edge_iterator ACDNeighborEdgeIt;
typedef ACDGraph::edge_iterator ACDEdgeIt;
typedef ACDGraph::vertex_iterator ACDNodeIt;

/**
* The core segmentation routine is a type of hierarchical watershed.
* We use the watershed cuts approach which formulates watersheds on edge weighted graphs:
* Jean Cousty, Gilles Bertrand, Laurent Najman, Michel Couprie, "Watershed Cuts: Minimum Spanning Forests and the Drop of Water Principle," 
* IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, no. 8, pp. 1362-1374, August, 2009 
*/

class SegmentACD
{
public:
	/**
	* Most of the segmentation work gets done here
	* This runs a two-layer segmentation method similar to a hierarchical watershed
	* The first layer is a standard watershed and identifies basins in the input image
	*/
	void Init(cv::Mat &img, SegmentParameter &param);


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

	SegmentACD();
	virtual ~SegmentACD();

private:
	int myW, myH;
	ACDGraph myM, myM2; 

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


