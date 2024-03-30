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
* We use the watershed cuts approach which formulates watersheds on edge weighted graphs.
* Technical papers related to the method include:
*		"Segmentation and Learning in the Quantitative Analysis of Microscopy Images", Christy Ruggiero, Amy Ross, Reid Porter, Proceedings of SPIE, Image Processing: Machine Vision Applications VIII, 2014. [LA-UR-15-21115]
*		"Learning Watershed Cut Energy Functions", Reid Porter, Diane Oyen, Beate G. Zimmer, International Symposium on Mathemathical Morphology (ISMM-2015), 2015. [LA-UR-15-20316]
*/

class SegmentWS
{
public:
	/**
	* The upfront segmentation work gets done here. But the segmentation is not
	* complete until UpdateThreshold has been called. 
	* This runs a two-layer segmentation method similar to a hierarchical watershed
	* The first layer is a standard watershed and identifies basins in the input image
	* @param	img		CV_32F Mat image 
	* @param	param   All parameters that determine the segmentation 
	*/
	void Init(cv::Mat &img, SegmentParameter &param);

	/**
	* This version does not run filters. 
	* @param	img		CV_32F Mat image which is typically an edge estimate that has already been processed. 	
	* @param	param   All parameters that determine the segmentation
	*/
	void InitNoPre(cv::Mat &img, SegmentParameter &param);

	/**
	* Updates labels based on thresholding the second layer merge graph
	* @param param contains threshold which is expected to be in range [0..1]
	* @returns the number of unique labels in the segmentation
	*/
	void UpdateThreshold(SegmentParameter &param);

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
	* Clears all internal data
	*/
	void Clear();

	Logger m_logger;

	SegmentWS();
	virtual ~SegmentWS();

protected:
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
	* Used to label components
	*/
	int LabelVertices(MamaVId id, int label, double localThresh);

	/**
	* Most of the segmentation work gets done here
	* It is called by the Init methods. 
	* @param		param		The parameters for the segmentation	
	*/
	void ProcessMST(SegmentParameter &param);

	/**
	* Additional processing used in waterfall	
	**/
	void WaterfallMergeTree();

};

#endif 


