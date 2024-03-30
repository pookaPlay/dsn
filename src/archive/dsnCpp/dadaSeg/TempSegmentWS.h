#if !defined(SegmentWS_H__)
#define SegmentWS_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"

#include <boost/graph/adjacency_list.hpp>
using namespace boost;

/**
* Parameters that affect the segmentation are passed through this object 
*/
typedef struct aSegParam
{
	aSegParam() : inputType(0), threshold(0.0) {}
	int inputType;
	double threshold;	
} MamaSegParam;

/**
* Attributes associated with vertices 
*/
typedef struct aVertexProp
{
	int x, y;
	int basin;
	int label;
	double weight;
} MamaVertexProp;

/**
* Attributes associated with edges
*/
typedef struct aEdgeProp
{
	double weight;
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
	void Init(cv::Mat &img, MamaSegParam &param); 

	/**
	* Updates labels based on thresholding the second layer merge graph
	* @param param contains threshold which is expected to be in range [0..1]
	* @returns the number of unique labels in the segmentation
	*/
	void UpdateThreshold(MamaSegParam &param);

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
	* Provides access to segmentation graph	
	*/
	MamaGraph & GetGraph() { return(this->myM); };	

	/**
	* Clears all internal data
	*/
	void Clear();

	SegmentWS();
	virtual ~SegmentWS();

private:
	Logger m_logger;

	int myW, myH;
	MamaGraph myM, myM2; 

	int myThresholdIndex;
	vector<double> myBasinMins;
	vector< pair<double, int> > mySortedMergeDepth;
	vector< pair<int, int> > myMergePair;
	map< pair<int, int>, int> mySeenMergeBefore;
	vector<int> myLabelTree;

	/**
	* Estimates the gradient  
	*/
	void Preprocess(cv::Mat &img, cv::Mat &segInput, MamaSegParam &param);

	/**
	* Constructs a Mama boost graph from an image
	*/
	void InitGraph(cv::Mat &img);

	/**
	* Used to recurse label tree
	*/
	int GetFinalLabel(int lin);

	/**
	* Most of the segmentation work gets done here
	* This runs a two-layer segmentation method similar to a hierarchical watershed
	* The first layer is a standard watershed and identifies basins in the input image
	* The second...
	* @param param contains watershed parameters. Currently not used in this function.
	*/
	void ProcessMST(MamaSegParam &param);


};

#endif 








/* OLD API

/// ADHOC Message passing for SegmentAlgorith includes:
// Float1 : threshold level
// Float2 : which mode of segmentation 
//    for partial this is either 0: no extra image, 1: gen extra image
//    for multi this is 2+selected 
// Int1   : type of segmentation 
//    0 : opening of gradient
//    1 : morph gradient
//    2 : ??
//    3 : geodesic (mask) distance
// Int2   : has two roles depending on direction
//    set externally to number of prior components (used for viz on main screen)
//    set internally to specify if we need extra image for multi (Float2)
// Need to specifiy mode to preview on main screen (and use for apply)

class SegmentAlgorithm
{
  public:
	   
	  //vector< WatershedT::Pointer > watershedITK;	  	  
	  
	  Array *img;
	  Array markers;
	  int markerMode;
	  //MamaITKImageType::Pointer itkimg; 	  

	  int minx, miny, maxx, maxy, labelCount, myType;
	  Array mask, chip;
	  Array scaleIn, scaleOut;
	  int useChip;

	  void Preprocess(Array &input, Array &output, Array *extraOutput = NULL);
	  void ExtractAndOverlaySegmentationEdges(Array &input, Array &data, Array &output);		  
	  
	  void ExtractChipAndMaskFromRectangle(Array *imgData, Array *segData);
	  void CleanUp();

	  void Apply(int segType = 0);
	  void Update(Array *input, Array *output, float level, Array *extraOutput = NULL);

	  void InitWatershed(Array *imgData, int segType = 0, Array *markers = NULL);
	  
	  void InitPartialWatershed(Array *imgData, Array *segData, vector<int> &vals, int segType = 0);
	  void ExtractChipAndMaskFromSelections(Array *imgData, Array *segData, vector<int> &vals);
	  int UpdatePartial(Array *input, Array *inputSeg, Array *output, float level, Array *extraOutput = NULL);

	  void InitRectWatershed(Array *imgData, Array *segData);
	  
	  void ModifyMarkerInput(Array &tempGrad, Array &markerMod);

	  void InitMultiPartial(Array *imgData, Array *segData, vector<int> &vals);
	  int UpdateMultiPartial(Array *input, Array *segInput, Array *preview, Array *output, float level, int selection, Array *extraOutput = NULL);

	  void Simple(Array *input, float level, float threshold, Array *output);
	  void SimpleLineSeg(Array *input, float level, float threshold, Array *output);

	  SegmentAlgorithm();
	  virtual ~SegmentAlgorithm();
};


*/


