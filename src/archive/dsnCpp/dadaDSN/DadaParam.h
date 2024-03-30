#if !defined(DadaParam_H__)
#define DadaParam_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"


class DadaParam 
{
	public:				
		int numFeatures; 
		int featureWinRadius;		
		int featureWhiten; 
		int featureSubsetSize;
		int featureSingle;
		string modelName; 
		
		string errorType;				// "rand", "iid"
		string featureType;				// "edges", ""
		string featureEdgeType;			// "distance", "abs"
		string featureMergeType;		// "max", "min"
		string classifierType;			// "stump", "linear", "tree"		
		string classifierInitType;      // "uniform", "random"
		string classifierUpdateType;	// "svm", "std"
		string classifierLossType;		// "svm", "log"
		string finalLossType;

		string segmentationType;		// "cc", "ws"
		int trainSegmentationType;	
		string ensembleType; // "", "tree", "forest"
		string treeType; // "merge", "error"
		int		ensembleSize; // ensemble size
		int		ensembleDepth; // tree depth
		int		voteType;
		int    classifierNormalizeWeight;	// true to normalize after each update
		int    classifierGradientThreshold;		// true if we include threshold in gradient descent

		int maxTrainIterations;
		double lrate; 
		double alpha;				// slope
		int currentIteration; 
		int printUpdate; 
		int stepUpdate; 
		int randomSeed; 
		double posWeight;
		double negWeight;
		double sampleProb;
		int slicK; 
		double acdEdgeMultiplier;
		int acdNumTargetLabels; 
		int acdOrigMerge;
		int acdSynMerge;
		void Export(string fname); 
		void Import(string fname);

		DadaParam();
		virtual ~DadaParam();
	private:						

		Logger m_logger;				
}; 

#endif 

