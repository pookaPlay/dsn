/* 
* Unless otherwise indicated, this software has been authored by an
* employee of Los Alamos National Security LLC (LANS), operator of the 
* Los Alamos National Laboratory under Contract No. DE-AC52-06NA25396 with 
* the U.S. Department of Energy. 
*
* The U.S. Government has rights to use, reproduce, and distribute this information. 
* Neither the Government nor LANS makes any warranty, express or implied, or 
* assumes any liability or responsibility for the use of this software.
*
* Distribution of this source code or of products derived from this
* source code, in part or in whole, including executable files and
* libraries, is expressly forbidden.  
*
* Funding was provided by Laboratory Directed Research and Development.  
*/

#include "DadaParam.h"
#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"

void DadaParam::Export(string fname)
{
	FileStorage fs(fname, FileStorage::WRITE);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem("Could not save meta"));

	//fs << "featureType" << featureType;
	//fs << "featureMergeType" << featureMergeType;
	

	fs.release();

}

void DadaParam::Import(string fname)
{
	FileStorage fs(fname, FileStorage::READ);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem("Could not load meta"));

	//fs["Labels"] >> m_labels;
	
	fs.release();
}


DadaParam::DadaParam() 
	: m_logger(LOG_GET_LOGGER("DadaParam"))
{
	featureMergeType = "max";
	featureWinRadius = 1; 
	numFeatures = 4;
	featureWhiten = 0;
	featureSubsetSize = -1; 
	featureSingle = 0;
	
	classifierInitType = "random";
	classifierNormalizeWeight = 0;
	classifierUpdateType = "svm"; 
	classifierLossType = "std"; 
	finalLossType = "std";
	treeType = "merge"; 
	ensembleType = ""; 
	ensembleSize = 1;
	ensembleDepth = 1;
	errorType = "rand";			// rand, iid

	modelName = "temp";
	featureType = "";
	featureEdgeType = "distance";	// 
	classifierType = "";
	segmentationType = "";
	trainSegmentationType = 0; 
	maxTrainIterations = 1000;
	lrate = 0.01;
	currentIteration = 0;
	printUpdate = 0;
	stepUpdate = 0;
	alpha = 1.0; 
	classifierGradientThreshold = 0;
	slicK = 500; 

	sampleProb = 1.0; 
	posWeight = 1.0; 
	negWeight = 1.0;
	voteType = -1; 
	acdEdgeMultiplier = 1.0;
	acdNumTargetLabels = -1; 
	acdOrigMerge = 1; 
	acdSynMerge = 0; 

	randomSeed = -1; 

}

DadaParam::~DadaParam()
{	
}

