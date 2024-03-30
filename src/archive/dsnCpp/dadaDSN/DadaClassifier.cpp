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

#include "DadaClassifier.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaEval.h"
#include "DadaWSUtil.h"

void DadaClassifier::Apply(MamaGraph &myGraph, DadaFeatureGenerator &fg)
{
	LOG_TRACE_METHOD(m_logger, "Apply to graph");	
	
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(myGraph);
	
	for (eit = estart; eit != eend; eit++) {	
		myGraph[*eit].weight = this->ApplyNoThreshold(*eit, fg);
	}
	
	// Do connected components		
	if (m_classParam->SegType() == string("cc")) {
		for (eit = estart; eit != eend; eit++) {
			myGraph[*eit].weight = this->ApplyThreshold(myGraph[*eit].weight);
		}
	} 
	
	// Do watershed 
	else if (m_classParam->SegType() == string("ws")) {
		
		DadaWSUtil::ApplyWatershed(myGraph, m_watershedNeighbors); 		

		for (eit = estart; eit != eend; eit++) {
			myGraph[*eit].weight = this->ApplyThreshold(myGraph[*eit].weight);
		}
	}
	else BOOST_THROW_EXCEPTION(UnexpectedType()); 
}

void DadaClassifier::Evaluate(MamaGraph &myGraph, DadaWSGT &gt)
{
	m_eval.ComputeMaxMin(myGraph, gt.Labels());
	m_eval.RandMSTError(gt, m_error); 
	//LOG_INFO(m_logger, "*** EVAL here is " << m_error.GetError()); 
}

double DadaClassifier::ApplyNoThreshold(MamaEId &mid, DadaFeatureGenerator &fg)
{	
	return(fg.GetEdgeFeature(mid, 0)); 
}

double DadaClassifier::ApplyThreshold(double &feature)
{
	return(feature); 
}

DadaClassifier::DadaClassifier(std::shared_ptr<DadaParam> param, std::shared_ptr<DadaClassifierParam> classParam)
	: m_logger(LOG_GET_LOGGER("DadaClassifier")),
	m_param(param),
	m_classParam(classParam),
	m_sawImprovment(0),
	m_trainDone(0)
{	
}

DadaClassifier::~DadaClassifier()
{	
}

