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

#include "DadaClassifierStump.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaEval.h"
#include "DadaIID.h"


void DadaClassifierStump::TrainInit()
{
	m_bestError.GetError() = LARGEST_DOUBLE;
	m_bestError.GetPosError() = 0.0;
	m_bestError.GetNegError() = 0.0;
	m_bestIndex = -1;
	m_bestThreshold = 0.0; 
	m_classParam->I() = -1;
	m_currentIndex = -1; 
	m_sawImprovment = 0;
	m_trainDone = 0;
	m_bestSegType = m_param->segmentationType;
	//LOG_INFO(m_logger, "  Train Init has " << m_classParam->GetValidFeatures().size()); 
}

void DadaClassifierStump::TrainIteration(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &gt)
{	
	// increment stump
	m_currentIndex++; 
	
	if (m_currentIndex >= m_classParam->GetValidFeatures().size()) {
		m_trainDone = 1;
		return;
	}
	
	DadaError error;
	vector<string> segTypes; 
	segTypes.clear(); 

	if (m_param->trainSegmentationType) {
		segTypes.push_back("cc");
		segTypes.push_back("ws");
	}
	else {
		segTypes.push_back(m_param->segmentationType);
	}

	for (auto &it : segTypes) {
		m_classParam->SegType() = it; 

		m_classParam->I() = m_classParam->GetValidFeatures()[m_currentIndex];
		m_classParam->T() = 0.0;

		DadaClassifier::Apply(myGraph, fg);
		m_eval.ComputeMaxMin(myGraph, gt.Labels());

		if (m_param->errorType == "iid") {
			DadaIID::TrainThreshold(m_classParam->T(), myGraph,  gt, error);
		}
		else {
			if (m_param->classifierLossType == "weighted") m_eval.TrainWeightedThreshold(m_classParam->T(), gt, error);
			else m_eval.TrainThreshold(m_classParam->T(), gt, error);
		}
		if (error.GetError() < m_bestError.GetError()) {
			m_bestError = error;
			m_bestThreshold = m_classParam->T();
			m_bestIndex = m_classParam->I();		
			m_bestSegType = m_classParam->SegType();
		}
		//LOG_INFO(m_logger, " S: " << m_classParam->SegType() << " I: " << m_classParam->I() << " T: " << m_classParam->T() << " had E: " << error.GetError() << " (" << error.GetPosError() << "," << error.GetNegError() << ")");
	}
	
}

void DadaClassifierStump::TrainFinalize()
{
	m_classParam->I() = m_bestIndex;
	m_classParam->T() = m_bestThreshold;	
	m_classParam->SegType() = m_bestSegType; 
	//m_classParam->T() = 20.0;
	m_error = m_bestError; 
	//LOG_INFO(m_logger, " S: " << m_bestSegType << "  I: " << m_classParam->I() << " T: " << m_classParam->T() << " with E: " << m_error.GetError() << "(" << m_error.GetPosError() << "," << m_error.GetNegError() << ")");
}

double DadaClassifierStump::ApplyNoThreshold(MamaEId &mid, DadaFeatureGenerator &fg)
{	
	return(fg.GetEdgeFeature(mid, m_classParam->I())); 
}

double DadaClassifierStump::ApplyThreshold(double &feature)
{
	return(feature - m_classParam->T());
}


DadaClassifierStump::DadaClassifierStump(std::shared_ptr<DadaParam> param, std::shared_ptr<DadaClassifierParam> classParam)
	: DadaClassifier(param, classParam)
{	
}

DadaClassifierStump::~DadaClassifierStump()
{	
}

