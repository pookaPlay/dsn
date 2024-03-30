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

#include "DadaSegmenter.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaClassifierStump.h"
#include "DadaClassifierLinear.h"
#include "DadaClassifierBFGS.h"
#include "DadaWS.h"
#include "DadaWSUtil.h"
#include "Normalize.h"

void DadaSegmenter::Train(std::shared_ptr<MamaGraph> &gp, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &gt)
{
	LOG_TRACE_METHOD(m_logger, "Train");
	
	m_class->TrainInit(); 
		
	m_class->TrainIteration(*(gp.get()), *(fg.get()), gt);

	while (!m_class->TrainDone()) {					
		m_class->TrainIteration(*(gp.get()), *(fg.get()), gt);
	}
	m_class->TrainFinalize(); 
	m_error = m_class->GetError(); 
	m_outputThreshold = m_classParam->T();
}

void DadaSegmenter::Apply(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg)
{	
	LOG_TRACE_METHOD(m_logger, "Apply");
	// run edge classifier
	MamaGraph &gp = *(myGraph.get());

	m_class->Apply(gp, *(fg.get())); 
	// run connected components
	m_numLabels = DadaWSUtil::ThresholdLabelEdgeWeight(gp);	
}

void DadaSegmenter::Evaluate(std::shared_ptr<MamaGraph> &myGraph, DadaWSGT &gt)
{
	MamaGraph &myG = *(myGraph.get());
	m_class->Evaluate(myG, gt); 
	m_error = m_class->GetError(); 
	//LOG_INFO(m_logger, "*** EVAL here is " << m_error.GetError());
}

void DadaSegmenter::UpdateOutputThreshold(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, double threshold)
{	
	m_classParam->T() = threshold; 
	m_outputThreshold = threshold;
	this->Apply(myGraph, fg); 
}

void DadaSegmenter::SetOutputThreshold(double threshold)
{
	m_classParam->T() = threshold;
	m_outputThreshold = threshold;
}

void DadaSegmenter::TrainInit(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	BOOST_THROW_EXCEPTION(NotImplemented());
}

void DadaSegmenter::TrainStep(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	BOOST_THROW_EXCEPTION(NotImplemented());
}

void DadaSegmenter::TrainFinish(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	BOOST_THROW_EXCEPTION(NotImplemented());
}

void DadaSegmenter::TrainThreshold(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	Train(myGraph, fg, groundTruth); 	
}


std::map<MamaVId, int> & DadaSegmenter::GetLabelMap(int index)
 {
	 LOG_TRACE_METHOD(m_logger, "DadaSegmenter::GetLabelMap " << m_labelMap.size());
	 return(m_labelMap); 
 }

std::map<MamaEId, double> & DadaSegmenter::GetEdgeMap(int index)
{
	return(m_edgeMap);
}

void DadaSegmenter::InitLabelMap(std::shared_ptr<MamaGraph> &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenter::InitLabelMap");

	MamaGraph &gp = *(myGraph.get());
	m_labelMap.clear();
	m_vertexMap.clear(); 

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gp);

	for (nit = nstart; nit != nend; nit++) {		
		m_labelMap[*nit] = gp[*nit].label;
		m_vertexMap[*nit] = *nit; 
	}
}

void DadaSegmenter::InitEdgeMap(std::shared_ptr<MamaGraph> &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenter::InitEdgeMap");

	MamaGraph &gp = *(myGraph.get());
	m_edgeMap.clear();

	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gp);
	for (eit = estart; eit != eend; eit++) {		
		m_edgeMap[*eit] = gp[*eit].weight;
	}
}

void DadaSegmenter::Init(std::shared_ptr<DadaFeatureGenerator> &fg)
{
	LOG_TRACE_METHOD(m_logger, "Init");	
	m_classParam = std::make_shared<DadaClassifierParam>(m_param);
	m_classParam->Init(fg->D());	
	m_class->SetParam(m_classParam);
}

std::shared_ptr< DadaClassifierParam > DadaSegmenter::GetClassParam(int index)
{
	return(m_classParam);
}

std::shared_ptr<DadaFeatureGenerator> DadaSegmenter::GetFeatureGenerator()
{
	return(m_features);
}

void DadaSegmenter::Save()
{
	m_classParam->Save();
}

void DadaSegmenter::Load()
{
	m_classParam->Load();
}

DadaSegmenter::DadaSegmenter(std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaSegmenter")),	
	m_param(param)	
{	
	//cout << "On init I have " << param->classifierType << "\n";
	//cout << "On init I have " << m_param->classifierType << "\n";
	if (m_param->classifierType == string("stump"))
	{
		m_class = make_unique<DadaClassifierStump>(m_param, nullptr);
	}
	else if (m_param->classifierType == string("linear"))
	{
		m_class = make_unique<DadaClassifierLinear>(m_param, nullptr);
	}
	else if (m_param->classifierType == string("bfgs"))
	{
		m_class = make_unique<DadaClassifierBFGS>(m_param, nullptr);
	}
	else {
		m_class = std::make_unique<DadaClassifier>(m_param, nullptr);
	}

}

DadaSegmenter::~DadaSegmenter()
{	
}

