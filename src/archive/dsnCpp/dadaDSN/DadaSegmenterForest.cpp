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

#include "DadaSegmenterForest.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaWS.h"
#include "Normalize.h"
#include "DadaWSUtil.h"
#include <random>
#include <algorithm>
#include "DadaWSACD.h"

void DadaSegmenterForest::Apply(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg)
{
	LOG_TRACE_METHOD(m_logger, "Apply");	

	m_treeLabels.clear(); 
	m_treeLabels.resize(m_forest->NumTrees());

	for (int i = 0; i < m_treeLabels.size(); i++) m_treeLabels[i].clear(); 

	for (int i = 0; i < m_forest->NumTrees(); i++) {		
		//LOG_INFO(m_logger, "=== Apply Tree " << i);

		DadaSegmenterTree::SetTreeParam(m_forest->GetTree(i)); 
		DadaSegmenterTree::Apply(myGraph, fg); 
		DadaSegmenterTree::InitLabelMap(myGraph); 
		m_treeLabels[i] = DadaSegmenterTree::GetLabelMap();			
	}

	//LOG_INFO(m_logger, "=== Apply Vote");
	this->ApplyVote(myGraph); 
}

void DadaSegmenterForest::ApplyVote(std::shared_ptr<MamaGraph> &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "ApplyVote");
	
	MamaGraph &gp = *(myGraph.get());

	this->GetVoteData(myGraph);
	
	m_vote->SetParam(m_forest->GetVote()); 
	m_vote->DadaClassifier::Apply(gp, m_voteData); 
	
	m_outputThreshold = m_forest->GetVote()->T();
	//LOG_INFO(m_logger, "ApplyVote threshold is " << m_outputThreshold);

	DadaWSUtil::ThresholdLabelEdgeWeight(gp);	
	
}

void DadaSegmenterForest::Evaluate(std::shared_ptr<MamaGraph> &myGraph, DadaWSGT &gt)
{
	MamaGraph &myG = *(myGraph.get());
	m_vote->Evaluate(myG, gt);
	m_error = m_vote->GetError();
	//LOG_INFO(m_logger, "*** EVAL here is " << m_error.GetError());
}

void DadaSegmenterForest::SetOutputThreshold(double threshold)
{
	LOG_TRACE_METHOD(m_logger, "SetOutputThreshold");	
	m_forest->GetVote()->T() = threshold;
}

void DadaSegmenterForest::UpdateOutputThreshold(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, double threshold)
{
	LOG_TRACE_METHOD(m_logger, "UpdateOutputThreshold");

	MamaGraph &gp = *(myGraph.get());
	m_forest->GetVote()->T() = threshold; 
	m_vote->SetParam(m_forest->GetVote());
	m_vote->DadaClassifier::Apply(gp, m_voteData);
	
	m_outputThreshold = m_forest->GetVote()->T();
	//LOG_INFO(m_logger, "Update threshold is " << m_outputThreshold);
	DadaWSUtil::ThresholdLabelEdgeWeight(gp);
}

void DadaSegmenterForest::GetVoteData(std::shared_ptr<MamaGraph> &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "GetVoteData");

	m_voteData.Clear(); 
	m_voteData.InitSingle(); 

	MamaGraph &gp = *(myGraph.get());	
	
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gp);

	m_outputMin = LARGEST_DOUBLE;
	m_outputMax = SMALLEST_DOUBLE;
	
	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, gp);
		MamaVId id2 = target(*eit, gp);

		double diffCount = 0.0;
		for (int i = 0; i < m_treeLabels.size(); i++) {
			//if (m_treeIndex == i) {
				if (m_treeLabels[i][id1] != m_treeLabels[i][id2]) {
					diffCount += 1.0;
				}
			//}
		}
		m_voteData.InitSingleEdgeScalar(*eit, diffCount); 
		m_outputMin = min(m_outputMin, diffCount);
		m_outputMax = max(m_outputMax, diffCount);
	}
}


void DadaSegmenterForest::InitLabelMap(std::shared_ptr<MamaGraph> &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenterForest::InitLabelMap");

	MamaGraph &gp = *(myGraph.get());
	m_labelMap.clear();

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gp);

	for (nit = nstart; nit != nend; nit++) {
		m_labelMap[*nit] = gp[*nit].label;
	}
}

std::map<MamaVId, int> & DadaSegmenterForest::GetLabelMap(int index)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenterForest::GetLabelMap");
	if (index < 0) 	return(m_labelMap);
	if (index < m_treeLabels.size()) return(m_treeLabels[index]);
	else return(m_labelMap);
}

void DadaSegmenterForest::Train(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenterForest Train");
	
	m_treeError.GetError() = LARGEST_DOUBLE; 	
	m_treeIndex = -1; 

	m_treeLabels.clear();
	m_treeLabels.resize(m_forest->NumTrees());
	m_treeErrors.clear(); 
	m_treeErrors.resize(m_forest->NumTrees());
	
	for (int i = 0; i < m_forest->NumTrees(); i++) {
		LOG_INFO(m_logger, "####### Tree " << i << "########"); 
		
		DadaSegmenterTree::SetTreeParam(m_forest->GetTree(i)); 		

		//DadaWSGT sample; 
		//this->SubSample(groundTruth, sample);
		//if (m_acd) {
		//	m_acd->PickNewEdges();
		//}

		DadaSegmenterTree::Train(myGraph, fg, groundTruth); 
		m_treeErrors[i] = m_error; 

		DadaSegmenterTree::InitLabelMap(myGraph); 
		//cout << "Assigning to " << m_treeLabels.size() << "\n"; 
		m_treeLabels[i] = DadaSegmenter::GetLabelMap();

		if (m_param->featureSingle) {
			Mat myResult;
			m_dada->GetImgLabels(myResult, i);
			VizMat::DisplayColorSeg(myResult, "s", 0, 1.0);
		}
		// may need to apply at some point
		if (m_error.GetError() < m_treeError.GetError()) {
			m_treeError = m_error;
			m_treeIndex = i; 
		}
		//cout << "Assigned label map\n";
	}
	// Now combine
	LOG_INFO(m_logger, "############ VOTE #############");
	
	this->TrainVote(myGraph, groundTruth);	

	LOG_INFO(m_logger, "### E: " << m_error.GetError() << " (" << m_treeError.GetError() << ")");
}

void DadaSegmenterForest::TrainInit(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	m_treeIndex = 0; 
}

void DadaSegmenterForest::TrainStep(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	if (m_treeIndex < m_forest->NumTrees()) {
		LOG_INFO(m_logger, "####### Tree " << m_treeIndex << "########");
		DadaSegmenterTree::SetTreeParam(m_forest->GetTree(m_treeIndex));
		DadaSegmenterTree::Train(myGraph, fg, groundTruth);
		m_treeIndex++;
		//m_treeLabels[i] = DadaSegmenter::GetLabelMap();
		//Mat myResult;
		//m_dada->GetImgLabels(myResult, i);
		//VizMat::DisplayColorSeg(myResult, "s", 0, 1.0);		

	}
}

void DadaSegmenterForest::TrainFinish(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{	
	m_vote->SetParam(m_forest->GetVote());
	m_forest->GetVote()->I() = 0;
	m_forest->GetVote()->T() = 0.0; 
	m_outputThreshold = 0.0; 
}

void DadaSegmenterForest::TrainThreshold(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	this->TrainVote(myGraph, groundTruth); 	
}


void DadaSegmenterForest::SubSample(DadaWSGT &gt, DadaWSGT &sample)
{
	
	sample.Labels() = gt.Labels(); 
	sample.ExtraPos() = 0.0;
	sample.ExtraNeg() = 0.0;
	sample.ErrorPos() = 0.0;
	sample.ErrorNeg() = 0.0;
	sample.PosWeight() = gt.PosWeight();
	sample.NegWeight() = gt.NegWeight();

	for (auto &nit : gt.Labels()) {
		double rf = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
		if (rf <= m_param->sampleProb) {
			sample.ExtraPos() += gt.VExtraPos()[nit.first];
			sample.ExtraNeg() += gt.VExtraNeg()[nit.first];
			sample.ErrorPos() += 0.0;
			sample.ErrorNeg() += gt.VErrorNeg()[nit.first];
		}
		else {
			sample.Labels()[nit.first].clear(); 
		}
	}	
}

void DadaSegmenterForest::TrainVote(std::shared_ptr<MamaGraph> &myGraph, DadaWSGT &groundTruth)
{
	LOG_TRACE_METHOD(m_logger, "TrainVote");
	
	MamaGraph &gp = *(myGraph.get());

	this->GetVoteData(myGraph);
	
	if (m_param->voteType < 0) {
		string temps = m_param->classifierLossType;
		m_param->classifierLossType = "std"; // m_param->finalLossType;

		m_vote->SetParam(m_forest->GetVote());
		m_vote->TrainInit();
		m_vote->TrainIteration(gp, m_voteData, groundTruth);
		m_vote->TrainFinalize();
		m_error = m_vote->GetError();
		m_outputThreshold = m_forest->GetVote()->T();
		m_param->classifierLossType = temps;
	}
	else {
		m_vote->SetParam(m_forest->GetVote());
		m_forest->GetVote()->I() = 0; 
		m_forest->GetVote()->T() = m_param->voteType;
		m_outputThreshold = m_param->voteType;
	}

	m_vote->DadaClassifier::Apply(gp, m_voteData);
	
	DadaWSUtil::ThresholdLabelEdgeWeight(gp);

}

void DadaSegmenterForest::Init(std::shared_ptr<DadaFeatureGenerator> &fg)
{
	LOG_TRACE_METHOD(m_logger, "Init with " << m_param->ensembleSize);	
	m_forest = std::make_shared<DadaSegmenterForestParam>(m_param);
	m_forest->Init();
}

void DadaSegmenterForest::Save()
{
	m_forest->Save(); 
}

void DadaSegmenterForest::Load()
{
	m_forest->Load();
}

DadaSegmenterForest::DadaSegmenterForest(std::shared_ptr<DadaParam> &param)
	: DadaSegmenterTree(param),
	m_voteData(param),
	m_vote(make_unique<DadaClassifierStump>(param, nullptr))	
{	
	m_logger.getInstance("DadaSegmenterForest");
	m_acd = 0; 
}

DadaSegmenterForest::~DadaSegmenterForest()
{	
}

