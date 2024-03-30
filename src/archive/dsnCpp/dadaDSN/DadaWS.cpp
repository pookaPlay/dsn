#include "DadaWS.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaSegmenterForest.h"
#include <fstream>
#include <boost/graph/graphml.hpp>
#include "DadaFeatureGenerator.h"

void DadaWS::InitSegmenter()
{
	LOG_INFO(m_logger, "InitSegmenter with D: " << m_features->D());
	//LOG_TRACE_METHOD(m_logger, "InitSegmenter with D: " << m_features->D());
	
	if (m_param->ensembleType == string("forest")) {
		m_seg = make_unique<DadaSegmenterForest>(m_param);		
	}
	else if (m_param->ensembleType == string("tree")) {
		m_seg = make_unique<DadaSegmenterTree>(m_param);
	}
	else {
		m_seg = make_unique<DadaSegmenter>(m_param);
	} 
	m_seg->SetDada(this);
	m_seg->Init(m_features);
}

void DadaWS::TrainSegmenter()
{
	LOG_TRACE_METHOD(m_logger, "TrainSegmenter"); 
	m_seg->Train(m_myGraph, m_features, m_gt);
}

void DadaWS::TrainInit()
{
	m_seg->TrainInit(m_myGraph, m_features, m_gt);
}

void DadaWS::TrainStep()
{
	m_seg->TrainStep(m_myGraph, m_features, m_gt);
}

void DadaWS::TrainFinish()
{
	m_seg->TrainFinish(m_myGraph, m_features, m_gt);
}

void DadaWS::TrainThreshold()
{
	m_seg->TrainThreshold(m_myGraph, m_features, m_gt);
}

void DadaWS::ApplySegmenter()
{
	LOG_TRACE_METHOD(m_logger, "ApplySegmenter");
	m_seg->Apply(m_myGraph, m_features);
	m_seg->InitLabelMap(m_myGraph);	
}

void DadaWS::UpdateOutputThreshold(double threshold)
{
	LOG_TRACE_METHOD(m_logger, "UpdateOutputThreshold");
	m_seg->UpdateOutputThreshold(m_myGraph, m_features, threshold); 
	m_seg->InitLabelMap(m_myGraph);
}

void DadaWS::SetOutputThreshold(double threshold)
{
	m_seg->SetOutputThreshold(threshold);
}

DadaError & DadaWS::EvalSegmenterMST()
{
	m_seg->Evaluate(m_myGraph, m_gt); 
	return(m_seg->GetError()); 
}

DadaError DadaWS::EvalSegmenterMat()
{
	LOG_TRACE_METHOD(m_logger, "EvalSegmenter");
	Mat mySeg; 
	this->GetImgLabels(mySeg);
		
	DadaError err;
	DadaEval::MatRandError(mySeg, m_desired, err);
	
	return(err); 
}

void DadaWS::Save()
{
	LOG_TRACE_METHOD(m_logger, "Save");
	m_seg->Save(); 
}

void DadaWS::Load()
{
	LOG_TRACE_METHOD(m_logger, "Load");
	m_seg->Load(); 
}

void DadaWS::UpdateTrainViz()
{
	Mat myResult;
	this->GetImgLabels(myResult);	
	VizMat::DisplayColorSeg(myResult, "result", 1, 0.5);
}

void DadaWS::GetImgLabels(cv::Mat &mySeg, int index)
{
	std::map<MamaVId, int> & labelMap = m_seg->GetLabelMap(index);

	mySeg = Mat::zeros(m_h, m_w, CV_32F);
	for (int j = 0; j < m_h; j++) {
		for (int i = 0; i < m_w; i++) {
			int myl = static_cast<int>(m_basins.at<float>(j, i));
			MamaVId nid = m_basinVMap->operator[](myl);
			int mynl = labelMap[nid];			
			mySeg.at<float>(j, i) = static_cast<float>(mynl); 
		}
	}	
}


void DadaWS::GetImgEdges(cv::Mat &mySeg, int index)
{
	m_seg->InitEdgeMap(m_myGraph);
	std::map<MamaEId, double> & edgeMap = m_seg->GetEdgeMap(index);
	
	MamaGraph &gp = *(m_myGraph.get());
	double lowestVal = LARGEST_DOUBLE; 

	mySeg = Mat::zeros(m_h, m_w, CV_32F);
	for (int j = 0; j < (m_h-1); j++) {
		for (int i = 0; i < (m_w-1); i++) {
			int myl = static_cast<int>(m_basins.at<float>(j, i));
			int myrl = static_cast<int>(m_basins.at<float>(j,i+1));
			int mybl = static_cast<int>(m_basins.at<float>(j+1,i));

			if (myl != myrl) {
				MamaVId nid1 = m_basinVMap->operator[](myl);
				MamaVId nid2 = m_basinVMap->operator[](myrl);
				if (!(boost::edge(nid1, nid2, gp).second)) BOOST_THROW_EXCEPTION(Unexpected());
				MamaEId eid = boost::edge(nid1, nid2, gp).first;
				if (!edgeMap.count(eid)) BOOST_THROW_EXCEPTION(Unexpected());
				if (edgeMap[eid] < lowestVal) lowestVal = edgeMap[eid];
			}
			if (myl != mybl) {
				MamaVId nid1 = m_basinVMap->operator[](myl);
				MamaVId nid2 = m_basinVMap->operator[](mybl);
				if (!(boost::edge(nid1, nid2, gp).second)) BOOST_THROW_EXCEPTION(Unexpected());
				MamaEId eid = boost::edge(nid1, nid2, gp).first;
				if (!edgeMap.count(eid)) BOOST_THROW_EXCEPTION(Unexpected());
				if (edgeMap[eid] < lowestVal) lowestVal = edgeMap[eid];
			}
			
		}
	}

	for (int j = 0; j < (m_h - 1); j++) {
		for (int i = 0; i < (m_w - 1); i++) {
			int myl = static_cast<int>(m_basins.at<float>(j, i));
			int myrl = static_cast<int>(m_basins.at<float>(j, i + 1));
			int mybl = static_cast<int>(m_basins.at<float>(j + 1, i));
			double val = lowestVal;

			if (myl != myrl) {
				MamaVId nid1 = m_basinVMap->operator[](myl);
				MamaVId nid2 = m_basinVMap->operator[](myrl);
				if (!(boost::edge(nid1, nid2, gp).second)) BOOST_THROW_EXCEPTION(Unexpected());
				MamaEId eid = boost::edge(nid1, nid2, gp).first;
				if (!edgeMap.count(eid)) BOOST_THROW_EXCEPTION(Unexpected());
				if (edgeMap[eid] > val) val = edgeMap[eid];
			}
			if (myl != mybl) {
				MamaVId nid1 = m_basinVMap->operator[](myl);
				MamaVId nid2 = m_basinVMap->operator[](mybl);
				if (!(boost::edge(nid1, nid2, gp).second)) BOOST_THROW_EXCEPTION(Unexpected());
				MamaEId eid = boost::edge(nid1, nid2, gp).first;
				if (!edgeMap.count(eid)) BOOST_THROW_EXCEPTION(Unexpected());
				if (edgeMap[eid] > val) val = edgeMap[eid];
			}

			mySeg.at<float>(j, i) = static_cast<float>(val);
		}
	}

}

void DadaWS::InitFromBasins(cv::Mat &basins)
{	
	LOG_TRACE_METHOD(m_logger, "InitFromBasins");
	if (basins.type() != CV_32F) BOOST_THROW_EXCEPTION(UnexpectedType());
		
	m_myGraph = std::make_shared<MamaGraph>(); 
	m_myGraph->clear();	

	MamaGraph &gp = *(m_myGraph.get()); 

	m_basinLabel.clear();
	m_basinVMap = std::make_shared< std::map<int, MamaVId> >(); 
	m_basinVMap->clear();

	m_basins = basins; 
	m_w = basins.cols;
	m_h = basins.rows;

	int labelCount = 0;
	for (int j = 0; j < m_h; j++) {
		for (int i = 0; i < m_w; i++) {
			int myl = static_cast<int>(basins.at<float>(j, i));

			if (!m_basinVMap->count(myl)) {
				m_basinVMap->operator[](myl) = add_vertex(gp);
				m_basinLabel[m_basinVMap->operator[](myl)] = myl;
			}
		}
	}

	// Add edges 
	for (int j = 0; j < m_h; j++) {
		for (int i = 0; i < m_w; i++) {
			int myl = static_cast<int>(basins.at<float>(j, i));
			if (i > 0) {
				int neighl = static_cast<int>(basins.at<float>(j, i - 1));
				if (neighl != myl) {
					add_edge(m_basinVMap->operator[](myl), m_basinVMap->operator[](neighl), gp);
				}
			}
			if (j > 0) {
				int neighl = static_cast<int>(basins.at<float>(j - 1, i));
				if (neighl != myl) {
					add_edge(m_basinVMap->operator[](myl), m_basinVMap->operator[](neighl), gp);
				}
			}
		}
	}
	//LOG_INFO(m_logger, "Graph V:" << num_vertices(gp) << " and E:" << num_edges(gp)); 
}

void DadaWS::InitFeatures(std::vector<cv::Mat> &imgs, int trainMode)
{
	LOG_TRACE_METHOD(m_logger, "InitFeatures");
	m_features = std::make_shared<DadaFeatureGenerator>(m_param);
	m_features->InitFromBasins(m_basins, m_basinVMap, m_myGraph);
	m_features->CalculateBaseFeatures(imgs, trainMode);
}

void DadaWS::InitFeatures(cv::Mat &img, int trainMode)
{
	vector<Mat> tempvm(1);
	tempvm[0] = img;
	this->InitFeatures(tempvm, trainMode);
}

void DadaWS::VizNodeFeatures()
{
	m_seg->GetFeatureGenerator()->VizNodeFeatures(*(m_myGraph.get()), m_seg->GetVertexMap());
}

void DadaWS::InitGroundTruth(cv::Mat &desiredSeg)
{
	LOG_TRACE_METHOD(m_logger, "InitGroundTruth");
	if (desiredSeg.rows != m_basins.rows) BOOST_THROW_EXCEPTION(UnexpectedSize()); 
	if (desiredSeg.cols != m_basins.cols) BOOST_THROW_EXCEPTION(UnexpectedSize());
	
	this->m_desired = desiredSeg; 

	this->m_gt.Clear(); 

	int labelCount = 0;

	for (int j = 0; j < m_h; j++) {
		for (int i = 0; i < m_w; i++) {
			int vali = static_cast<int>(m_basins.at<float>(j, i));
			int gti = static_cast<int>(desiredSeg.at<float>(j, i));
			if (gti >= 0) {
				// get myGraph vertex
				MamaVId gid = m_basinVMap->operator[](vali);
				if (m_gt.Labels().count(gid) == 0) {
					m_gt.Labels()[gid].clear();
				}

				if (m_gt.Labels()[gid].count(gti) == 0) {
					m_gt.Labels()[gid][gti] = 0.0;
				}
				m_gt.Labels()[gid][gti] += 1.0;
				labelCount++;
			}
		}
	}

	// get error already iccured at lower level		

	for (auto &nit : m_gt.Labels()) {

		double tsum = 0.0;
		double tssum = 0.0;
		
		for (auto &it : m_gt.Labels()[nit.first]) {
			tssum += (it.second * it.second - it.second) / 2.0;
			tsum += it.second;
		}
		double myCount = (tsum*tsum - tsum) / 2.0;
		m_gt.VExtraPos()[nit.first] = tssum;
		m_gt.VExtraNeg()[nit.first] = (myCount - tssum);
		m_gt.VErrorPos()[nit.first] = 0.0; 
		m_gt.VErrorNeg()[nit.first] = (myCount - tssum);		// its a merge to start

		m_gt.ExtraPos() += tssum;
		m_gt.ExtraNeg() += (myCount - tssum);
		m_gt.ErrorPos() += 0.0;
		m_gt.ErrorNeg() += (myCount - tssum);

	}

	m_gt.PosWeight() = m_param->posWeight; 
	m_gt.NegWeight() = m_param->negWeight;

	//LOG_INFO(m_logger, "GT " << labelCount << " labels (" << m_gt.PosWeight() << "/" << m_gt.NegWeight() << ") : " << m_gt.ExtraPos() << ", N : " << m_gt.ExtraNeg() << " extras");

}


DadaWS::DadaWS(std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaWS")),
	  m_param(param)	  
{			
	m_myGraph = nullptr; 
	m_basinLabel.clear();
	m_features = nullptr; 
	m_basinVMap = nullptr;
}

DadaWS::~DadaWS()
{	
}
