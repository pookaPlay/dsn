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

#include "DadaFeatureGenerator.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"

#include "DadaFeatures.h"
#include "DadaFeaturesEdges.h"
#include "DadaFeaturesVecEdges.h"
#include "DadaFeaturesEdge.h"
#include "DadaFeaturesKMeans.h"
#include "DadaFeaturesMoments.h"
#include "DadaFeaturesHistogram.h"

double DadaFeatureGenerator::GetEdgeFeature(MamaEId &eid, int ind)
{		
	if (ind >= m_featureIndex.size()) BOOST_THROW_EXCEPTION(UnexpectedSize("ind not in feature index"));
	if (ind >= m_featureSubIndex.size()) BOOST_THROW_EXCEPTION(UnexpectedSize("ind not in feature sub index"));
	int fi = m_featureIndex[ind];
	if (fi >= m_features.size()) BOOST_THROW_EXCEPTION(UnexpectedSize("no feature object at index"));
		
	std::map<MamaEId, cv::Mat> &ef = m_features[fi]->GetEFeatures();
	if (!ef.count(eid)) BOOST_THROW_EXCEPTION(Unexpected("Edge not in feature map"));
	int si = m_featureSubIndex[ind];
	if (si >= ef[eid].cols) BOOST_THROW_EXCEPTION(UnexpectedSize("sub index too big"));
	return(ef[eid].at<double>(si)); 
}

void DadaFeatureGenerator::InitFromBasins(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<MamaGraph> &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "InitFromBasins");

	m_basins = basins;
	m_labelMap = myVId;
	m_baseGraph = myGraph;
	m_features.clear();	

	boost::char_separator<char> sep(",");
	StringTokenizer tok(m_param->featureType, sep);

	for (StringTokenizer::iterator it = tok.begin(); it != tok.end(); ++it) {
		string featureType = *it;
		//LOG_INFO(m_logger, " Creating base feature type: " << featureType);
		
		m_features.push_back(CreateFeature(featureType, 1));
		
	}
}

void DadaFeatureGenerator::CalculateBaseFeatures(std::vector<cv::Mat> &imgs, int trainMode)
{	
	LOG_TRACE_METHOD(m_logger, "CalculateBaseFeatures"); 
	m_D = 0; 
	m_featureIndex.clear(); 
	m_featureSubIndex.clear();
	for (int i = 0; i < m_features.size(); i++) {
		
		// generate feature images 
		m_features[i]->GenerateFeatures(imgs, trainMode);
		
		// pool feature images 
		m_features[i]->PoolFeatures(*(m_baseGraph.get()));

		//LOG_INFO(m_logger, " Generated " << m_features[i]->D() << " features for " << m_features[i]->GetType());

		for (int fi = 0; fi < m_features[i]->D(); fi++) {
			m_featureIndex.push_back(i); 
			m_featureSubIndex.push_back(fi);
			m_D++;
		}		
	}	
}


void DadaFeatureGenerator::CalculateMergeFeatures(DadaFeatureGenerator &fp, MamaGraph &gp, MamaGraph &gc,
												  std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge)
{
	LOG_TRACE_METHOD(m_logger, "CalculateMergeFeatures");
	this->Clear(); 
	// propagate feature types from parent
	for (int fi = 0; fi < fp.GetFeatureProcessors().size(); fi++) {
		
		m_features.push_back(CreateFeature(fp.GetFeatureProcessor(fi)->GetType(), 0)); 

		m_features[m_features.size() - 1]->MergeFeatures(fp.GetFeatureProcessor(fi), gp, gc, labelChild, childParentEdge);

		LOG_TRACE(m_logger, " Generated " << m_features[m_features.size() - 1]->D() << " features for " << m_features[m_features.size() - 1]->GetType());

		for (int di = 0; di < m_features[m_features.size() - 1]->D(); di++) {
			m_featureIndex.push_back(fi);
			m_featureSubIndex.push_back(di);
			m_D++;
		}

	}
}



void DadaFeatureGenerator::CalculateSplitFeatures(DadaFeatureGenerator &fgp, MamaGraph &gp, MamaGraph &gc, std::map<MamaVId, MamaVId> &parentChildVertex)
{
	LOG_TRACE_METHOD(m_logger, "CalculateMergeFeatures");
	this->Clear();
	// propagate feature types from parent
	for (int fi = 0; fi < fgp.GetFeatureProcessors().size(); fi++) {

		m_features.push_back(CreateFeature(fgp.GetFeatureProcessor(fi)->GetType(), 0));

		m_features[m_features.size() - 1]->SplitFeatures(fgp.GetFeatureProcessor(fi), gp, gc, parentChildVertex);

		LOG_TRACE(m_logger, " Generated " << m_features[m_features.size() - 1]->D() << " features for " << m_features[m_features.size() - 1]->GetType());

		for (int di = 0; di < m_features[m_features.size() - 1]->D(); di++) {
			m_featureIndex.push_back(fi);
			m_featureSubIndex.push_back(di);
			m_D++;
		}

	}
	//this->VizEdgeFeatures(gc, 0, 1.0); 
}
void DadaFeatureGenerator::VizFeatureImages(int stepLast, double mag)
{
	for (int i = 0; i < m_features.size(); i++) {
		m_features[i]->VizFeatureImages(stepLast, mag); 
	}
}

void DadaFeatureGenerator::VizNodeFeatures(MamaGraph &gp, std::map<MamaVId, MamaVId> &vmap)
{
	for (int fi = 0; fi < m_features.size(); fi++) {
		std::map<MamaVId, cv::Mat> &nodef = m_features[fi]->GetVFeatures();
		int myD = nodef.begin()->second.cols;		
		
		for (int f = 0; f < myD; f++) {
			LOG_INFO(m_logger, "Feature " << f << "/" << myD << " for " << fi << "/" << m_features.size());

			Mat imgf = Mat::zeros(m_basins.rows, m_basins.cols, CV_32F);
			for (int j = 0; j < imgf.rows; j++) {
				for (int i = 0; i < imgf.cols; i++) {
					int myl = static_cast<int>(m_basins.at<float>(j, i));
					MamaVId oid = static_cast<MamaVId>(myl); 
					if (!vmap.count(oid)) BOOST_THROW_EXCEPTION(Unexpected("could not find vertex in node viz"));
					MamaVId nid = vmap[oid];
					double val = nodef[nid].at<double>(f);
					imgf.at<float>(j, i) = static_cast<float>(val);
				}
			}
			VizMat::DisplayFloat(imgf, "f", 0, 1.0);
		}
	}
}


void DadaFeatureGenerator::VizEdgeFeatures(MamaGraph &gp, int stepLast, double mag)
{
	for (int fi = 0; fi < m_features.size(); fi++) {

		std::map<MamaEId, cv::Mat> &edgef = m_features[fi]->GetEFeatures();
		int myD = edgef.begin()->second.cols;


		int kb;
		for (int f = 0; f < myD; f++) {
			LOG_INFO(m_logger, "Feature " << f << "/" << myD << " for " << fi << "/" << m_features.size());
			if (stepLast) {
				if (f < myD - 1) kb = 1;
				else kb = 0;
			}
			else kb = 0;
			Mat imgf = Mat::zeros(m_basins.rows, m_basins.cols, CV_32F);

			for (int j = 0; j < (imgf.rows - 1); j++) {
				for (int i = 0; i < (imgf.cols - 1); i++) {
					int myl = static_cast<int>(m_basins.at<float>(j, i));
					int myrl = static_cast<int>(m_basins.at<float>(j, i + 1));
					int mybl = static_cast<int>(m_basins.at<float>(j + 1, i));
					double val = 0.0;

					if (myl != myrl) {
						MamaVId nid1 = m_labelMap->operator[](myl);
						MamaVId nid2 = m_labelMap->operator[](myrl);
						if (!(boost::edge(nid1, nid2, gp).second)) BOOST_THROW_EXCEPTION(Unexpected());
						MamaEId eid = boost::edge(nid1, nid2, gp).first;
						if (!edgef.count(eid)) BOOST_THROW_EXCEPTION(Unexpected());
						if (edgef[eid].at<double>(f) > val) val = edgef[eid].at<double>(f);
					}
					if (myl != mybl) {
						MamaVId nid1 = m_labelMap->operator[](myl);
						MamaVId nid2 = m_labelMap->operator[](mybl);
						if (!(boost::edge(nid1, nid2, gp).second)) BOOST_THROW_EXCEPTION(Unexpected());
						MamaEId eid = boost::edge(nid1, nid2, gp).first;
						if (!edgef.count(eid)) BOOST_THROW_EXCEPTION(Unexpected());
						if (edgef[eid].at<double>(f) > val) val = edgef[eid].at<double>(f);
					}

					imgf.at<float>(j, i) = static_cast<float>(val);
				}
			}
			VizMat::DisplayFloat(imgf, "f", kb, static_cast<float>(mag));
		}
	}
}


void DadaFeatureGenerator::InitSingle()
{
	this->Clear(); 
	m_D = 1;
	m_featureIndex.push_back(0); 
	m_featureSubIndex.push_back(0); 
	m_features.resize(1); 
	m_features[0] = std::make_shared<DadaFeatures>(m_basins, m_labelMap, m_param);
}

void DadaFeatureGenerator::InitSingleEdgeScalar(MamaEId &eid, double value)
{
	m_features[0]->GetEFeatures()[eid] = Mat::ones(1, 1, CV_64F) * value; 	
}

std::shared_ptr<DadaFeatures> DadaFeatureGenerator::CreateFeature(string featureType, int createBase)
{			
	if (featureType == string("edge"))
	{
		if (createBase)
			return(std::make_shared<DadaFeaturesEdge>(m_basins, m_labelMap, m_param));
		else 
			return(std::make_shared<DadaFeaturesEdge>(m_param));
	}
	else if (featureType == string("edges"))
	{		
		if (createBase)
			return(std::make_shared<DadaFeaturesEdges>(m_basins, m_labelMap, m_param));
		else 
			return(std::make_shared<DadaFeaturesEdges>(m_param));
	}
	else if (featureType == string("vecEdges"))
	{
		if (createBase)
			return(std::make_shared<DadaFeaturesVecEdges>(m_basins, m_labelMap, m_param));
		else
			return(std::make_shared<DadaFeaturesVecEdges>(m_param));
	}
	else if (featureType == string("kmeans"))
	{
		if (createBase)
			return(std::make_shared<DadaFeaturesKMeans>(m_basins, m_labelMap, m_param));
		else
			return(std::make_shared<DadaFeaturesKMeans>(m_param));
	}
	else if (featureType == string("moments"))
	{
		if (createBase)
			return(std::make_shared<DadaFeaturesMoments>(m_basins, m_labelMap, m_param));
		else 
			return(std::make_shared<DadaFeaturesMoments>(m_param));
	}
	else if (featureType == string("histogram"))
	{		
		if (createBase)
			return(std::make_shared<DadaFeaturesHistogram>(m_basins, m_labelMap, m_param));
		else 
			return(std::make_shared<DadaFeaturesHistogram>(m_param));
	}
	else {
		if (createBase)
			return(std::make_shared<DadaFeatures>(m_basins, m_labelMap, m_param));
		else 
			return(std::make_shared<DadaFeatures>(m_param));
	}
}

void DadaFeatureGenerator::Clear()
{
	m_features.clear();
	m_D = 0;
	m_featureIndex.clear();
	m_featureSubIndex.clear();	
}

DadaFeatureGenerator::DadaFeatureGenerator(std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaFeatureGenerator")),
	  m_param(param)
{	
	Clear(); 
}

DadaFeatureGenerator::~DadaFeatureGenerator()
{	
}

