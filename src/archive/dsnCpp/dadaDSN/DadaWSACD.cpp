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

#include "DadaWSACD.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include <fstream>
#include "DadaWSUtil.h"

void DadaWSACD::InitACDApply()
{
	LOG_TRACE_METHOD(m_logger, "InitACDApply");

	MamaGraph &gp = *(m_myGraph.get());
	LOG_INFO(m_logger, "On Apply V:" << num_vertices(gp) << " and E:" << num_edges(gp));
	/*
//	std::map<MamaEId, cv::Mat> &eFeatures = m_features->GetE();
//	std::map<MamaVId, cv::Mat> &vFeatures = m_features->GetV();

	eFeatures.clear();
	//Mat v1, v2;
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gp);

	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, gp);
		MamaVId id2 = target(*eit, gp);

		if (!vFeatures.count(id1)) BOOST_THROW_EXCEPTION(Unexpected());
		if (!vFeatures.count(id2)) BOOST_THROW_EXCEPTION(Unexpected());

		Mat &f1 = vFeatures[id1];
		Mat &f2 = vFeatures[id2];
		
		//m_D = f1.cols + f2.cols;
		eFeatures[*eit] = Mat(); 
		DadaFeatures::CalculateEdgeFeatureFromVertexFeatures(f1, f2, eFeatures[*eit]);
	}
	
	//LOG_INFO(m_logger, "Edge feature dim is " << m_D); 
	*/
}

void DadaWSACD::InitACDTrain()
{
	LOG_TRACE_METHOD(m_logger, "InitACD");
			
	MamaGraph &gp = *(m_myGraph.get()); 
	LOG_INFO(m_logger, "Orig Graph V:" << num_vertices(gp) << " and E:" << num_edges(gp));

	this->m_gt.Clear();
	// Original segs get all same label
	int gti = 0; 
	for (int j = 0; j < m_h; j++) {
		for (int i = 0; i < m_w; i++) {
			int vali = static_cast<int>(m_basins.at<float>(j, i));

			// get myGraph vertex
			MamaVId gid = m_basinVMap->operator[](vali);
			if (m_gt.Labels().count(gid) == 0) {
				m_gt.Labels()[gid].clear();
			}

			if (m_param->acdOrigMerge) {
				if (m_gt.Labels()[gid].count(0) == 0) m_gt.Labels()[gid][0] = 1.0;
				else m_gt.Labels()[gid][0] += 1.0;
			}
			else {
				int labeli = static_cast<int>(gid);
				if (m_gt.Labels()[gid].count(labeli) == 0) m_gt.Labels()[gid][labeli] = 1.0;
				else m_gt.Labels()[gid][labeli] += 1.0;
			}
		}
	}
	
	// Now add new vertices and edges
	m_newOldVertex.clear();
	m_oldNewVertex.clear();
	m_origVertices.clear(); 
	m_newVertices.clear();

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gp);
	for (nit = nstart; nit != nend; nit++) {
		m_origVertices.push_back(*nit);
		m_newOldVertex[*nit] = *nit; 
	}
	
	for(int i=0; i< m_origVertices.size(); i++) {
		MamaVId id1 = add_vertex(gp);
		m_newVertices.push_back(id1);
		m_newOldVertex[id1] = m_origVertices[i];
		m_oldNewVertex[m_origVertices[i]] = id1;
	}
	
	//int numNewEdges = static_cast<int>(static_cast<double>(num_edges(gp)) * m_param->acdEdgeMultiplier);
	//DadaWSUtil::ChooseRandomEdges(m_origVertices, gp, numNewEdges, m_newEdgeOrigVertices);
	DadaWSUtil::ChooseDegreeEdges(m_origVertices, gp, static_cast<int>(m_param->acdEdgeMultiplier), m_newEdgeOrigVertices);

	for (int i = 0; i < m_newEdgeOrigVertices.size(); i++) {

		if (!m_oldNewVertex.count(m_newEdgeOrigVertices[i].first)) BOOST_THROW_EXCEPTION(Unexpected());
		if (!m_oldNewVertex.count(m_newEdgeOrigVertices[i].second)) BOOST_THROW_EXCEPTION(Unexpected());
		MamaVId id1 = m_oldNewVertex[m_newEdgeOrigVertices[i].first];
		MamaVId id2 = m_oldNewVertex[m_newEdgeOrigVertices[i].second];
		MamaEId mid; bool newEdge;
		std::tie(mid, newEdge) = add_edge(id1, id2, gp);
	}
	
	if (m_param->acdNumTargetLabels < 1) {
		// Assign each new node its own label
		for (int i = 0; i < m_newVertices.size(); i++) {
			MamaVId nid = m_newVertices[i];
			MamaVId oid = m_newOldVertex[nid];
			int olabel = static_cast<int>(oid);

			if (!m_gt.Labels().count(oid)) BOOST_THROW_EXCEPTION(Unexpected());
			
			m_gt.Labels()[nid].clear();

			int syni = (m_param->acdSynMerge) ? static_cast<int>(m_newVertices[0]) : static_cast<int>(nid);

			m_gt.Labels()[nid][syni] = (m_param->acdOrigMerge) ? m_gt.Labels()[oid][0] : m_gt.Labels()[nid][syni] = m_gt.Labels()[oid][olabel];
		}
	}
	else {
		int numWLabels = static_cast<int>(sqrt(static_cast<double>(m_param->acdNumTargetLabels)));
		int blockw;
		if (m_w > m_h) blockw = m_w / numWLabels;
		else blockw = m_h / numWLabels;

		Mat temp = Mat::zeros(m_h, m_w, CV_32F); 

		for (int j = 0; j < m_h; j++) {
			for (int i = 0; i < m_w; i++) {
				// get the block index
				int blockiw = i / blockw;
				int blockih = j / blockw;
				int blocki = blockih * numWLabels + blockiw + 1;

				int vali = static_cast<int>(m_basins.at<float>(j, i));
				// get myGraph vertex
				MamaVId gid = m_basinVMap->operator[](vali);
				MamaVId nid = m_oldNewVertex[gid];
				if (!m_gt.Labels().count(nid)) {
					m_gt.Labels()[nid].clear();
				}

				if (!m_gt.Labels()[nid].count(blocki)) {
					m_gt.Labels()[nid][blocki] = 0.0;
				}
				m_gt.Labels()[nid][blocki] += 1.0;
				temp.at<float>(j, i) = static_cast<float>(blocki); 
			}
		}
		VizMat::DisplayColorSeg(temp, "Other", 0, 1.0);
	}

	

	m_gt.PosWeight() = m_param->posWeight;
	m_gt.NegWeight() = m_param->negWeight;
	
	m_gt.FinalizeCounts(); 

	m_gt.ExtraPos() = 0.0;
	m_gt.ExtraNeg() = 0.0; 
	m_gt.ErrorPos() = 0.0;
	m_gt.ErrorNeg() = 0.0; 
	
	map<int, int> lc; 
	lc.clear();
	for (auto &it : m_gt.Labels()) {
		for (auto &lit : it.second) {
			lc[lit.first] = 1;
		}
	}	
	

	LOG_INFO(m_logger, "ACD Graph V:" << num_vertices(gp) << " and E:" << num_edges(gp) << " GT: " << lc.size() << ", P:" << m_gt.PosCount() << ", N:" << m_gt.NegCount());
	
	this->InitACDFeatures(); 
	//m_gt.Print(); 
	
}

void DadaWSACD::PickNewEdges()
{
	MamaGraph &gp = *(m_myGraph.get());
	LOG_INFO(m_logger, "PickNew Start V:" << num_vertices(gp) << " and E:" << num_edges(gp));

	for (int i = 0; i < m_newEdgeOrigVertices.size(); i++) {
		MamaVId id1 = m_oldNewVertex[m_newEdgeOrigVertices[i].first];
		MamaVId id2 = m_oldNewVertex[m_newEdgeOrigVertices[i].second];	
		remove_edge(id1, id2, gp);
	}

	LOG_INFO(m_logger, "Mid Graph V:" << num_vertices(gp) << " and E:" << num_edges(gp));

	//int numNewEdges = static_cast<int>(static_cast<double>(num_edges(gp)) * m_param->acdEdgeMultiplier);
	//DadaWSUtil::ChooseRandomEdges(m_origVertices, gp, numNewEdges, m_newEdgeOrigVertices);
	DadaWSUtil::ChooseDegreeEdges(m_origVertices, gp, static_cast<int>(m_param->acdEdgeMultiplier), m_newEdgeOrigVertices);

	for (int i = 0; i < m_newEdgeOrigVertices.size(); i++) {

		if (!m_oldNewVertex.count(m_newEdgeOrigVertices[i].first)) BOOST_THROW_EXCEPTION(Unexpected());
		if (!m_oldNewVertex.count(m_newEdgeOrigVertices[i].second)) BOOST_THROW_EXCEPTION(Unexpected());
		MamaVId id1 = m_oldNewVertex[m_newEdgeOrigVertices[i].first];
		MamaVId id2 = m_oldNewVertex[m_newEdgeOrigVertices[i].second];
		MamaEId mid; bool newEdge;
		std::tie(mid, newEdge) = add_edge(id1, id2, gp);		
		//if (!newEdge) BOOST_THROW_EXCEPTION(Unexpected()); 
		//LOG_INFO(m_logger, " New edge: " << m_newEdgeOrigVertices[i].first << ", " << m_newEdgeOrigVertices[i].second);
	}
	LOG_INFO(m_logger, "Pick Graph V:" << num_vertices(gp) << " and E:" << num_edges(gp));
	this->InitACDEdgeFeatures(); 
}


void DadaWSACD::InitACDFeatures()
{
	LOG_TRACE_METHOD(m_logger, "InitACDFeatures");
	/*
	MamaGraph &gp = *(m_myGraph.get());
	
	std::map<MamaVId, cv::Mat> &vFeatures = m_features->GetV();

	// Assign each new node its own label
	for (int i = 0; i < m_newVertices.size(); i++) {
		MamaVId nid = m_newVertices[i];
		MamaVId oid = m_newOldVertex[nid];

		if (!vFeatures.count(oid)) BOOST_THROW_EXCEPTION(Unexpected("No vertex feature"));
		if (vFeatures.count(nid)) BOOST_THROW_EXCEPTION(Unexpected("Vertex feature already exists"));
		// copy vertex features
		vFeatures[nid] = vFeatures[oid].clone();
	}
	this->InitACDEdgeFeatures(); 
	*/
}

void DadaWSACD::InitACDEdgeFeatures()
{
	/*
	MamaGraph &gp = *(m_myGraph.get());
	std::map<MamaVId, cv::Mat> &vFeatures = m_features->GetV();
	std::map<MamaEId, cv::Mat> &eFeatures = m_features->GetE();

	// Now edge features	
	eFeatures.clear(); 
	//Mat v1, v2;
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gp);
	int ei = 0; 

	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, gp);
		MamaVId id2 = target(*eit, gp);
		if (!m_newOldVertex.count(id1)) BOOST_THROW_EXCEPTION(Unexpected());
		if (!m_newOldVertex.count(id2)) BOOST_THROW_EXCEPTION(Unexpected());

		Mat &f1 = vFeatures[id1]; 
		Mat &f2 = vFeatures[id2];
		
		//LOG_INFO(m_logger, " Edge " << ei << ": " << static_cast<int>(id1) << ", " << static_cast<int>(id2));
		ei++;
		eFeatures[*eit] = Mat(); 
		//m_D = f1.cols + f2.cols;		
		DadaFeatures::CalculateEdgeFeatureFromVertexFeatures(f1, f2, eFeatures[*eit]);
	}
		
	//LOG_INFO(m_logger, "Edge feature dim is " << m_D); 
	*/
}

DadaWSACD::DadaWSACD(std::shared_ptr<DadaParam> &param)
	: DadaWS(param), 
	  m_logger(LOG_GET_LOGGER("DadaWSACD"))
{			
}

DadaWSACD::~DadaWSACD()
{	
}


/*

void DadaWSACD::InitACDTrain()
{
LOG_TRACE_METHOD(m_logger, "InitACD");

MamaGraph &gp = *(m_myGraph.get());
LOG_INFO(m_logger, "Orig Graph V:" << num_vertices(gp) << " and E:" << num_edges(gp));

this->m_gt.Clear();
// Original segs get all same label
int gti = 0;
for (int j = 0; j < m_h; j++) {
for (int i = 0; i < m_w; i++) {
int vali = static_cast<int>(m_basins.at<float>(j, i));

// get myGraph vertex
MamaVId gid = m_basinVMap->operator[](vali);
if (m_gt.Labels().count(gid) == 0) {
m_gt.Labels()[gid].clear();
}

if (m_gt.Labels()[gid].count(gti) == 0) {
m_gt.Labels()[gid][gti] = 0.0;
}
m_gt.Labels()[gid][gti] += 1.0;
}
}

// Now add new vertices and edges
m_newOldVertex.clear();
m_origVertices.clear();
MamaNodeIt nit, nstart, nend;
std::tie(nstart, nend) = vertices(gp);
for (nit = nstart; nit != nend; nit++) {
m_origVertices.push_back(*nit);
m_newOldVertex[*nit] = *nit;
}

DadaWSUtil::ChooseRandomEdges(m_origVertices, m_param->acdNumSamples, m_newEdgeOrigVertices);
m_newEdgeNewVertices.clear();
m_newEdgeNewVertices.resize(m_newEdgeOrigVertices.size());
m_newEdgeIndex.clear();
m_newVertices.clear();

for (int i = 0; i < m_newEdgeOrigVertices.size(); i++) {
MamaVId id1 = add_vertex(gp);
MamaVId id2 = add_vertex(gp);
m_newVertices.push_back(id1);
m_newVertices.push_back(id2);
m_newOldVertex[id1] = m_newEdgeOrigVertices[i].first;
m_newOldVertex[id2] = m_newEdgeOrigVertices[i].second;
m_newEdgeNewVertices[i].first = id1;
m_newEdgeNewVertices[i].second = id2;
MamaEId mid; bool newEdge;
std::tie(mid, newEdge) = add_edge(id1, id2, gp);
if (!newEdge) BOOST_THROW_EXCEPTION(Unexpected());
m_newEdgeIndex[mid] = i;
//LOG_INFO(m_logger, " New edge: " << m_newEdgeOrigVertices[i].first << ", " << m_newEdgeOrigVertices[i].second);
}
// Assign each new node its own label
for (int i = 0; i < m_newVertices.size(); i++) {
MamaVId nid = m_newVertices[i];
MamaVId oid = m_newOldVertex[nid];

if (!m_gt.Labels().count(oid)) BOOST_THROW_EXCEPTION(Unexpected());
if (!m_gt.Labels()[oid].count(0)) BOOST_THROW_EXCEPTION(Unexpected());

m_gt.Labels()[nid].clear();
m_gt.Labels()[nid][i + 1] = m_gt.Labels()[oid][0];
}

m_gt.PosWeight() = m_param->posWeight;
m_gt.NegWeight() = m_param->negWeight;

m_gt.FinalizeCounts();

m_gt.ExtraPos() = 0.0;
m_gt.ExtraNeg() = 0.0;
m_gt.ErrorPos() = 0.0;
m_gt.ErrorNeg() = 0.0;

LOG_INFO(m_logger, "ACD Graph V:" << num_vertices(gp) << " and E:" << num_edges(gp) << " GT: " << m_gt.PosCount() << ", " << m_gt.NegCount());

this->InitACDFeatures();
//m_gt.Print();

}

*/