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

#include "DadaPooler.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "Normalize.h"

void DadaPooler::SplitFeatures(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc, std::map<MamaVId, MamaVId> &parentChildVertex)
{
	m_VD = pp.VD();
	m_VFeatures.clear();

	// first pool by label
	std::map<int, cv::Mat> labelPool; 
	map<int, double> labelCount;
	labelCount.clear();
	labelPool.clear(); 
	
	MamaNodeIt nit, nstart, nend;
	std::map<MamaVId, cv::Mat> &pfeatures = pp.GetVFeatures();

	std::tie(nstart, nend) = vertices(gp);
	for (nit = nstart; nit != nend; nit++) {
		int lid = gp[*nit].label;
		if (!labelCount.count(lid)) {
			labelCount[lid] = 0.0; 
			labelPool[lid] = Mat::zeros(1, m_VD, CV_64F);
		}
		labelPool[lid] = labelPool[lid] + pfeatures[*nit];
		labelCount[lid] += 1.0;
	}
	// Normalize
	for (auto &it : labelPool) {
		it.second = it.second / labelCount[it.first];
	}

	// Now add features for new graph
	m_VFeatures.clear();
	for (nit = nstart; nit != nend; nit++) {
		int lid = gp[*nit].label;
		MamaVId myn = parentChildVertex[*nit]; 
		// Use the difference between me and my segment average
		m_VFeatures[myn] = abs(pfeatures[*nit] - labelPool[lid]); 			
	}
	// And get edge features
	CalculateEdgesFromVertices(gc);
}

void DadaPooler::MergeFeatures(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc,
	std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge)
{
	PoolParent(pp, gp, gc, labelChild);
	CalculateEdgesFromVertices(gc);
}

void DadaPooler::PoolParent(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc, std::map<int, MamaVId> &labelChild)
{
	LOG_TRACE_METHOD(m_logger, "DadaPooler::PoolParent");
	std::map<MamaVId, cv::Mat> &pfeatures = pp.GetVFeatures();

	m_VD = pp.VD(); 
	m_VFeatures.clear();

	// Initialize to zero
	map<MamaVId, double> vcount;
	vcount.clear();
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gc);
	for (nit = nstart; nit != nend; nit++) {
		m_VFeatures[*nit] = Mat::zeros(1, m_VD, CV_64F);
		vcount[*nit] = 0.0;
	}

	// Pool
	std::tie(nstart, nend) = vertices(gp);
	for (nit = nstart; nit != nend; nit++) {
		int lid = gp[*nit].label;
		MamaVId nid = labelChild[lid];

		if (!pfeatures.count(*nit)) BOOST_THROW_EXCEPTION(Unexpected());

		m_VFeatures[nid] = m_VFeatures[nid] + pfeatures[*nit];
		vcount[nid] += 1.0;
	}

	// then normalize by count
	std::tie(nstart, nend) = vertices(gc);
	for (nit = nstart; nit != nend; nit++) {
		if (vcount[*nit] > 0.0) {
			m_VFeatures[*nit] = m_VFeatures[*nit] / vcount[*nit];
		}
	}
}

void DadaPooler::CalculateEdgesFromVertices(MamaGraph &gc)
{
	LOG_TRACE_METHOD(m_logger, "DadaPooler::CalculateEdgesFromVertices");
	// edge feature for pooled edges
	m_EFeatures.clear();
	
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gc);

	for (eit = estart; eit != eend; eit++) {
		m_EFeatures[*eit] = Mat();
		MamaVId id1 = source(*eit, gc);
		MamaVId id2 = target(*eit, gc);
		if (!m_VFeatures.count(id1)) BOOST_THROW_EXCEPTION(Unexpected());
		if (!m_VFeatures.count(id2)) BOOST_THROW_EXCEPTION(Unexpected());
		this->CalculateEdgeFeatureFromVertexFeatures(m_VFeatures[id1], m_VFeatures[id2], m_EFeatures[*eit]);
	}
}

void DadaPooler::CalculateEdgeFeatureFromVertexFeatures(cv::Mat &f1, cv::Mat &f2, cv::Mat &e)
{
	if (m_VD != f1.cols) BOOST_THROW_EXCEPTION(Unexpected("Vertex feature wrong dim"));
	m_ED = m_VD; 
	e = abs(f1 - f2);
}

void DadaPooler::PoolFeatures(vector<Mat> &imgs, Mat &basins, MamaGraph &myGraph)
{
	m_VD = imgs.size();
	this->InitPools(m_VD); 

	Mat tempv = Mat::zeros(1, m_VD, CV_64F);

	for (int j = 0; j < basins.rows; j++) {
		for (int i = 0; i < basins.cols; i++) {
			int myl = static_cast<int>(basins.at<float>(j, i));

			// Get the data into a vector
			for (int di = 0; di < imgs.size(); di++) {
				tempv.at<double>(di) = static_cast<double>(imgs[di].at<float>(j, i));
			}

			this->AddToPool(tempv, myl);
		}
	}

	this->FinalizePools(myGraph);
		
}

void DadaPooler::InitPools(int featureDim)
{	
	m_EFeatures.clear();
	m_VFeatures.clear();	
	
	for (auto &it : *(m_VId.get())) {
		m_VFeatures[it.second] = Mat::zeros(1, featureDim, CV_64F);		
	}	
}

void DadaPooler::AddToPool(cv::Mat &feature, int label)
{
	if (m_VId->count(label) == 0) BOOST_THROW_EXCEPTION(Unexpected()); 
	MamaVId nid = m_VId->operator[](label);
	m_VFeatures[nid] = m_VFeatures[nid] + feature; 
}

void DadaPooler::FinalizePools(MamaGraph &myGraph)
{	
	this->CalculateEdgesFromVertices(myGraph);
}

cv::Mat & DadaPooler::GetVFeature(MamaVId nid)
{
	if (!m_VFeatures.count(nid)) BOOST_THROW_EXCEPTION(Unexpected());
	return(m_VFeatures[nid]);
}

cv::Mat & DadaPooler::GetEFeature(MamaEId eid)
{	
	if (!m_EFeatures.count(eid)) BOOST_THROW_EXCEPTION(Unexpected()); 
	return(m_EFeatures[eid]);	
}


DadaPooler::DadaPooler(std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaPooler")),
	m_param(param)
{
	m_EFeatures.clear();
	m_VFeatures.clear();
	m_basinInit = 0; 
	m_VD = 0;
	m_ED = 0;
}

DadaPooler::DadaPooler(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaPooler")),
	m_basins(basins),
	m_VId(myVId),
	m_param(param)
{	
	m_EFeatures.clear();
	m_VFeatures.clear();
	m_basinInit = 1;
	m_VD = 0; 
	m_ED = 0;
}

DadaPooler::~DadaPooler()
{	
}

