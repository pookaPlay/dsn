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

#include "DadaPoolerEdges.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "VecMath.h"

void DadaPoolerEdges::MergeFeatures(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc,
	std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge)
{
	LOG_TRACE_METHOD(m_logger, "DadaPoolerEdges::MergeFeatures");
	
	if (pp.GetBasinInit()) {	
		// first merge so double dimension
		int D = pp.D();
		m_VD = 0;
		m_ED = D*2;

		LOG_TRACE(m_logger, "Merge from basin " << D);		

		MamaEdgeIt eit, estart, eend;
		std::tie(estart, eend) = edges(gc);

		for (eit = estart; eit != eend; eit++) {
			
			m_EFeatures[*eit] = Mat::ones(1, 2*D, CV_64F) * LARGEST_DOUBLE;
			for (int fi = 0; fi< D; fi++) m_EFeatures[*eit].at<double>(0, fi+D) = SMALLEST_DOUBLE;
			
			if (!childParentEdge.count(*eit)) BOOST_THROW_EXCEPTION(Unexpected("Couldn't find edge")); 

			for (auto &it : childParentEdge[*eit]) {
				if (pp.GetEFeature(it).cols != pp.D()) BOOST_THROW_EXCEPTION(Unexpected("edge features wrong size"));
				for (int fi = 0; fi < D; fi++) {
					m_EFeatures[*eit].at<double>(0, fi) = min(m_EFeatures[*eit].at<double>(0, fi), pp.GetEFeature(it).at<double>(0, fi));
					m_EFeatures[*eit].at<double>(0, fi + D) = max(m_EFeatures[*eit].at<double>(0, fi + D), pp.GetEFeature(it).at<double>(0, fi));
				}
			}			
		}

	}
	else {
		// Continue min and max up hierarchy		
		int D = pp.D() / 2;		
		m_VD = 0;
		m_ED = pp.D();

		LOG_TRACE(m_logger, "Merge in tree " << pp.D());

		MamaEdgeIt eit, estart, eend;
		std::tie(estart, eend) = edges(gc);

		for (eit = estart; eit != eend; eit++) {

			m_EFeatures[*eit] = Mat::ones(1, 2*D, CV_64F) * LARGEST_DOUBLE;
			for (int i = 0; i< D; i++) m_EFeatures[*eit].at<double>(0, i + D) = SMALLEST_DOUBLE;

			if (!childParentEdge.count(*eit)) BOOST_THROW_EXCEPTION(Unexpected("Couldn't find edge"));

			for (auto &it : childParentEdge[*eit]) {
				if (pp.GetEFeature(it).cols != pp.D()) BOOST_THROW_EXCEPTION(Unexpected("edge features wrong size"));

				for (int fi = 0; fi < D; fi++) {
					m_EFeatures[*eit].at<double>(0, fi) = min(m_EFeatures[*eit].at<double>(0, fi), pp.GetEFeature(it).at<double>(0, fi));			
					m_EFeatures[*eit].at<double>(0, fi + D) = max(m_EFeatures[*eit].at<double>(0, fi + D), pp.GetEFeature(it).at<double>(0, fi + D));
				}
			}
		}

	}
}

void DadaPoolerEdges::FinalizeEdgeFeatures(MamaGraph &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "FinalizeEdgeFeatures");
	
	// Lets get perimeter
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(myGraph);		

	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, myGraph);
		MamaVId id2 = target(*eit, myGraph);
		//double depth1 = minE[*eit] - minV[id1];
		//double depth2 = minE[*eit] - minV[id2];
		//double minDepth = std::min(depth1, depth2);
		m_EFeatures[*eit] = Mat(minE[*eit]).t();
		//m_EFeatures[*eit] = Mat::zeros(1, m_ED, CV_64F);		
		//for (int f = 0; f < m_ED; f++) m_EFeatures[*eit].at<double>(f) = minE[*eit][f]; 
	}
}

void DadaPoolerEdges::PoolFeatures(vector<Mat> &imgs, Mat &basins, MamaGraph &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "PoolFeatures");
	//LOG_INFO(m_logger, "PoolFeatures has " << imgs.size()); 
	minV.clear(); maxV.clear(); accV.clear();  numV.clear();
	minE.clear(); maxE.clear(); accE.clear();  numE.clear();

	vector<double> myv, neighv;
	myv.clear(); neighv.clear(); 
	myv.resize(imgs.size()); 
	neighv.resize(imgs.size());

	for (int j = 0; j < basins.rows; j++) {
	for (int i = 0; i < basins.cols; i++) {
		int myl = static_cast<int>(basins.at<float>(j, i));				
		MamaVId nid = m_VId->operator[](myl);
		
		for (int f = 0; f < imgs.size(); f++) {
			myv[f] = static_cast<double>(imgs[f].at<float>(j, i));
		}

		if (!numV.count(nid)) {
			numV[nid] = 1.0;
			minV[nid] = myv; 
			maxV[nid] = myv; 
			accV[nid] = myv; 
		}
		else {
			numV[nid] = numV[nid] + 1.0;
			VecMath::VectorMin(minV[nid], myv, minV[nid]);
			VecMath::VectorMax(maxV[nid], myv, maxV[nid]);
			VecMath::VectorSum(accV[nid], myv, accV[nid]);
		}

		if (i > 0) {
			int leftl = static_cast<int>(basins.at<float>(j, i-1));			
			if (leftl != myl) {
				for (int f = 0; f < imgs.size(); f++) {
					neighv[f] = static_cast<double>(imgs[0].at<float>(j, i - 1));
				}
				this->PoolEdge(leftl, neighv, myl, myv, myGraph);
			}
		}
		if (j > 0) {
			int topl = static_cast<int>(basins.at<float>(j-1, i));			
			if (topl != myl) {
				for(int f = 0; f < imgs.size(); f++) {
					neighv[f] = static_cast<double>(imgs[0].at<float>(j - 1, i));
				}
				this->PoolEdge(topl, neighv, myl, myv, myGraph);
			}
		}
	}
	}

	this->FinalizeEdgeFeatures(myGraph);
	m_VD = 0;
	m_ED = imgs.size(); 

}

void DadaPoolerEdges::PoolEdge(int leftl, vector<double> &leftv, int myl, vector<double> &myv, MamaGraph &myGraph)
{
	MamaVId lid = m_VId->operator[](leftl);
	MamaVId nid = m_VId->operator[](myl);

	if (!edge(nid, lid, myGraph).second) LOG_INFO(m_logger, "Couldn't find edge " << nid << " to " << lid);

	MamaEId eid = edge(nid, lid, myGraph).first;

	vector<double> emin, emax, eavg; 
	VecMath::VectorMin(leftv, myv, emin); 
	VecMath::VectorMax(leftv, myv, emax);
	VecMath::VectorAvg(leftv, myv, eavg);

	if (!numE.count(eid)) {
		numE[eid] = 1.0;
		minE[eid] = emin;
		maxE[eid] = emax;
		accE[eid] = eavg;
	}
	else {
		numE[eid] = numE[eid] + 1.0;

		VecMath::VectorMin(minE[eid], emin, minE[eid]);
		VecMath::VectorMax(maxE[eid], emax, maxE[eid]);
		VecMath::VectorSum(accE[eid], eavg, accE[eid]);
	}	
}

DadaPoolerEdges::DadaPoolerEdges(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaPooler(basins, myVId, param)
{	
}

DadaPoolerEdges::DadaPoolerEdges(std::shared_ptr<DadaParam> &param)
	: DadaPooler(param)
{
}

DadaPoolerEdges::~DadaPoolerEdges()
{	
}

