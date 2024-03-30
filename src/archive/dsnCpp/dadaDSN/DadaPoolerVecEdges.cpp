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

#include "DadaPoolerVecEdges.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "VecMath.h"

void DadaPoolerVecEdges::MergeFeatures(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc,
	std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge)
{
	LOG_TRACE_METHOD(m_logger, "DadaPoolerVecEdges::MergeFeatures");
		
	m_VD = 0;
	m_ED = pp.D();
		
	/*
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gc);

	for (eit = estart; eit != eend; eit++) {
			
		m_EFeatures[*eit] = Mat::ones(1, D, CV_64F) * LARGEST_DOUBLE;
		for (int fi = 0; fi< D; fi++) m_EFeatures[*eit].at<double>(0, fi+D) = SMALLEST_DOUBLE;
			
		if (!childParentEdge.count(*eit)) BOOST_THROW_EXCEPTION(Unexpected("Couldn't find edge")); 

		for (auto &it : childParentEdge[*eit]) {
			if (pp.GetEFeature(it).cols != pp.D()) BOOST_THROW_EXCEPTION(Unexpected("edge features wrong size"));
			for (int fi = 0; fi < D; fi++) {
				m_EFeatures[*eit].at<double>(0, fi) = min(m_EFeatures[*eit].at<double>(0, fi), pp.GetEFeature(it).at<double>(0, fi));
			}
		}					
	}
	*/
}

void DadaPoolerVecEdges::SmoothEdges(MamaGraph &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "SmoothEdges");

	map<MamaEId, Mat> tempEdges; 
	map<MamaEId, double> tempCount;
	tempEdges.clear(); 
	tempCount.clear(); 

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(myGraph);

	for (nit = nstart; nit != nend; nit++) {
		MamaNeighborIt nnit1, nnit2, nnstart, nnend;
		std::tie(nnstart, nnend) = adjacent_vertices(*nit, myGraph);
		for (nnit1 = nnstart; nnit1 != nnend; nnit1++) {
			for (nnit2 = nnstart; nnit2 != nnend; nnit2++) {
				if (*nnit1 < *nnit2) {
					if (edge(*nnit1, *nnit2, myGraph).second) {
						if (!edge(*nnit1, *nit, myGraph).second) BOOST_THROW_EXCEPTION(Unexpected("Couldn't find edge")); 
						if (!edge(*nnit2, *nit, myGraph).second) BOOST_THROW_EXCEPTION(Unexpected("Couldn't find edge"));

						MamaEId eid1 = edge(*nnit1, *nit, myGraph).first;
						MamaEId eid2 = edge(*nnit2, *nit, myGraph).first;

						if (!tempEdges.count(eid1)) {
							tempEdges[eid1] = m_EFeatures[eid1]; 
							tempCount[eid1] = 1.0; 
						}
						if (!tempEdges.count(eid2)) {
							tempEdges[eid2] = m_EFeatures[eid2];
							tempCount[eid2] = 1.0;
						}
						// Now add up
						tempEdges[eid1] = tempEdges[eid1] + tempEdges[eid2];
						tempCount[eid1] += 1.0;
						tempEdges[eid2] = tempEdges[eid2] + tempEdges[eid1];
						tempCount[eid2] += 1.0;

					}
				}
			}
		}
	}

	// Normalize ?
	for (auto &it : tempEdges) {
		it.second = it.second / tempCount[it.first];
	}
	// Set back to edges
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(myGraph);	

	for (eit = estart; eit != eend; eit++) {
		if (tempEdges.count(*eit)) {
			m_EFeatures[*eit] = tempEdges[*eit].clone();
		}
	}
}


void DadaPoolerVecEdges::FinalizeEdgeFeatures(MamaGraph &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "FinalizeEdgeFeatures");
	
	m_VD = 0;
	m_ED = 9;

	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(myGraph);
	pair<MamaVId, MamaVId> p;

	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, myGraph);
		MamaVId id2 = target(*eit, myGraph);
		m_EFeatures[*eit] = Mat::zeros(1, m_ED, CV_64F);
		double dh = 0.0; 
		double dv = 0.0; 

		p.first = id1;
		p.second = id2;
		if (numHL.count(p)) dh += numHL[p];
		if (numVU.count(p)) dv += numVU[p];

		p.first = id2;
		p.second = id1;
		if (numHL.count(p)) dh -= numHL[p];
		if (numVU.count(p)) dv -= numVU[p];
		double angle = atan2(dv, dh); // -PI to +PI;
		double sval = (angle + PI) / (2.0*PI);
		//double mag = sqrt(dh*dh + dv*dv); 
		double mag = 1.0; // sqrt(dh*dh + dv*dv);
		double w1, w2; 
		int b1, b2; 
		LinearBins(sval, m_ED, w1, b1, w2, b2);
		m_EFeatures[*eit].at<double>(0, b1) += w1*mag; 
		m_EFeatures[*eit].at<double>(0, b2) += w2*mag;
	}

	this->SmoothEdges(myGraph);
}

void DadaPoolerVecEdges::LinearBins(double val, int numBins, double &w1, int &b1, double &w2, int &b2)
{
	
	double nbinsd = static_cast<double>(numBins);
	double binw = 1.0 / nbinsd;	// 1.0 is expected range
	double bin1 = binw / 2.0;
	double binn = bin1 + (nbinsd - 1.0)*binw;

	double binc, binc2, alpha;

	if (val < bin1) {
		alpha = (bin1 - val) / binw;
		binc = 0.0;
		binc2 = nbinsd - 1.0;
	}
	else if (val > binn) {
		alpha = (val - binn) / binw;
		binc = nbinsd - 1.0;
		binc2 = 0.0;
	}
	else {
		double bind = (val - bin1) / binw;
		double bfloor = floor(bind);
		alpha = (bind - bfloor);
		binc = bfloor;
		binc2 = bfloor + 1.0;
	}
	w1 = 1.0 - alpha; 
	b1 = static_cast<int>(binc); 
	w2 = alpha; 
	b2 = static_cast<int>(binc2);
}

void DadaPoolerVecEdges::PoolFeatures(vector<Mat> &imgs, Mat &basins, MamaGraph &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "PoolFeatures");
	//LOG_INFO(m_logger, "PoolFeatures has " << imgs.size()); 
	
	numVU.clear(); 	
	numHL.clear(); 		

	for (int j = 0; j < basins.rows; j++) {
	for (int i = 0; i < basins.cols; i++) {
		int myl = static_cast<int>(basins.at<float>(j, i));				
		MamaVId nid = m_VId->operator[](myl);
		pair<MamaVId, MamaVId> p;
		p.first = nid;

		if (i > 0) {
			int leftl = static_cast<int>(basins.at<float>(j, i-1));			
			if (leftl != myl) {
				MamaVId lid = m_VId->operator[](leftl);	
				p.second = lid; 

				if (!numHL.count(p)) {
					numHL[p] = 0.0; 					
					numVU[p] = 0.0;					
				}
				numHL[p] += 1.0;				
			}
		}

		if (j > 0) {
			int topl = static_cast<int>(basins.at<float>(j-1, i));			
			if (topl != myl) {
				MamaVId lid = m_VId->operator[](topl);
				p.second = lid;
				if (!numVU.count(p)) {
					numHL[p] = 0.0;
					numVU[p] = 0.0;
				}
				numVU[p] += 1.0;
			}
		}
	}
	}

	this->FinalizeEdgeFeatures(myGraph);
}

DadaPoolerVecEdges::DadaPoolerVecEdges(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaPooler(basins, myVId, param)
{	
}

DadaPoolerVecEdges::DadaPoolerVecEdges(std::shared_ptr<DadaParam> &param)
	: DadaPooler(param)
{
}

DadaPoolerVecEdges::~DadaPoolerVecEdges()
{	
}

