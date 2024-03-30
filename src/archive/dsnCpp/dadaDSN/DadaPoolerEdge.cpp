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

#include "DadaPoolerEdge.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"

void DadaPoolerEdge::FinalizeEdgeFeatures(MamaGraph &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "FinalizeEdgeFeatures");
	m_VD = 0;
	
	// Lets get perimeter
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(myGraph);

	double depthMin = LARGEST_DOUBLE; 
	double depthMax = SMALLEST_DOUBLE;
	m_ED = 1;

	for (eit = estart; eit != eend; eit++) {
		m_EFeatures[*eit] = Mat::zeros(1, m_ED, CV_64F);
		int upto = 0;
		m_EFeatures[*eit].at<double>(upto) = minE[*eit]; // min edge

		depthMin = std::min(depthMin, minE[*eit]);
		depthMax = std::max(depthMax, minE[*eit]);
	}

	//LOG_INFO(m_logger, "Min edge feature from " << depthMin << " to " << depthMax); 
}

void DadaPoolerEdge::PoolFeatures(vector<Mat> &imgs, Mat &basins, MamaGraph &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "PoolFeatures");

	minV.clear(); maxV.clear(); numV.clear(); numEV.clear();
	minE.clear(); maxE.clear(); numE.clear();

	for (int j = 0; j < basins.rows; j++) {
	for (int i = 0; i < basins.cols; i++) {
		int myl = static_cast<int>(basins.at<float>(j, i));		
		double myv = static_cast<double>(imgs[0].at<float>(j, i));
		MamaVId nid = m_VId->operator[](myl);
		if (!numV.count(nid)) {
			numV[nid] = 1.0;
			minV[nid] = myv;
			maxV[nid] = myv;
		}
		else {
			numV[nid] = numV[nid] + 1.0;
			minV[nid] = std::min(minV[nid], myv); 
			maxV[nid] = std::max(maxV[nid], myv); 
		}

		if (i > 0) {
			int leftl = static_cast<int>(basins.at<float>(j, i-1));			
			if (leftl != myl) {
				double leftv = static_cast<double>(imgs[0].at<float>(j, i - 1));
				this->PoolEdge(leftl, leftv, myl, myv, myGraph);
			}
		}
		if (j > 0) {
			int topl = static_cast<int>(basins.at<float>(j-1, i));			
			if (topl != myl) {
				double topv = static_cast<double>(imgs[0].at<float>(j - 1, i));
				this->PoolEdge(topl, topv, myl, myv, myGraph);
			}
		}
	}
	}

	this->FinalizeEdgeFeatures(myGraph);
}

void DadaPoolerEdge::PoolEdge(int leftl, double leftv, int myl, double myv, MamaGraph &myGraph)
{
	MamaVId lid = m_VId->operator[](leftl);
	MamaVId nid = m_VId->operator[](myl);

	if (!edge(nid, lid, myGraph).second) LOG_INFO(m_logger, "Couldn't find edge " << nid << " to " << lid);

	MamaEId eid = edge(nid, lid, myGraph).first;

	double emin = (leftv < myv) ? leftv : myv;
	double emax = (leftv > myv) ? leftv : myv;

	if (!numE.count(eid)) {
		numE[eid] = 1.0;
		minE[eid] = emin;
		maxE[eid] = emax;
	}
	else {
		numE[eid] = numE[eid] + 1.0;
		minE[eid] = std::min(minE[eid], emin); 
		maxE[eid] = std::max(maxE[eid], emax); 
	}	
}

DadaPoolerEdge::DadaPoolerEdge(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaPooler(basins, myVId, param)
{	
}

DadaPoolerEdge::DadaPoolerEdge(std::shared_ptr<DadaParam> &param)
	: DadaPooler(param)
{
}

DadaPoolerEdge::~DadaPoolerEdge()
{	
}


/*

void DadaPoolerEdge::FinalizeEdgeFeatures(MamaGraph &myGraph)
{
LOG_TRACE_METHOD(m_logger, "FinalizeEdgeFeatures");
m_VD = 0;
m_ED = 1;

// Lets get perimeter
MamaEdgeIt eit, estart, eend;
std::tie(estart, eend) = edges(myGraph);


for (eit = estart; eit != eend; eit++) {
MamaVId id1 = source(*eit, myGraph);
MamaVId id2 = target(*eit, myGraph);
if (!numEV.count(id1))		numEV[id1] = numE[*eit];
else						numEV[id1] += numE[*eit];
if (!numEV.count(id2))		numEV[id2] = numE[*eit];
else						numEV[id2] += numE[*eit];
}

double depthMin = LARGEST_DOUBLE;
double depthMax = SMALLEST_DOUBLE;

for (eit = estart; eit != eend; eit++) {
MamaVId id1 = source(*eit, myGraph);
MamaVId id2 = target(*eit, myGraph);
double depth1 = minE[*eit] - minV[id1];
double depth2 = minE[*eit] - minV[id2];
double minDepth = std::min(depth1, depth2);

depthMin = std::min(depthMin, minDepth);
depthMax = std::max(depthMax, minDepth);

double vol1 = depth1 * numV[id1];
double vol2 = depth2 * numV[id2];
double minVol = std::min(vol1, vol2);

// perimeter remaining after shared edge
double rem1 = numEV[id1] - numE[*eit];
double rem2 = numEV[id2] - numE[*eit];
double minRem = std::min(rem1, rem2);

// aspect ratios
double ar1, ar2, comar;
if (numEV[id1] > 0.0)   ar1 = (4.0 * PI * numV[id1]) / (numEV[id1] * numEV[id1]);
else					ar1 = 0.0;
if (numEV[id2] > 0.0)   ar2 = (4.0 * PI * numV[id2]) / (numEV[id2] * numEV[id2]);
else					ar2 = 0.0;
double coma = numV[id1] + numV[id2];
double comp = rem1 + rem2;
if (comp > 0.0)   comar = (4.0 * PI * coma) / (comp * comp);
else			  comar = 0.0;

m_EFeatures[*eit] = Mat::zeros(1, m_ED, CV_64F);
int upto = 0;

m_EFeatures[*eit].at<double>(upto) = minDepth; // depth
upto++;
m_EFeatures[*eit].at<double>(upto) = minE[*eit]; // min edge
upto++;
m_EFeatures[*eit].at<double>(upto) = minVol; // depth
upto++;
m_EFeatures[*eit].at<double>(upto) = numE[*eit]; // length
upto++;
m_EFeatures[*eit].at<double>(upto) = minRem; // remaing edge
upto++;
m_EFeatures[*eit].at<double>(upto) = minRem; // remaing edge
upto++;
m_EFeatures[*eit].at<double>(upto) = comar; // combined aspect ratio
upto++;

}

LOG_INFO(m_logger, "Depth feature from " << depthMin << " to " << depthMax);
}

*/