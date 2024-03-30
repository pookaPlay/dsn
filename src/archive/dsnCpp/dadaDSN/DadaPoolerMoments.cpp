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

#include "DadaPoolerMoments.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"


void DadaPoolerMoments::PoolFeatures(vector<Mat> &imgs, Mat &basins, MamaGraph &myGraph)
{
	this->InitPools(imgs.size() * 2);

	Mat tempv = Mat::zeros(1, imgs.size(), CV_64F);

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

void DadaPoolerMoments::InitPools(int featureDim)
{
	m_EFeatures.clear();
	m_VFeatures.clear();
	m_VD = featureDim; 

	for (auto &it : *(m_VId.get())) {
		m_VFeatures[it.second] = Mat::zeros(1, m_VD, CV_64F);
		m_numV[it.second] = 0.0; 
	}
}

void DadaPoolerMoments::AddToPool(cv::Mat &feature, int label)
{
	if (m_VId->count(label) == 0) BOOST_THROW_EXCEPTION(Unexpected());
	MamaVId nid = m_VId->operator[](label);
	int D = m_VD / 2;

	m_numV[nid] += 1.0; 
	for (int i = 0; i < D; i++) {
		m_VFeatures[nid].at<double>(i) += feature.at<double>(i);
		m_VFeatures[nid].at<double>(D + i) += (feature.at<double>(i) * feature.at<double>(i));
	}
}

void DadaPoolerMoments::FinalizePools(MamaGraph &myGraph)
{
	int D = m_VD / 2;

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(myGraph);

	for (nit = nstart; nit != nend; nit++) {
		for (int i = 0; i < D; i++) {
			// mean
			m_VFeatures[*nit].at<double>(i) = m_VFeatures[*nit].at<double>(i) / m_numV[*nit];
			// variance
			m_VFeatures[*nit].at<double>(D + i) = m_VFeatures[*nit].at<double>(D + i) / m_numV[*nit];
			m_VFeatures[*nit].at<double>(D + i) = m_VFeatures[*nit].at<double>(D + i) - (m_VFeatures[*nit].at<double>(i) * m_VFeatures[*nit].at<double>(i));
		}
	}

	// default edge feature is euclidean difference
	Mat v1, v2;
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(myGraph);

	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, myGraph);
		MamaVId id2 = target(*eit, myGraph);
		m_EFeatures[*eit] = Mat::zeros(1, 1, CV_64F);
		for (int i = 0; i < m_VD; i++) {
			double tempd = m_VFeatures[id1].at<double>(i) - m_VFeatures[id2].at<double>(i);
			m_EFeatures[*eit].at<double>(0) += (tempd*tempd);
		}
	}
	m_ED = 1;

}

DadaPoolerMoments::DadaPoolerMoments(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaPooler(basins, myVId, param)
{	
}

DadaPoolerMoments::DadaPoolerMoments(std::shared_ptr<DadaParam> &param)
	: DadaPooler(param)
{
}

DadaPoolerMoments::~DadaPoolerMoments()
{	
}
