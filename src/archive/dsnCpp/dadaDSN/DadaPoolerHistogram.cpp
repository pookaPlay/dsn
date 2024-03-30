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

#include "DadaPoolerHistogram.h"
#include "DadaFeatures.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"


void DadaPoolerHistogram::CalculateEdgeFeatureFromVertexFeatures(cv::Mat &f1, cv::Mat &f2, cv::Mat &e)
{
	//LOG_TRACE_METHOD(m_logger, "DadaPoolerHistogram::CalculateEdgeFeatureFromVertexFeatures"); 
	if (m_VD != f1.cols) BOOST_THROW_EXCEPTION(Unexpected("Vertex feature wrong dim"));

	int numNF = m_VD / HISTOGRAM_GRAD_BINS;
	m_ED = m_VD + numNF + 1;		// normalize each histogram and normalize all 
	e = Mat::zeros(1, m_ED, CV_64F);
	Mat v1 = abs(f1 - f2);
	v1.copyTo(e(Range(0, 1), Range(0, f1.cols)));


	double squaredTotal = 0.0;
	for (int f = 0; f < numNF; f++) {
		double currentTotal = 0.0;
		for (int h = 0; h < HISTOGRAM_GRAD_BINS; h++) {
			int ind = f * HISTOGRAM_GRAD_BINS + h;
			double diff = f1.at<double>(ind) -f2.at<double>(ind);
			currentTotal += (diff*diff);
		}
		e.at<double>(f1.cols + f) = currentTotal;
		squaredTotal += currentTotal;
	}
	e.at<double>(f1.cols + numNF) = squaredTotal;
}


void DadaPoolerHistogram::PoolFeatures(vector<Mat> &imgs, Mat &basins, MamaGraph &myGraph)
{
	//LOG_INFO(m_logger, "PoolFeatures has " << imgs.size());

	int numBands = imgs.size() / 2;
	
	m_VD = HISTOGRAM_GRAD_BINS * numBands;

	this->InitPools(m_VD);	

	Mat tempv = Mat::zeros(1, numBands*4, CV_64F);

	for (int j = 0; j < basins.rows; j++) {
		for (int i = 0; i < basins.cols; i++) {
			int myl = static_cast<int>(basins.at<float>(j, i));

			// Get the data into a vector
			for (int f = 0; f < numBands; f++) {
				// mag 1
				tempv.at<double>(f*4) = static_cast<double>(imgs[f*2].at<Vec2f>(j, i)[0]); 
				// ind 1
				tempv.at<double>(f*4+1) = static_cast<double>(imgs[f*2+1].at<Vec2b>(j, i)[0]);
				// mag 2
				tempv.at<double>(f*4+2) = static_cast<double>(imgs[f*2].at<Vec2f>(j, i)[1]);
				// ind 2
				tempv.at<double>(f*4+3) = static_cast<double>(imgs[f*2+1].at<Vec2b>(j, i)[1]);
			}			

			this->AddToPool(tempv, myl);
		}
	}

	this->FinalizePools(myGraph);
}

void DadaPoolerHistogram::InitPools(int featureDim)
{	
	m_EFeatures.clear();
	m_VFeatures.clear();

	for (auto &it : *(m_VId.get())) {
		m_VFeatures[it.second] = Mat::zeros(1, featureDim, CV_64F);
		m_numV[it.second] = 0.0; 
	}
}

void DadaPoolerHistogram::AddToPool(cv::Mat &feature, int label)
{
	if (m_VId->count(label) == 0) BOOST_THROW_EXCEPTION(Unexpected());
	MamaVId nid = m_VId->operator[](label);

	int numBands = feature.cols / 4; 
	m_numV[nid] += 1.0;

	for (int f = 0; f < numBands; f++) {
		double mag1 = feature.at<double>(f * 4);
		int ind1 = static_cast<int>(feature.at<double>(f * 4 + 1));
		double mag2 = feature.at<double>(f * 4 + 2);
		int ind2 = static_cast<int>(feature.at<double>(f * 4 + 3));

		int aind1 = f*HISTOGRAM_GRAD_BINS + ind1;
		int aind2 = f*HISTOGRAM_GRAD_BINS + ind2;

		m_VFeatures[nid].at<double>(aind1) += mag1;
		m_VFeatures[nid].at<double>(aind2) += mag2;
	}
}

void DadaPoolerHistogram::FinalizePools(MamaGraph &myGraph)
{
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(myGraph);

	for (nit = nstart; nit != nend; nit++) {
		if (m_numV[*nit] > 0.0) {
			for (int f = 0; f < m_VFeatures[*nit].cols; f++) {				
				m_VFeatures[*nit].at<double>(f) = m_VFeatures[*nit].at<double>(f) / m_numV[*nit]; 				
			}
		}
	}

	this->CalculateEdgesFromVertices(myGraph);

}

DadaPoolerHistogram::DadaPoolerHistogram(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaPooler(basins, myVId, param)
{	
}

DadaPoolerHistogram::DadaPoolerHistogram(std::shared_ptr<DadaParam> &param)
	: DadaPooler(param)
{
}

DadaPoolerHistogram::~DadaPoolerHistogram()
{	
}
