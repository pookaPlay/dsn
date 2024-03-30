#if !defined(DadaPoolerHistogram_H__)
#define DadaPoolerHistogram_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaPooler.h"

#define HISTOGRAM_GRAD_BINS		9

class DadaPoolerHistogram : public DadaPooler
{
	public:				
		void PoolFeatures(std::vector<cv::Mat> &imgs, cv::Mat &basins, MamaGraph &myGraph) override;

		void CalculateEdgeFeatureFromVertexFeatures(cv::Mat &f1, cv::Mat &f2, cv::Mat &e) override;

		void InitPools(int featureDim);
		void AddToPool(cv::Mat &feature, int label);
		void FinalizePools(MamaGraph &myGraph);

		DadaPoolerHistogram(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaPoolerHistogram(std::shared_ptr<DadaParam> &param);
		virtual ~DadaPoolerHistogram();
	private:
		std::map<MamaVId, double> m_numV;
		//std::map<MamaEId, double> minE, maxE, numE;

}; 

#endif 

