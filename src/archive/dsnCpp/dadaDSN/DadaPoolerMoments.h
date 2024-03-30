#if !defined(DadaPoolerMoments_H__)
#define DadaPoolerMoments_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaPooler.h"

class DadaPoolerMoments : public DadaPooler
{
	public:				
		void PoolFeatures(std::vector<cv::Mat> &imgs, cv::Mat &basins, MamaGraph &myGraph) override;
		
		void InitPools(int featureDim);
		void AddToPool(cv::Mat &feature, int label);
		void FinalizePools(MamaGraph &myGraph);

		DadaPoolerMoments(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaPoolerMoments(std::shared_ptr<DadaParam> &param);
		virtual ~DadaPoolerMoments();
	private:
		std::map<MamaVId, double> m_numV;
		//std::map<MamaEId, double> minE, maxE, numE;

}; 

#endif 

