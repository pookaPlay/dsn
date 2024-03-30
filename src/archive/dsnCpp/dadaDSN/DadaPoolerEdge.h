#if !defined(DadaPoolerEdge_H__)
#define DadaPoolerEdge_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaPooler.h"

class DadaPoolerEdge : public DadaPooler
{
	public:				
		void PoolFeatures(std::vector<cv::Mat> &imgs, cv::Mat &basins, MamaGraph &myGraph) override;
		
		void FinalizeEdgeFeatures(MamaGraph &myGraph);

		void PoolEdge(int leftl, double leftv, int myl, double myv, MamaGraph &myGraph); 

		DadaPoolerEdge(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaPoolerEdge(std::shared_ptr<DadaParam> &param);

		virtual ~DadaPoolerEdge();
	private:
		std::map<MamaVId, double> minV, maxV, numV, numEV;
		std::map<MamaEId, double> minE, maxE, numE;

}; 

#endif 

