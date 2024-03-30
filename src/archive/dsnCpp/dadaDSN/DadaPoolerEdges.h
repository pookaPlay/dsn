#if !defined(DadaPoolerEdges_H__)
#define DadaPoolerEdges_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaPooler.h"

class DadaPoolerEdges : public DadaPooler
{
	public:				
		void PoolFeatures(std::vector<cv::Mat> &imgs, cv::Mat &basins, MamaGraph &myGraph) override;

		void MergeFeatures(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc,
			std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge) override; 

		void FinalizeEdgeFeatures(MamaGraph &myGraph);

		void PoolEdge(int leftl, vector<double> &leftv, int myl, vector<double> &myv, MamaGraph &myGraph);

		DadaPoolerEdges(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaPoolerEdges(std::shared_ptr<DadaParam> &param);
		virtual ~DadaPoolerEdges();
	private:
		std::map<MamaVId, vector<double> > minV, maxV, accV; 
		std::map<MamaVId, double> numV;
		std::map<MamaEId, vector<double> > minE, maxE, accE; 
		std::map<MamaEId, double> numE; 

}; 

#endif 

