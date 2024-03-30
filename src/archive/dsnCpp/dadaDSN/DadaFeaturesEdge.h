#if !defined(DadaFeaturesEdge_H__)
#define DadaFeaturesEdge_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "DadaFeatures.h"


class DadaFeaturesEdge : public DadaFeatures
{
	public:				
		void GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode = 0) override;
	
		DadaFeaturesEdge(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaFeaturesEdge(std::shared_ptr<DadaParam> &param);
		virtual ~DadaFeaturesEdge();

}; 

#endif 

