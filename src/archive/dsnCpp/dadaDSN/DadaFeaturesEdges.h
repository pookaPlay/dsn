#if !defined(DadaFeaturesEdges_H__)
#define DadaFeaturesEdges_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "DadaFeatures.h"


class DadaFeaturesEdges : public DadaFeatures
{
	public:				
		void GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode = 0) override;
	
		DadaFeaturesEdges(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaFeaturesEdges(std::shared_ptr<DadaParam> &param);
		virtual ~DadaFeaturesEdges();

}; 

#endif 

