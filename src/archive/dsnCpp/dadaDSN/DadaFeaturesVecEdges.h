#if !defined(DadaFeaturesVecEdges_H__)
#define DadaFeaturesVecEdges_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "DadaFeatures.h"


class DadaFeaturesVecEdges : public DadaFeatures
{
	public:				
		void GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode = 0) override;
	
		DadaFeaturesVecEdges(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaFeaturesVecEdges(std::shared_ptr<DadaParam> &param);
		virtual ~DadaFeaturesVecEdges();

}; 

#endif 

