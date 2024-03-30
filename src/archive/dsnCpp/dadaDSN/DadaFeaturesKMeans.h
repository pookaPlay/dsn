#if !defined(DadaFeaturesKMeans_H__)
#define DadaFeaturesKMeans_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "DadaFeatures.h"


class DadaFeaturesKMeans : public DadaFeatures
{
	public:				
		void GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode = 0) override;
	
		DadaFeaturesKMeans(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaFeaturesKMeans(std::shared_ptr<DadaParam> &param);
		virtual ~DadaFeaturesKMeans();

}; 

#endif 

