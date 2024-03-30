#if !defined(DadaFeaturesHistogram_H__)
#define DadaFeaturesHistogram_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "DadaFeatures.h"


class DadaFeaturesHistogram : public DadaFeatures
{
	public:				
		void GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode = 0) override;

		void RGB2XYZ(float &sR, float &sG, float &sB, double &X, double &Y, double &Z);
		void RGB2LAB(float &sR, float &sG, float &sB, double& lval, double& aval, double& bval);		


		DadaFeaturesHistogram(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		DadaFeaturesHistogram(std::shared_ptr<DadaParam> &param);
		virtual ~DadaFeaturesHistogram();

}; 

#endif 

