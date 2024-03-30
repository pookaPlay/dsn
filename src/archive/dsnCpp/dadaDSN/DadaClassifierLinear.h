#if !defined(DadaClassifierLinear_H__)
#define DadaClassifierLinear_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaClassifierParam.h"
#include "DadaClassifier.h"

class DadaClassifierLinear  : public DadaClassifier
{
	public:				
		
		void TrainInit() override; 
		void TrainIteration(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &groundTruth) override;
		void TrainFinalize() override;
		
		double ApplyNoThreshold(MamaEId &mid, DadaFeatureGenerator &fg) override;
		double ApplyThreshold(double &feature) override;

		void UpdateWeight(cv::Mat &gradient, double weight);

		void NormalizeWeight();

		DadaClassifierLinear(std::shared_ptr<DadaParam> param, std::shared_ptr<DadaClassifierParam> classParam);
		virtual ~DadaClassifierLinear();

	protected:		

		cv::Mat m_bestW;
		double m_bestThreshold;
		double m_bestError;
		int m_bestIteration;

}; 

#endif 

