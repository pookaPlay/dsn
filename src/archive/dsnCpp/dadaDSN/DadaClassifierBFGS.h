#if !defined(DadaClassifierBFGS_H__)
#define DadaClassifierBFGS_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaClassifierParam.h"
#include "DadaClassifier.h"
#include "DadaWSGT.h"

#include "OWLQN.h"

class DadaClassifierBFGS;

struct SegObjective : public DifferentiableFunction {
	DadaClassifierBFGS *problem;	
	double Eval(const DblVec& input, DblVec& gradient);	
};

class DadaClassifierBFGS  : public DadaClassifier
{
	public:				
		
		void TrainInit() override; 
		void TrainIteration(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &groundTruth) override;
		void TrainFinalize() override;

		double ApplyNoThreshold(MamaEId &mid, DadaFeatureGenerator &fg) override;
		double ApplyThreshold(double &feature) override;

		
		double EstimateGradient(const std::vector<double> &current, std::vector<double> &gradient);
		double EstimateGradient(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &groundTruth);

		DadaClassifierBFGS(std::shared_ptr<DadaParam> param, std::shared_ptr<DadaClassifierParam> classParam);
		virtual ~DadaClassifierBFGS();

	protected:		

		MamaGraph *m_myGraph; 
		std::map<MamaEId, cv::Mat> *m_features; 
		DadaWSGT *m_gt; 

		cv::Mat m_gradient;		
		double m_gradientThresh;

		cv::Mat m_bestW;
		double m_bestThreshold;
		double m_bestError;
		int m_bestIteration;
		struct SegObjective *m_obj;		
}; 

#endif 

