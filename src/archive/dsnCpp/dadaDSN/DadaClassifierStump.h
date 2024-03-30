#if !defined(DadaClassifierStump_H__)
#define DadaClassifierStump_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaClassifierParam.h"
#include "DadaClassifier.h"
#include "DadaEval.h"

class DadaClassifierStump  : public DadaClassifier
{
	public:				
		void TrainInit() override;
		void TrainIteration(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &groundTruth) override;
		void TrainFinalize() override; 
				
		double ApplyNoThreshold(MamaEId &mid, DadaFeatureGenerator &fg) override;
		double ApplyThreshold(double &feature) override;

		DadaClassifierStump(std::shared_ptr<DadaParam> param, std::shared_ptr<DadaClassifierParam> classParam);
		virtual ~DadaClassifierStump();

	protected:	
		double m_bestThreshold;
		int m_bestIndex; 
		int m_currentIndex; 
		string m_bestSegType;

		DadaError m_bestError;

}; 

#endif 

