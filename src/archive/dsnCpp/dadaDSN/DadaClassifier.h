#if !defined(DadaClassifier_H__)
#define DadaClassifier_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaClassifierParam.h"
#include "DadaEval.h"
#include "DadaFeatureGenerator.h"

class DadaClassifier 
{
	public:				
				
		void Apply(MamaGraph &myGraph, DadaFeatureGenerator &fg);
		void Evaluate(MamaGraph &myGraph, DadaWSGT &gt);

		virtual double ApplyNoThreshold(MamaEId &mid, DadaFeatureGenerator &fg);
		virtual double ApplyThreshold(double &feature);

		virtual void TrainInit() {};
		virtual void TrainIteration(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &groundTruth) {};
		virtual void TrainFinalize() {};
				
		int SawImprovement() { return(m_sawImprovment); };
		int TrainDone() { return(m_trainDone); };

		void SetParam(std::shared_ptr< DadaClassifierParam > aParam) {
			m_classParam = aParam; 
		};		

		std::shared_ptr< DadaClassifierParam > GetParam() {
			return(m_classParam); 
		};

		DadaError & GetError() { return(m_error); };

		DadaClassifier(std::shared_ptr<DadaParam> param, std::shared_ptr<DadaClassifierParam> classParam);
		virtual ~DadaClassifier();

	protected:				
		/**
		 * This tracks the minimax edge for each primary edge
		 **/
		map<MamaEId, MamaEId> m_watershedNeighbors; 
		
		DadaEval m_eval;

		std::shared_ptr< DadaParam > m_param;
		std::shared_ptr< DadaClassifierParam > m_classParam;

		DadaError m_error; 

		int m_sawImprovment;
		int m_trainDone;

		Logger m_logger;				
}; 

#endif 

