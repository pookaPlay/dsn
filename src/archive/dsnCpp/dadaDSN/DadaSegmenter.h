#if !defined(DadaSegmenter_H__)
#define DadaSegmenter_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaClassifier.h"
#include "DadaClassifierParam.h"
#include "DadaFeatureGenerator.h"

class DadaWS;
class DadaWSACD;

class DadaSegmenter 
{
	public:				
		virtual void Init(std::shared_ptr<DadaFeatureGenerator> &fg);
		virtual void Save();
		virtual void Load();

		virtual void Apply(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg);
		virtual void Train(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth);
		virtual void UpdateOutputThreshold(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, double threshold);
		virtual void SetOutputThreshold(double threshold);
		virtual void Evaluate(std::shared_ptr<MamaGraph> &myGraph, DadaWSGT &gt);

		virtual void TrainInit(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth);
		virtual void TrainStep(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth);
		virtual void TrainFinish(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth);
		virtual void TrainThreshold(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth);

		virtual void InitLabelMap(std::shared_ptr<MamaGraph> &myGraph);
		virtual void InitEdgeMap(std::shared_ptr<MamaGraph> &myGraph);
		virtual std::map<MamaVId, int> & GetLabelMap(int index = -1);
		virtual std::map<MamaEId, double> & GetEdgeMap(int index = -1);

		virtual std::shared_ptr< DadaClassifierParam > GetClassParam(int index = -1);
		virtual std::shared_ptr<DadaFeatureGenerator> GetFeatureGenerator();

		void SetClassParam(std::shared_ptr< DadaClassifierParam > aParam) { 
			m_classParam = aParam;  
			m_class->SetParam(aParam); 
		};
	
		DadaError & GetError() { return(m_error); };

		void GetOutputThresh(double &outputMin, double &outputMax, double &outputThreshold) {
			outputMin = m_outputMin;
			outputMax = m_outputMax;
			outputThreshold = m_outputThreshold;
		};

		void SetOutputThresh(double outputThreshold) {
			m_outputThreshold = outputThreshold; 
		};

		void SetDada(DadaWS *aDada) {
			m_dada = aDada;
		}

		void SetACD(DadaWSACD *acd)
		{
			m_acd = acd;
		};

		int GetNumLabels() {
			return(m_numLabels);
		};

		std::map<MamaVId, MamaVId> & GetVertexMap() {
			return(m_vertexMap);
		};
		

		DadaSegmenter(std::shared_ptr<DadaParam> &param);
		virtual ~DadaSegmenter();

	protected:		
		DadaWS *m_dada;
		DadaWSACD *m_acd;

		std::map<MamaVId, int> m_labelMap;
		std::map<MamaVId, MamaVId> m_vertexMap;
		std::map<MamaEId, double> m_edgeMap;

		std::shared_ptr< DadaParam > m_param;
		
		Logger m_logger;				
		int m_numLabels;
		
		DadaError m_error;		

		double m_outputMin;
		double m_outputMax;
		double m_outputThreshold;
		
		std::shared_ptr<DadaFeatureGenerator> m_features; 
		std::shared_ptr<DadaClassifierParam> m_classParam;
		std::unique_ptr< DadaClassifier > m_class;		
}; 

#endif 

