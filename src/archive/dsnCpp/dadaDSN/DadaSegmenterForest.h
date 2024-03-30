#if !defined(DadaSegmenterForest_H__)
#define DadaSegmenterForest_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaClassifierStump.h"
#include "DadaSegmenterTreeParam.h"
#include "DadaSegmenterTree.h"
#include "DadaWSGT.h"

class DadaSegmenterForest  : public DadaSegmenterTree
{
	public:				
		void Init(std::shared_ptr<DadaFeatureGenerator> &fg) override;
		void Save() override;
		void Load() override;

		void Apply(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg) override;
		void Train(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth) override;
		void UpdateOutputThreshold(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, double threshold) override;
		void SetOutputThreshold(double threshold) override; 
		void Evaluate(std::shared_ptr<MamaGraph> &myGraph, DadaWSGT &gt) override;

		void TrainInit(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth) override;
		void TrainStep(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth) override;
		void TrainFinish(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth) override;
		void TrainThreshold(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth) override;

		void InitLabelMap(std::shared_ptr<MamaGraph> &myGraph) override;
		std::map<MamaVId, int> & GetLabelMap(int index) override; 

		void SubSample(DadaWSGT &gt, DadaWSGT &sample);
		void TrainVote(std::shared_ptr<MamaGraph> &myGraph, DadaWSGT &groundTruth);
		void ApplyVote(std::shared_ptr<MamaGraph> &myGraph);		
		void GetVoteData(std::shared_ptr<MamaGraph> &myGraph);
		
		DadaError GetTreeError() { return(m_treeError); };
		
		void Clear();
		DadaSegmenterForest(std::shared_ptr<DadaParam> &param);
		virtual ~DadaSegmenterForest();
	
	protected:				
		DadaError m_treeError; 
		int m_treeIndex;		
		
		DadaFeatureGenerator m_voteData;

		std::unique_ptr< DadaClassifierStump > m_vote;				

		std::shared_ptr< DadaSegmenterForestParam > m_forest;
		
		std::vector< std::map<MamaVId, int> > m_treeLabels; 

		std::vector< DadaError > m_treeErrors;
}; 

#endif 

