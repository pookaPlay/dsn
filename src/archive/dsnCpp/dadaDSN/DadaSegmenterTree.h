#if !defined(DadaSegmenterTree_H__)
#define DadaSegmenterTree_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaSegmenterForestParam.h"
#include "DadaSegmenter.h"
#include "DadaSegmenterTreeNode.h"

class DadaSegmenterTree  : public DadaSegmenter
{
	public:				
		void Init(std::shared_ptr<DadaFeatureGenerator> &fg) override;
		void Save() override;
		void Load() override;

		void Apply(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg) override;
		void Train(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth) override;
		void InitLabelMap(std::shared_ptr<MamaGraph> &myGraph) override;		

		void Evaluate(std::shared_ptr<MamaGraph> &myGraph, DadaWSGT &gt) override;

		void AddNode(int pid, int cid, string myType);
		void RemoveNode(int pid, int cid); 

		void SetTreeParam(std::shared_ptr< DadaSegmenterTreeParam > treeParam)
		{ 
			m_treeParam = treeParam; 			
		}; 
				
		std::shared_ptr<DadaFeatureGenerator> GetFeatureGenerator() override; 


		DadaSegmenterTree();
		DadaSegmenterTree(std::shared_ptr<DadaParam> &param);
		virtual ~DadaSegmenterTree();
	
	protected:		
		std::shared_ptr< DadaSegmenterTreeParam > m_treeParam;
		std::map<int, std::shared_ptr< DadaSegmenterTreeNode > > m_treeNodes;		

}; 

#endif 

