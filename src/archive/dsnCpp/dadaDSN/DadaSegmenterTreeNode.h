#if !defined(DadaSegmenterTreeNode_H__)
#define DadaSegmenterTreeNode_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaWSGT.h"
#include "DadaFeatureGenerator.h"

class DadaSegmenterTreeNode
{
	public:				

		void Init(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg);
		
		void InitGT(DadaWSGT &groundTruth);

		void Init();
		void Init(std::shared_ptr<DadaSegmenterTreeNode> &myParent); 		
		virtual void InitGraph(std::shared_ptr<DadaSegmenterTreeNode> &myParent) = 0;
		virtual void InitFeatures(std::shared_ptr<DadaSegmenterTreeNode> &myParent) = 0;
		virtual void LabelParent(std::map<MamaVId, int> &myMap) = 0;
		virtual void InitGT(std::shared_ptr<DadaSegmenterTreeNode> &myParent);
		virtual void Finalize(std::shared_ptr<DadaSegmenterTreeNode> &myParent);

		virtual void Clear(); 

		void LabelGraph(std::map<MamaVId, int> &myMap);

		MamaGraph & GetGraph() {   return(*(m_graph.get()));  };

		std::shared_ptr<MamaGraph> GetGraphPtr() { return(m_graph); }; 
				
		DadaWSGT & GetGT() {  return(m_gt);	}; 
		std::shared_ptr<DadaFeatureGenerator> & GetFG() { return(m_fg); };

		std::map<MamaVId, MamaVId> & GetParentChildVertex() { return(m_parentChildVertex); };

		DadaSegmenterTreeNode(std::shared_ptr<DadaParam> &param);
		virtual ~DadaSegmenterTreeNode();
	
	protected:
		Logger m_logger;

		std::shared_ptr< MamaGraph > m_graph;
		std::shared_ptr<DadaFeatureGenerator> m_fg;

		DadaWSGT					 m_gt; 

		std::map<MamaVId, MamaVId>   m_parentChildVertex;
		std::map<MamaEId, MamaEId>   m_parentChildEdge;

		std::shared_ptr< DadaParam > m_param;

		std::shared_ptr< DadaSegmenterTreeNode > m_parent; 
		
}; 

#endif 

