#if !defined(DadaSegmenterTreeSplit_H__)
#define DadaSegmenterTreeSplit_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaWSGT.h"
#include "DadaSegmenterTreeNode.h"
#include "DadaSegmenterTreeMerge.h"

class DadaSegmenterTreeSplit : public DadaSegmenterTreeNode
{
	public:				
		
		void InitGraph(std::shared_ptr<DadaSegmenterTreeNode> &myParent) override;
		void InitFeatures(std::shared_ptr<DadaSegmenterTreeNode> &myParent) override;
		void InitGT(std::shared_ptr<DadaSegmenterTreeNode> &myParent) override; 		
		void Finalize(std::shared_ptr<DadaSegmenterTreeNode> &myParent) override;
		void LabelParent(std::map<MamaVId, int> &myMap) override;

		void Clear() override; 

		DadaSegmenterTreeSplit(std::shared_ptr<DadaParam> &param);
		virtual ~DadaSegmenterTreeSplit();
	
private:
	std::map<MamaVId, MamaVId>   m_childParentVertex;	
	std::map<MamaEId, MamaEId>   m_childParentEdge;
	DadaSegmenterTreeMerge		 m_mergeNode; 
}; 

#endif 

