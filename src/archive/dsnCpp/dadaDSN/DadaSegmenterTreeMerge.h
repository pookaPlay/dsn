#if !defined(DadaSegmenterTreeMerge_H__)
#define DadaSegmenterTreeMerge_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaWSGT.h"
#include "DadaSegmenterTreeNode.h"

class DadaSegmenterTreeMerge : public DadaSegmenterTreeNode
{
	public:				
		void InitGraph(std::shared_ptr<DadaSegmenterTreeNode> &myParent) override;
		void InitFeatures(std::shared_ptr<DadaSegmenterTreeNode> &myParent) override;
		void InitGT(std::shared_ptr<DadaSegmenterTreeNode> &myParent) override;
		
		void PoolParent(std::shared_ptr<DadaSegmenterTreeNode> &myParent);

		void LabelParent(std::map<MamaVId, int> &myMap) override;

		void Clear() override; 

		DadaSegmenterTreeMerge(std::shared_ptr<DadaParam> &param);
		virtual ~DadaSegmenterTreeMerge();
	
private:
	std::map<int, MamaVId>   m_labelChild; 
	std::map<MamaEId, vector<MamaEId> >   m_childParentEdge;
	

}; 

#endif 

