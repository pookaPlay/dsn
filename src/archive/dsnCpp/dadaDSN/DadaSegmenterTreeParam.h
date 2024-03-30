#if !defined(DadaSegmenterTreeParam_H__)
#define DadaSegmenterTreeParam_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "DadaClassifierParam.h"
#include "SegmentWS.h"

class DadaSegmenterTreeParam
{
	public:						
		void Init(); 
		void AddRootNode(int nid, int D);

		void AddChild(int pid, int nid, int typei, int D);
		void RemoveChild(int pid, int nid);

		int GetNumNodes() { return(m_nodeParams.size()); };

		std::shared_ptr<DadaClassifierParam> GetNode(int nid);

		int GetNodeType(int nid); 

		std::map<int, std::shared_ptr<DadaClassifierParam>> & GetNodeMap() {
			return(m_nodeParams);
		};

		int HasChildren(int nid) { return(m_children.count(nid)); }; 

		std::vector<int> & GetChildren(int nid) {			
			return(m_children[nid]);
		}; 

		std::map<int, int> & GetParentMap() {
			return(m_parent);
		}

		void SetValidFeatures(std::vector<int> &findex) { m_validFeatures = findex; }; 		
		std::vector<int> & GetValidFeatures() { return(m_validFeatures); }; 

		void Print();
		void CopyTo(std::shared_ptr<DadaSegmenterTreeParam> &dest);

		void Save();
		void Load();
		void Save(string fname);
		void Load(string fname);
		void Save(cv::FileStorage &fs) const;
		void Load(cv::FileStorage &fs);
		void Load(cv::FileNodeIterator &fs);

		DadaSegmenterTreeParam(std::shared_ptr<DadaParam> &param);
		virtual ~DadaSegmenterTreeParam();
	
	protected:		
		std::vector<int> m_validFeatures; 
		
		std::map<int, int> m_nodeType; 
		std::map<int, std::shared_ptr<DadaClassifierParam>> m_nodeParams;

		int m_rootNode;
		std::map<int, vector<int> > m_children;
		std::map<int, int> m_parent;
						
		std::shared_ptr< DadaParam > m_param;
		Logger m_logger;
}; 

#endif 

