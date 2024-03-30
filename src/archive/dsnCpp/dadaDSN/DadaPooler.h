#if !defined(DadaPooler_H__)
#define DadaPooler_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"

class DadaPooler 
{
	public:				
				
		virtual void PoolFeatures(std::vector<cv::Mat> &imgs, cv::Mat &basins, MamaGraph &myGraph);		

		virtual void MergeFeatures(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc,
			std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge);
		
		virtual void SplitFeatures(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc, std::map<MamaVId, MamaVId> &parentChildVertex);

		virtual void PoolParent(DadaPooler &pp, MamaGraph &gp, MamaGraph &gc, std::map<int, MamaVId> &labelChild);

		virtual void CalculateEdgesFromVertices(MamaGraph &gc);

		virtual void CalculateEdgeFeatureFromVertexFeatures(cv::Mat &f1, cv::Mat &f2, cv::Mat &e);
		
		void InitPools(int featureDim);
		void AddToPool(cv::Mat &feature, int label);
		void FinalizePools(MamaGraph &myGraph);

		std::map<MamaVId, cv::Mat> & GetVFeatures() { return(m_VFeatures); };
		std::map<MamaEId, cv::Mat> & GetEFeatures() { return(m_EFeatures); };
			
		cv::Mat & GetVFeature(MamaVId nid); 
		cv::Mat & GetEFeature(MamaEId eid); 
		
		std::shared_ptr< std::map<int, MamaVId> > GetVMap() { return(m_VId); };

		int & D() { return(m_ED); };

		int & VD() { return(m_VD); };

		int & GetBasinInit()
		{
			return(m_basinInit);
		};

		DadaPooler(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		
		DadaPooler(std::shared_ptr<DadaParam> &param);

		virtual ~DadaPooler();

	protected:				
		cv::Mat m_basins;		

		int m_VD; 

		int m_ED;

		// keep track of how it was initialized
		int m_basinInit;

		std::shared_ptr< std::map<int, MamaVId> > m_VId;
		std::shared_ptr< DadaParam > m_param;

		/**
		* External store for myGraph vertex data
		**/
		std::map<MamaVId, cv::Mat> m_VFeatures;
		
		/**
		* External store for myGraph edge data
		**/
		std::map<MamaEId, cv::Mat> m_EFeatures;

		Logger m_logger;				
}; 

#endif 

