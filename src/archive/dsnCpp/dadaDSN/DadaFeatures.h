#if !defined(DadaFeatures_H__)
#define DadaFeatures_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"
#include "DadaPooler.h"


class DadaFeatures 
{
	public:				
		virtual void GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode = 0);
		
		void MergeFeatures(std::shared_ptr<DadaFeatures> &fp, MamaGraph &gp, MamaGraph &gc,
			std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge);

		void SplitFeatures(std::shared_ptr<DadaFeatures> &fp, MamaGraph &gp, MamaGraph &gc, std::map<MamaVId, MamaVId> &parentChildVertex);

		void PoolFeatures(MamaGraph &myGraph);

		void VizFeatureImages(int stepLast = 0, double mag = 1.0);
	
		void VizNodeFeatures(int stepLast = 0, double mag = 1.0);

		void VizEdgeFeatures(MamaGraph &gp, int stepLast = 0, double mag = 1.0);

		vector<cv::Mat> & GetFeatures() { return(m_imgs); };

		cv::Mat & GetVFeature(MamaVId nid) { return(m_pool->GetVFeature(nid)); };

		cv::Mat & GetEFeature(MamaEId eid) { return(m_pool->GetEFeature(eid)); };
		
		std::map<MamaEId, cv::Mat> & GetEFeatures() { return(m_pool->GetEFeatures()); };

		std::map<MamaVId, cv::Mat> & GetVFeatures() { return(m_pool->GetVFeatures()); };

		static int CalculateEdgeFeatureFromVertexFeatures(cv::Mat &f1, cv::Mat &f2, cv::Mat &e);

		int & D() { return(m_pool->D()); };

		string & GetType() 
		{
			return(m_type); 
		};

		DadaPooler * GetPooler() 
		{
			return(m_pool.get()); 
		}

		DadaFeatures(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param);
		
		DadaFeatures(std::shared_ptr<DadaParam> &param);

		virtual ~DadaFeatures();

	protected:				
		/**
		 * Store raw images here
		 **/
		vector<cv::Mat> m_imgs;

		/** keep a reference
		**/
		cv::Mat m_basins; 
		/**
		 * reference to parameters
		 **/
		std::shared_ptr<DadaParam> m_param;

		/**
		 * Pooler object so that it can be subclassed
		 **/
		std::unique_ptr<DadaPooler> m_pool;
		
		string m_type; 

		Logger m_logger;				
}; 

#endif 

