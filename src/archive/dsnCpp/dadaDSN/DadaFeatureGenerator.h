#if !defined(DadaFeatureGenerator_H__)
#define DadaFeatureGenerator_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "DadaFeatures.h"

/** 
 * provides an interface for accessing features as a linear index
 * when really they could be from different types
 **/
class DadaFeatureGenerator 
{
	public:				
		
		void InitFromBasins(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<MamaGraph> &myGraph);
		void InitSingle();
		void InitSingleEdgeScalar(MamaEId &eid, double value);

		void CalculateBaseFeatures(std::vector<cv::Mat> &imgs, int trainMode = 0); 

		void CalculateMergeFeatures(DadaFeatureGenerator &fp, MamaGraph &gp, MamaGraph &gc,
									std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge);
		
		void CalculateSplitFeatures(DadaFeatureGenerator &fgp, MamaGraph &gp, MamaGraph &gc, std::map<MamaVId, MamaVId> &parentChildVertex);

		void VizFeatureImages(int stepLast = 0, double mag = 1.0);
		void VizNodeFeatures(MamaGraph &gp, std::map<MamaVId, MamaVId> &vmap);
		void VizEdgeFeatures(MamaGraph &gp, int stepLast, double mag);

		double GetEdgeFeature(MamaEId &eid, int ind);		

		std::vector< std::shared_ptr<DadaFeatures> > & GetFeatureProcessors()
		{
			return(m_features); 
		};

		std::shared_ptr<DadaFeatures> & GetFeatureProcessor(int fi)
		{
			return(m_features[fi]);
		};

		int & D() 
		{ 
			return(m_D); 
		};		

		std::shared_ptr<DadaFeatures> CreateFeature(string featureType, int createBase = 1);		

		void Clear(); 

		void SetBasins(cv::Mat &bin) { m_basins = bin; }; 
		cv::Mat & GetBasins() { return(m_basins); }; 


		DadaFeatureGenerator(std::shared_ptr<DadaParam> &param);
		virtual ~DadaFeatureGenerator();

	protected:				
		/**
		* Configuration preferences
		**/
		std::shared_ptr<DadaParam> m_param;

		/** keep a reference
		**/
		cv::Mat m_basins;
		std::shared_ptr< std::map<int, MamaVId> > m_labelMap;
		std::shared_ptr<MamaGraph> m_baseGraph; 

		int m_D; 
		// this tracks linear index to feature object
		vector<int> m_featureIndex;
		vector<int> m_featureSubIndex;
		

		std::vector< std::shared_ptr<DadaFeatures> > m_features;


		Logger m_logger;				
}; 

#endif 

