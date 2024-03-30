#if !defined(DadaWS_H__)
#define DadaWS_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "SegmentWS.h"
#include "DadaFeatureGenerator.h"
#include "DadaParam.h"
#include "DadaSegmenter.h"
#include "DadaWSGT.h"

class DadaWS 
{
	public:		
		
		void InitFromBasins(cv::Mat &basins);

		void InitFeatures(std::vector<cv::Mat> &imgs, int trainMode = 0);
		void InitFeatures(cv::Mat &img, int trainMode = 0);	
		
		void InitGroundTruth(cv::Mat &desiredSeg);

		void InitSegmenter();

		void TrainSegmenter();
		void TrainInit();
		void TrainStep();
		void TrainFinish();
		void TrainThreshold();

		DadaError & EvalSegmenterMST();

		DadaError EvalSegmenterMat();

		void ApplySegmenter();

		void UpdateOutputThreshold(double threshold);

		void SetOutputThreshold(double threshold);

		void GetOutputThresh(double &outputMin, double &outputMax, double &outputThreshold) {
			m_seg->GetOutputThresh(outputMin, outputMax, outputThreshold); 
		};

		void Save();

		void Load();

		void UpdateTrainViz(); 		

		void GetImgLabels(cv::Mat &mySeg, int index = -1);

		void GetImgEdges(cv::Mat &mySeg, int index = -1);

		std::shared_ptr<DadaClassifierParam> GetClassParam(int index = -1) {
			return(m_seg->GetClassParam(index)); 
		};

		std::shared_ptr<DadaFeatureGenerator> Features() { return(m_features); };

		DadaError & GetError() { return(m_seg->GetError()); };

		std::shared_ptr<MamaGraph> & GetGraph() {
			return(m_myGraph);
		}

		std::map<MamaVId, int> & GetLabelMap(int index = -1) {
			return(m_seg->GetLabelMap(index));
		}

		std::shared_ptr< std::map<int, MamaVId> > GetBasinMap()
		{
			return(m_basinVMap);
		}

		int GetNumLabels()
		{
			return(m_seg->GetNumLabels()); 
		};

		void VizNodeFeatures(); 		

		DadaWS(std::shared_ptr<DadaParam> &param);
		virtual ~DadaWS();
	protected:				
		/**
		* Configuration preferences
		**/
		std::shared_ptr<DadaParam> m_param; 
		/**
		* This holds the learnable model
		**/
		std::unique_ptr<DadaSegmenter> m_seg;

		/**
		* Just convienient to keep around
		**/
		cv::Mat m_basins, m_desired;
		int m_w, m_h;

		/** 
		* This is a second layer graph that gets trained
		**/
		std::shared_ptr<MamaGraph> m_myGraph;

		/**
		* This is a second layer graph that gets trained
		**/
		std::shared_ptr<DadaFeatureGenerator> m_features;

		/**
		* Maps basin labels to vertixes in m_myGraph
		**/
		std::shared_ptr< std::map<int, MamaVId> > m_basinVMap;

		/**
		* The basin label associated with vertixes in m_myGraph
		**/
		std::map<MamaVId, int> m_basinLabel;
		
		/**
		* Ground truth
		**/
		DadaWSGT m_gt; 

		Logger m_logger;				
}; 

#endif 

