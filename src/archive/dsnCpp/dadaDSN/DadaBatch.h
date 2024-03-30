#if !defined(DadaBatch_H__)
#define DadaBatch_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "SegmentWS.h"
#include "DadaFeatures.h"
#include "DadaParam.h"
#include "DadaSegmenter.h"
#include "DadaWS.h"

class DadaBatch
{
	public:		
		void Train(int trainMode);
		double TrainThresholds(int trainMode);

		void Eval(int trainMode, int numImages);
		void LoadImage(int imgi, int trainMode);

		void Load(int index, int trainMode, int viz = 0);
		void GenerateBasinsFromGray();
		void GenerateBasinsWithSLIC(int viz = 0);
		void InitParam(int expi); 
		
		void RunBaseline(int trainMode = 1);

		void SetTrainFiles(vector<string> &imgnames, vector<string> &segnames)
		{
			m_trainImgNames = imgnames; 
			m_trainSegNames = segnames;
		};

		void SetTestFiles(vector<string> &imgnames, vector<string> &segnames)
		{
			m_imgNames = imgnames;
			m_segNames = segnames;
		};
		vector<double> m_exp1Train;
		vector<double> m_exp1Test;
		std::shared_ptr<DadaParam> m_param;
		int m_numImages;
		int m_numTestImages;
		int m_numTreesPerImage;
		std::vector<int> m_trainIds; 
		DadaBatch();
		virtual ~DadaBatch();
	protected:					
		Logger m_logger;				
		vector<string> m_imgNames; 
		vector<string> m_segNames;

		vector<string> m_trainImgNames;
		vector<string> m_trainSegNames;		

		
		std::shared_ptr<DadaWS> m_dada; 
		
		vector<double> m_trainThresholds;
		vector<double> m_trainErrors;
		vector<double> m_finalTrainErrors;
		vector<double> m_finalTestErrors;

		map<int, int> m_badTrain, m_badTest;

		cv::Mat m_inputSeg;
		cv::Mat m_inputGray;
		cv::Mat m_inputColor;
		cv::Mat m_basins;
		vector<cv::Mat> m_inputStack; 

}; 

#endif 

