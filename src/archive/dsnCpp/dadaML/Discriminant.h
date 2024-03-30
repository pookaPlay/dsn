#if !defined(Discriminant_H__)
#define Discriminant_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"

#include "Logger.h"

class Discriminant 
{
	public:					
		cv::Mat myWeights; 
		float myThresh; 
		int setParam;
		Logger m_logger; 

		void InitUniform(int D);
		void InitRandom(int D);
		void InitZero(int D);

		void NormalizeMag();

		void GetOutputRange(float &minval, float &maxval);
		
		virtual void Apply(cv::Mat &mlData, cv::Mat &result); 
		virtual void Train(cv::Mat &mlData, cv::Mat &labels, cv::Mat &weights, float regularize = -1.0f);	

		void TrainThreshold(cv::Mat &result, cv::Mat &labels, cv::Mat &weights); 
		void ApplyThreshold(cv::Mat &data, cv::Mat &result, int hard = 0);
		void SetThreshold(float in) { myThresh = in; };
		
		cv::Mat & w() { return(myWeights); };
		float & t() { return(myThresh); };

		void Clear();	
		void Print(int endLine = 1);

		void SaveModel(string fname);
		void LoadModel(string fname);

		Discriminant();
		virtual ~Discriminant();
}; 


#endif 

