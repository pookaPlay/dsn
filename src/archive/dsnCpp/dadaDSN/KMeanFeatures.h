#if !defined(KMeanFeatures_H__)
#define KMeanFeatures_H__

#include "opencv2/opencv.hpp"
#include <vector>
#include "Logger.h"
#include "Whiten.h"

using namespace std;

class KMeanFeatures {
public:
	void LearnFeatures(cv::Mat &img, int numFeatures, int winSize, int useWeight);	

	void GenerateFeatures(cv::Mat &img, vector< cv::Mat > &features, string myType, int winSize);

	void NormalizeOutputFeatures(vector< cv::Mat > &features);	

	void SaveModel(string fname);
	
	void LoadModel(string fname);

	KMeanFeatures();
	virtual ~KMeanFeatures();
private:
	cv::Mat dict;
	Whiten white;
	int myUseWhite;
	Logger m_logger; 
};


#endif
