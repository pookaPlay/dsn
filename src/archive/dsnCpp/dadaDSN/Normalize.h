#if !defined(Normalize_H__)
#define Normalize_H__

#include "opencv2/opencv.hpp"
#include <vector>
using namespace std;
#include "Logger.h"

class Normalize {
public:
	void EstimateInit(int D);
	void EstimateAdd(cv::Mat &rowVec);
	void EstimateFinalize();
	void Apply(cv::Mat &rowVec);

	void SaveModel(string fname);	
	void LoadModel(string fname);

	void InitCheckRange(int D);
	void CheckRangeRow(cv::Mat &rowVec);
	void FinalizeCheckRange();

	Normalize();
	virtual ~Normalize();
	
private:
	cv::Mat myMean;
	cv::Mat myVar;
	cv::Mat myMin;
	cv::Mat myMax;
	cv::Mat myDiv;
	cv::Mat tempMin;
	cv::Mat tempMax;
	int myCount;
	Logger m_logger;
};


#endif
