#if !defined(Whiten_H__)
#define Whiten_H__

#include "opencv2/opencv.hpp"
#include <vector>
using namespace std;
#include "Logger.h"

class Whiten {
public:
	void Estimate(cv::Mat &features);
	void Apply(cv::Mat &features, int addThresh = 0);

	void SaveModel(string fname);	
	void LoadModel(string fname);

	cv::Mat & M() { return(this->myMean); };
	cv::Mat & IC() { return(this->myInvCovar); };

	Whiten();
	virtual ~Whiten();
	
private:
	cv::Mat myMean;
	cv::Mat myInvCovar;

	Logger m_logger;
};


#endif
