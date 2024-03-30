#if !defined(MatlabIO_H__)
#define MatlabIO_H__

#include "opencv2/opencv.hpp"
#include <vector>
using namespace std;

class MatlabIO {
public:

	static void LoadRCD(string fname, cv::Mat &data, cv::Mat &labels);

	MatlabIO();
	virtual ~MatlabIO();
	
};


#endif
