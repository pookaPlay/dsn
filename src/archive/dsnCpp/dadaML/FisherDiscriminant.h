#if !defined(FisherDiscriminant_H__)
#define FisherDiscriminant_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"

#include "Discriminant.h"

class FisherDiscriminant : public Discriminant
{
	public:					
		virtual void Train(cv::Mat &mlData, cv::Mat &labels, cv::Mat &weights, float regularize = -1.0f);

		FisherDiscriminant();
		virtual ~FisherDiscriminant();
}; 


#endif 

