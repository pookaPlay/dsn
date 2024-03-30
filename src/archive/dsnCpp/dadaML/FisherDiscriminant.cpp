#include "FisherDiscriminant.h"
#include "Info.h"

using namespace std;
static Info info;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"

void FisherDiscriminant::Train(Mat &mlData, Mat &labels, Mat &weights, float regularize)
{	
	int D = mlData.cols;
	int N = mlData.rows;

	Mat m1 = Mat::zeros(1, D, CV_32F);
	Mat m0 = Mat::zeros(1, D, CV_32F); 
	float c1 = 0.0f, c0 = 0.0f;	

	for(int ni=0; ni< N; ni++) {
		if ( labels.at<float>(ni) > 0.0f ) {
			Mat tempr = mlData.row(ni) * weights.at<float>(ni);
			m1 += tempr;
			c1 += weights.at<float>(ni);
		} else if ( labels.at<float>(ni) < 0.0f ) {
			Mat tempr = mlData.row(ni) * weights.at<float>(ni);
			m0 += tempr; 
			c0 += weights.at<float>(ni);
		}
	}
	m1 = m1 / c1;
	m0 = m0 / c0;
	Mat mdiff = m1 - m0;
	
	//info(1, "Mean 1: "); cout << m1 << "\n";
	//info(1, "Mean 0: "); cout << m0 << "\n";

	Mat meanAdj = Mat(); 

	Mat covar = Mat::zeros(D, D, CV_32F); 

	for (int ni = 0; ni < N; ni++) {
		if ( labels.at<float>(ni) > 0.0f ) {
			meanAdj = mlData.row(ni) - m1;
		} else if ( labels.at<float>(ni) < 0.0f ) {
			meanAdj = mlData.row(ni) - m0;			
		}
		if ( labels.at<float>(ni) != 0.0f ) {
			for (int di = 0; di < D; di++) {
			for (int dii = 0; dii < D; dii++) {
				float tempf = meanAdj.at<float>(di) * meanAdj.at<float>(dii) * weights.at<float>(ni);
				covar.at<float>(di, dii) += tempf;
			}
			}
		}
	}

	covar = covar / (c1+c0);
	//info(1, "Covar!\n"); cout << covar << "\n";
	
	this->myWeights = Mat::zeros(D, 1, CV_32F);
	
	//info(1, "Mean Diff "); cout << mdiff << "\n";
	
	solve(covar, mdiff.t(), this->myWeights, DECOMP_SVD);
	//info(1, "Weights: "); cout << this->myWeights.t() << "\n";
	this->setParam = 0;
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
FisherDiscriminant::FisherDiscriminant() 
{		
}

FisherDiscriminant::~FisherDiscriminant()
{	
}
