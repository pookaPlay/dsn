#include "SynData.h"
#include "Info.h"
#include "MamaDef.h"
#include "NormalDataset.h"

#include "opencv2/opencv.hpp"
//#include "cv.h"
using namespace cv;
//using namespace std;
static Info info;


void SynData::Syn2DGaussian(int N, double myTerm, double crossTerm, vector< vector<float> > &mlData)
{
	int D = 2;
	double *mu = new double[D];
	double *a = new double[D*D];

	for (uint i = 0; i< D; i++) mu[i] = 0.0;
	for (uint i = 0; i< D*D; i++) a[i] = 0.0;
	a[0] = myTerm; 	
	a[1] = crossTerm;
	a[2] = crossTerm;
	a[3] = myTerm;
	
	NormalDataset::GenerateSamples(N, D, a, mu, 1, mlData);

}



void SynData::SynGaussianData(int N, int D, vector< vector<float> > &mlData, vector<float> &labels)
{
	double *mu = new double[D];
	double *a = new double[D*D];
	
	for(uint i=0; i< D; i++) mu[i] = 0.0;
	for(uint i=0; i< D*D; i++) a[i] = 0.0;
	for(uint i=0; i< D; i++) a[i*D+i] = 1.0;
	NormalDataset::GenerateSamples(N, D, a, mu, 1, mlData); 

}

void SynData::SynTwoClassData(int N, int D, vector< vector<float> > &mlData, vector<float> &labels, float sigma)
{
	double *mu = new double[D];
	double *mu2 = new double[D];
	double *a = new double[D*D];
	
	//for(uint i=0; i< D; i++) mu[i] = 5.0;
	//mu[1] = 0.0;
	for(uint i=0; i< D; i++) mu[i] = 1.0;
	//for(uint i=0; i< D; i++) mu2[i] = 1.0;
	//mu2[1] = 3.0;
	for(uint i=0; i< D; i++) mu2[i] = 3.0;

	for(uint i=0; i< D*D; i++) a[i] = 0.0;
	for(uint i=0; i< D; i++) a[i*D+i] = sigma;

	NormalDataset::GenerateSamples(N, D, a, mu, 1, mlData); 
	labels.clear();
	labels.resize(N, -1.0f);

	vector< vector<float> > data2;
	NormalDataset::GenerateSamples(N, D, a, mu2, 1, data2); 

	for(uint i=0; i< data2.size(); i++) {
		mlData.push_back( data2[i] );
		labels.push_back( 1.0f );
	}

}

void SynData::SynTwoClassThreshold(int N, int D, vector< vector<float> > &mlData, vector<float> &labels, float sigma)
{
	double *mu = new double[D];
	double *mu2 = new double[D];
	double *a = new double[D*D];
	double *a2 = new double[D*D];
	
	for(uint i=0; i< D; i++) mu[i] = -2.0;
	for(uint i=0; i< D; i++) mu2[i] = 0.0;
	for(uint i=0; i< D*D; i++) {
		a[i] = 0.0; a2[i] = 0.0;
	}
	for(uint i=0; i< D; i++) {
		a[i*D+i] = sigma; a2[i*D+i] = sigma;
	}
	a2[1] = sigma/2.0f;
	a2[3] = a2[1]; 
	NormalDataset::GenerateSamples(N/4, D, a, mu, 1, mlData); 
	//for(uint i=0; i< mlData.size(); i++) {
	//	mlData[i].push_back(1.0f);
	//}
	labels.clear();
	labels.resize(mlData.size(), -1.0f);

	vector< vector<float> > data2;
	NormalDataset::GenerateSamples(N, D, a2, mu2, 1, data2); 
	
	for(uint i=0; i< data2.size(); i++) {
		//data2[i].push_back(1.0f);
		mlData.push_back( data2[i] );
		labels.push_back( 1.0f );
	}

}
void SynData::PopCode(vector< vector<float> > &mlData, vector< vector<float> > &mlDataOut, float minVal, float maxVal, int numCount)
{	
	float spac = (maxVal - minVal) / (float) (numCount - 1);
	int newsz = mlData[0].size()*numCount;
	mlDataOut.clear();
	mlDataOut.resize( mlData.size() );
	for(uint i=0; i< mlData.size(); i++) {
		mlDataOut[i].clear();
		mlDataOut[i].resize( newsz, 0.0f );
		for(uint ii=0; ii< mlData[i].size(); ii++) {			
			for(uint f=0; f< numCount; f++) {
				float meanVal = minVal + (float) f * spac;
				float fracVal = exp(-(mlData[i][ii] - meanVal) * (mlData[i][ii] - meanVal) / spac );					
				mlDataOut[i][f + ii*numCount] = fracVal;
			}
		}
	}	
}

void SynData::PlotData(vector< vector<float> > &mlData, vector<float> &labels, int imgSize, int pointScale)
{
	Mat img; 
	img.create(imgSize, imgSize, CV_8UC3);
	for(unsigned int i=0; i< img.rows * img.cols * 3; i++) {
		*(img.data+i) = (unsigned char) 0; 
	}	
	for(unsigned int i=0; i< mlData.size(); i++) {		
		CvPoint p1 = cvPoint(mlData[i][0] * pointScale + imgSize/2, mlData[i][1] * pointScale + imgSize / 2);

		if (labels.size() == mlData.size()) {
			if (labels[i] > 0.0f) {
				circle(img, p1, 1, Scalar(0, 0, 255), -1, 8, 0);		
			} else {
				circle(img, p1, 1, Scalar(0, 255, 0), -1, 8, 0);		
			}

		} else circle(img, p1, 1, Scalar(0, 255, 255), -1, 8, 0);		
	}
	namedWindow("Data");
	imshow("Data", img);
	waitKey();
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
SynData::SynData() 
{	
}

SynData::~SynData()
{	
}


