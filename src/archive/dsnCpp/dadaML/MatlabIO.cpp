#include "MatlabIO.h"
#include "MamaDef.h"
#include "Info.h"

static Info info;

#include <limits>
//#include "HistFuncs.h"
#include "MamaException.h"

using namespace cv;


void MatlabIO::LoadRCD(string fname, cv::Mat &data, cv::Mat &labels)
{
	ifstream fin(fname);
	int N, D, tempLabel;
	fin >> N;
	fin >> D;
	float tempData;
	int c1 = 0; 
	int c0 = 0;
	data = Mat::zeros(N, D, CV_32F);
	labels = Mat::zeros(N, 1, CV_32F);

	for (int i = 0; i < N; i++) {
		fin >> tempLabel;
		if (tempLabel == 1) {
			labels.at<float>(i) = 1.0f;
			c1++;
		}
		else {
			labels.at<float>(i) = tempLabel;
			c0++;
		}

		for (int d = 0; d < D; d++) {
			fin >> tempData;
			data.at<float>(i, d) = tempData;
		}
	}

	cout << "I have " << N << " samples of " << D << " dimensions: " << c1 << " =1, " << c0 << " other\n";
}


/*
void MatlabIO::SaveModel(string fname)
{
	string sname = fname + ".MatlabIO.yml";
	FileStorage fs(sname, FileStorage::WRITE);
	fs << "mean" << myMean; 
	fs << "covar" << myInvCovar; 
	fs.release();
}
	
void MatlabIO::LoadModel(string fname)
{
	myMean = Mat();
	myInvCovar= Mat();
	string sname = fname + ".MatlabIO.yml";
	FileStorage fs(sname, FileStorage::READ);
	fs["mean"] >> myMean; 
	fs["covar"] >> myInvCovar; 
	fs.release();    
}
*/

MatlabIO::MatlabIO(){	
}

MatlabIO::~MatlabIO(){
}

