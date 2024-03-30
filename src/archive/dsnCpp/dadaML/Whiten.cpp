#include "Whiten.h"
#include "MamaDef.h"
#include "Info.h"

static Info info;

#include <limits>
//#include "HistFuncs.h"
#include "MamaException.h"

using namespace cv;


void Whiten::Estimate(Mat &features)
{
	Mat myCovar;
	calcCovarMatrix(features, myCovar, myMean, CV_COVAR_NORMAL + CV_COVAR_ROWS + CV_COVAR_SCALE, CV_32F); 
	invert(myCovar, myInvCovar, DECOMP_SVD);	
}


void Whiten::Apply(Mat &features, int addThresh)
{		
	if (addThresh) {
		Mat oldFeatures = features.clone();
		features = Mat::zeros(oldFeatures.rows, oldFeatures.cols+1, CV_32F);
		for(uint i=0; i< oldFeatures.rows; i++) {				
			Mat mt = oldFeatures.row(i);
			Mat ms = mt - myMean;
			Mat nv = ms*myInvCovar;
			Mat nvt = nv.t();
			nvt.push_back(1.0f);
			nv = nvt.t();
			//features.row(i) = nv + 0; 
			nv.copyTo(features.row(i)); 	
		}

	} else {
		for(uint i=0; i< features.rows; i++) {				
			Mat mt = features.row(i);
			Mat ms = mt - myMean;
			Mat nv = ms*myInvCovar;
			//features.row(i) = nv + 0; 
			nv.copyTo(features.row(i)); 	
		}
	}
}

void Whiten::SaveModel(string fname)
{
	string sname = fname + ".whiten.yml";
	FileStorage fs(sname, FileStorage::WRITE);
	fs << "mean" << myMean; 
	fs << "covar" << myInvCovar; 
	fs.release();
}
	
void Whiten::LoadModel(string fname)
{
	myMean = Mat();
	myInvCovar= Mat();
	string sname = fname + ".whiten.yml";
	FileStorage fs(sname, FileStorage::READ);
	fs["mean"] >> myMean; 
	fs["covar"] >> myInvCovar; 
	fs.release();    
}


Whiten::Whiten(){	
}

Whiten::~Whiten(){
}

