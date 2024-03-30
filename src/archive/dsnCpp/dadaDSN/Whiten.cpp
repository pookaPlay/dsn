/*
* Unless otherwise indicated, this software has been authored by an
* employee of Los Alamos National Security LLC (LANS), operator of the
* Los Alamos National Laboratory under Contract No. DE-AC52-06NA25396 with
* the U.S. Department of Energy.
*
* The U.S. Government has rights to use, reproduce, and distribute this information.
* Neither the Government nor LANS makes any warranty, express or implied, or
* assumes any liability or responsibility for the use of this software.
*
* Distribution of this source code or of products derived from this
* source code, in part or in whole, including executable files and
* libraries, is expressly forbidden.
*
* Funding was provided by Laboratory Directed Research and Development.
*/

#include "Whiten.h"
#include "MamaDef.h"

#include <limits>
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
		for(int i=0; i< oldFeatures.rows; i++) {				
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
		for(int i=0; i< features.rows; i++) {				
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


Whiten::Whiten()
	: m_logger(LOG_GET_LOGGER("Whiten"))
{	
}

Whiten::~Whiten(){
}

