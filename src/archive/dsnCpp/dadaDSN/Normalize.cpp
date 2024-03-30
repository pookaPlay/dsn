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

#include "Normalize.h"
#include "MamaDef.h"

#include <limits>
#include "MamaException.h"

using namespace cv;

void Normalize::EstimateInit(int D)
{
	myCount = 0; 
	myMean = Mat::zeros(1, D, CV_64F);
	myMin = LARGEST_DOUBLE * Mat::ones(1, D, CV_64F);
	myMax = SMALLEST_DOUBLE * Mat::ones(1, D, CV_64F);
}

void Normalize::EstimateAdd(Mat &rowVec)
{
	myCount++;
	myMean = myMean + rowVec; 
	myMin = min(myMin, rowVec); 
	myMax = max(myMax, rowVec);
}

void Normalize::EstimateFinalize()
{
	myMean = myMean / myCount; 
	myDiv = myMax - myMin; 
}

void Normalize::Apply(Mat &rowvec)
{
	rowvec = rowvec - myMin;
	for (int i = 0; i < myDiv.cols; i++) {
		if (abs(myDiv.at<double>(i)) > FEATURE_TOLERANCE) {
			rowvec.at<double>(i) = rowvec.at<double>(i) / myDiv.at<double>(i); 
			rowvec.at<double>(i) = 2.0 * rowvec.at<double>(i) -1.0; 
		}
		else {
			rowvec.at<double>(i) = 0.0; 
		}
	}	
}

void Normalize::SaveModel(string fname)
{
	string sname = fname + ".normalize.yml";
	FileStorage fs(sname, FileStorage::WRITE);
	fs << "mean" << myMean; 
	fs << "var" << myVar; 
	fs << "min" << myMin;
	fs << "max" << myMax;
	fs << "div" << myDiv;
	fs.release();
}
	
void Normalize::LoadModel(string fname)
{
	myMean = Mat();
	myVar = Mat();
	myMin = Mat();
	myMax = Mat();

	string sname = fname + ".normalize.yml";
	FileStorage fs(sname, FileStorage::READ);
	fs["mean"] >> myMean; 
	fs["var"] >> myVar; 
	fs["min"] >> myMin;
	fs["max"] >> myMax;
	fs["div"] >> myDiv;
	fs.release();    
}

void Normalize::InitCheckRange(int D)
{	
	tempMin = LARGEST_DOUBLE * Mat::ones(1, D, CV_64F);
	tempMax = SMALLEST_DOUBLE * Mat::ones(1, D, CV_64F);
}

void Normalize::CheckRangeRow(cv::Mat &rowVec)
{
	for (int i = 0; i < tempMin.cols; i++) {
		if (rowVec.at<double>(i) < tempMin.at<double>(i)) tempMin.at<double>(i) = rowVec.at<double>(i);
		if (rowVec.at<double>(i) > tempMax.at<double>(i)) tempMax.at<double>(i) = rowVec.at<double>(i);
	}
}

void Normalize::FinalizeCheckRange()
{
	LOG_INFO(m_logger, "Feature Range");
	LOG_INFO(m_logger, "-------------");
	for (int i = 0; i < tempMin.cols; i++) {
		LOG_INFO(m_logger, "  " << tempMin.at<double>(i) << " -> " << tempMax.at<double>(i));
	}
	LOG_INFO(m_logger, "-------------");
}


Normalize::Normalize()
	: m_logger(LOG_GET_LOGGER("Normalize"))
{	
}

Normalize::~Normalize(){
}

