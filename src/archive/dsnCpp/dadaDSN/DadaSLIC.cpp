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

#include "DadaSLIC.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "SLIC.h"

Logger DadaSLIC::m_logger(LOG_GET_LOGGER("DadaSLIC"));

void DadaSLIC::Run(cv::Mat &img, cv::Mat &bseg, int desiredK, double compactness)
{
	LOG_TRACE_METHOD(m_logger, "Run");

	if (img.type() != CV_8UC3) {
		LOG_INFO(m_logger, "Image type is " << img.type());
		BOOST_THROW_EXCEPTION(UnexpectedType()); 
	}
	
	// unsigned int (32 bits) to hold a pixel in ARGB format as follows:
	// from left to right,
	// the first 8 bits are for the alpha channel (and are ignored)
	// the next 8 bits are for the red channel
	// the next 8 bits are for the green channel
	// the last 8 bits are for the blue channel
	//unsigned int* pbuff = new UINT[sz];
	int width = img.cols;
	int height = img.rows;
	int sz = width*height; 
	
	unsigned int *pbuff = new unsigned int [sz]; 
	int *klabels = new int[sz]; 

	unsigned char *cbuff = (unsigned char *) pbuff;
	unsigned char *mbuf = img.ptr();

	int srci = 0; 
	int dsti = 0;
	for (int i = 0; i < sz; i++) {
		*(cbuff + dsti + 0) = 0; 
		*(cbuff + dsti + 1) = *(mbuf + srci + 2);
		*(cbuff + dsti + 2) = *(mbuf + srci + 1);
		*(cbuff + dsti + 3) = *(mbuf + srci + 0);
		dsti += 4; 
		srci += 3; 
	}
	

	//----------------------------------
	// Initialize parameters
	//----------------------------------
	//int k = 500;//Desired number of superpixels.
	//double m = 20;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10	
	int numlabels(0);

	//----------------------------------
	// Perform SLIC on the image buffer
	//----------------------------------
	//LOG_INFO(m_logger, "Into SLIC!");
	SLIC segment;
	segment.PerformSLICO_ForGivenK(pbuff, width, height, klabels, numlabels, desiredK, compactness);
	// Alternately one can also use the function PerformSLICO_ForGivenStepSize() for a desired superpixel size

	srci = 0; 
	bseg = Mat::zeros(height, width, CV_32F); 
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			bseg.at<float>(j, i) = static_cast<float>(*(klabels + srci)); 
			srci++;
		}
	}
	//LOG_INFO(m_logger, "Success!");

	if (pbuff) delete[] pbuff;
	if (klabels) delete[] klabels;
}

DadaSLIC::DadaSLIC()	
{	
}

DadaSLIC::~DadaSLIC()
{	
}

