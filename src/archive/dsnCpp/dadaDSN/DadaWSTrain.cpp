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

#include "DadaWSTrain.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"

#include <fstream>
#include <boost/graph/graphml.hpp>


/** 
*	This modifies editSeg
**/
void DadaWSTrain::Init(std::vector< cv::Mat > &origImg,  cv::Mat &origSeg, cv::Mat &editSeg, cv::Mat &origBasins)	
{
	if (origSeg.type() != CV_32F) BOOST_THROW_EXCEPTION(UnexpectedType());
	if (editSeg.type() != CV_32F) BOOST_THROW_EXCEPTION(UnexpectedType());
	if (origImg.size() < 1) BOOST_THROW_EXCEPTION(UnexpectedSize("where are the images"));
	if (origSeg.rows != origBasins.rows) BOOST_THROW_EXCEPTION(UnexpectedSize("base seg and basins awol"));
	if (editSeg.rows != origBasins.rows) BOOST_THROW_EXCEPTION(UnexpectedSize("edit seg and basins awol"));
	// get the difference mask
	cv::Mat diffImg;
	absdiff(origSeg, editSeg, diffImg); 
	//VizMat::DisplayFloat(diffImg, "test", 0, 1.0); 
	// Get the basin labels that have some change
	map<int, int> changed;
	changed.clear();

	for (int j = 0; j < diffImg.rows; j++) {
		for (int i = 0; i < diffImg.cols; i++) {
			if (diffImg.at<float>(j, i) > 0.5f) {			
				int myl = static_cast<int>(origBasins.at<float>(j, i));
				changed[myl] = 1; 				
			}
			else {
				editSeg.at<float>(j, i) = -1.0f; 
			}
		}
	}
	//VizMat::DisplayFloat(editSeg, "test", 0, 1.0); 

	// Add neighbors 
	map<int, int> touched;
	touched.clear();
	touched = changed; 

	for (int j = 0; j < (origBasins.rows - 1); j++) {
		for (int i = 0; i < (origBasins.cols - 1); i++) {
			// right 
			int myr = static_cast<int>(origBasins.at<float>(j, i+1));
			if (changed.count(myr)) {
				int myl = static_cast<int>(origBasins.at<float>(j, i));
				touched[myl] = 1; 
			}
			// down 
			int myb = static_cast<int>(origBasins.at<float>(j+1, i));
			if (changed.count(myb)) {
				int myl = static_cast<int>(origBasins.at<float>(j, i));
				touched[myl] = 1;
			}
		}
	}

	// Final pass to get image range
	m_colRange.start = LARGEST_INT; 
	m_colRange.end = SMALLEST_INT;
	m_rowRange.start = LARGEST_INT;
	m_rowRange.end = SMALLEST_INT;

	for (int j = 0; j < origBasins.rows; j++) {
		for (int i = 0; i < origBasins.cols; i++) {
			int myr = static_cast<int>(origBasins.at<float>(j, i));
			//if (touched.count(myr)) {
				m_colRange.start = min(m_colRange.start, i);
				m_colRange.end = max(m_colRange.end, i);
				m_rowRange.start = min(m_rowRange.start, j);
				m_rowRange.end = max(m_rowRange.end, j);
			//}
		}
	}
	m_colRange.end += 1; 
	m_rowRange.end += 1;
	LOG_INFO(m_logger, changed.size() << " diff, " << touched.size() << " total, [" << m_colRange.start << "," << m_rowRange.start << "] -> [" << m_colRange.end << "," << m_rowRange.end << "]");
	m_input.clear();
	m_input.resize(origImg.size());
	for (int i = 0; i < origImg.size(); i++) {
		m_input[i] = origImg[i](m_rowRange, m_colRange);
	}
	m_basins = origBasins(m_rowRange, m_colRange);
	m_groundTruth = editSeg(m_rowRange, m_colRange);
}

DadaWSTrain::DadaWSTrain(std::vector< cv::Mat > &origImg, cv::Mat &origSeg, cv::Mat &editSeg, cv::Mat &origBasins)
	: DadaWSTrain()
{
	Init(origImg, origSeg, editSeg, origBasins);
}

DadaWSTrain::DadaWSTrain()
	: m_logger(LOG_GET_LOGGER("DadaWSTrain"))
{

}

DadaWSTrain::~DadaWSTrain()
{	
}

/*
std::vector<int> dadaColMap = boost::assign::list_of(41)(9)(12)(29)(33)(35)(58)(10)(34)(53)(45)(13)(49)(61)(25)(16)(5)(21)(11)(55)(27)(48)(60)(1)(36)(44)(43)(6)(54)(14)(37)(26)(51)(39)(31)(23)(19)(38)(8)(20)(63)(47)(4)(52)(57)(28)(2)(3)(18)(62)(17)(0)(46)(40)(50)(30)(59)(7)(56)(24)(22)(15)(42)(32);


void DadaWSTrain::VizSubGraphs(string id, int waitVal, double mag)
{
	// First compress
	Mat picy = Mat::zeros(m_h, m_w, CV_8UC3);

	double ilum = 255.0;
	for (int j = 0; j < m_h; j++) {
		for (int i = 0; i < m_w; i++) {
						
			int vali = static_cast<int>(m_basins.at<float>(j, i));
			if (m_trainBasins.count(m_basinVMap->operator[](vali))) {
				// in the training set
				ilum = 255.0;
			}
			else {
				// not in training 
				ilum = 64.0; 
			}

			int cii = dadaColMap[vali % 64];
			int col1 = (int)NEAREST_INT(jetcolormap[cii * 3 + 0] * ilum);
			int col2 = (int)NEAREST_INT(jetcolormap[cii * 3 + 1] * ilum);
			int col3 = (int)NEAREST_INT(jetcolormap[cii * 3 + 2] * ilum);
			picy.at<Vec3b>(j, i)[0] = col1;
			picy.at<Vec3b>(j, i)[1] = col2;
			picy.at<Vec3b>(j, i)[2] = col3;
		}
	}


	Size sz = Size(static_cast<int>(m_w*mag), static_cast<int>(m_h*mag));
	Mat out = Mat::zeros(sz, CV_8UC3);
	resize(picy, out, sz);

	namedWindow(id);
	imshow(id, out);
	if (waitVal >= 0) waitKey(waitVal);	
	
}
*/