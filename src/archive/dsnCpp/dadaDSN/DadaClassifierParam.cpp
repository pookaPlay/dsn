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

#include "DadaClassifierParam.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"

void DadaClassifierParam::Init(int D)
{
	LOG_TRACE_METHOD(m_logger, "Init");
	m_D = D; 
	if (m_param->classifierInitType == string("uniform")) {		// uniform
		m_weights = Mat::zeros(D, 1, CV_64F);
		for (int i = 0; i < m_weights.rows; i++) {
			m_weights.at<double>(i) = (1.0 / boost::numeric_cast<double>(m_weights.rows));
		}
		m_threshold = 0.0;
	}
	else if (m_param->classifierInitType == string("random")) {	// random
		m_weights = Mat::zeros(D, 1, CV_64F);
		for (int i = 0; i < m_weights.rows; i++) {
			double rf = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			double weightRange = 0.1;
			m_weights.at<double>(i) = ((weightRange * rf) - weightRange / 2.0);
		}
		m_threshold = 0.0;
	}
	else if (m_param->classifierInitType == string("zero")) {	// random
		m_weights = Mat::zeros(D, 1, CV_64F);
		m_threshold = 0.0;
	}
	else {
		m_weights = Mat(); 
		m_threshold = 0.0;
		m_index = -1; 		
	}

	m_segType = m_param->segmentationType; 

	m_validFeatures.clear();
	for (int i = 0; i < D; i++) m_validFeatures.push_back(i); 
	
}


void DadaClassifierParam::CopyTo(std::shared_ptr<DadaClassifierParam> &dest)
{
	dest->W() = m_weights; 
	dest->T() = m_threshold; 
}

void DadaClassifierParam::Print()
{
	LOG_INFO(m_logger, "I: " << m_index << " and T: " << m_threshold); 	
}

void DadaClassifierParam::Save()
{
	string sname = m_param->modelName + string(".param.yml");
	Save(sname); 
}

void DadaClassifierParam::Save(string fname)
{
	FileStorage fs(fname, FileStorage::WRITE);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem());
	Save(fs); 
	fs.release();
}

void DadaClassifierParam::Save(FileStorage &fs) const
{
	fs << "thresh" << m_threshold;
	fs << "index" << m_index;
	fs << "segType" << m_segType;
}


void DadaClassifierParam::Load()
{
	string sname = m_param->modelName + string(".param.yml");
	Load(sname);
}

void DadaClassifierParam::Load(string fname)
{
	FileStorage fs(fname, FileStorage::READ);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem());
	Load(fs);
	fs.release();
}

void DadaClassifierParam::Load(FileStorage &fs)
{
	LOG_TRACE_METHOD(m_logger, "Load FileStorage"); 
	fs["thresh"] >> m_threshold;
	fs["index"] >> m_index;
	fs["segType"] >> m_segType;
}

void DadaClassifierParam::Load(FileNodeIterator &fs)
{
	LOG_TRACE_METHOD(m_logger, "Load FileNode");
	(*fs)["thresh"] >> m_threshold;
	(*fs)["index"] >> m_index;
	(*fs)["segType"] >> m_segType;
}


DadaClassifierParam::DadaClassifierParam(std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaClassifierParam")),
	m_param(param)
{
	m_probType = 0;
	m_validFeatures.clear();
}

DadaClassifierParam::~DadaClassifierParam()
{	
}

