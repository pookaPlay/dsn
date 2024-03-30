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

#include "DadaSegmenterForestParam.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaWSUtil.h"

void DadaSegmenterForestParam::Init()
{
	LOG_TRACE_METHOD(m_logger, "Init");
	
	m_treeParams.clear(); 
	
	for (int i = 0; i < m_param->ensembleSize; i++) {
		m_treeParams.push_back(std::make_shared<DadaSegmenterTreeParam>(m_param));
		m_treeParams[m_treeParams.size() - 1]->Init();
	}

	m_voteParam = std::make_shared<DadaClassifierParam>(m_param);
	m_voteParam->Init(1); 
}

void DadaSegmenterForestParam::CopyTo(std::shared_ptr<DadaSegmenterForestParam> &dest)
{
	BOOST_THROW_EXCEPTION(NotImplemented()); 
	//dest->W() = m_weights; 
	//dest->T() = m_threshold; 
}

void DadaSegmenterForestParam::Print()
{
	//LOG_INFO(m_logger, "W: " << m_weights);
	//LOG_INFO(m_logger, "T: " << m_threshold); 	
	BOOST_THROW_EXCEPTION(NotImplemented());
}


void DadaSegmenterForestParam::Save()
{
	string sname = m_param->modelName + string(".param.yml");
	Save(sname);
}

void DadaSegmenterForestParam::Save(string fname)
{
	FileStorage fs(fname, FileStorage::WRITE);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem());
	Save(fs);
	fs.release();
}

void DadaSegmenterForestParam::Save(FileStorage &fs) const
{
	m_voteParam->Save(fs); 

	fs << "Trees" << "[";
	for (int i = 0; i < m_treeParams.size(); i++) {
		fs << "{";
		fs << "index" << i;
		m_treeParams[i]->Save(fs);
		fs << "}";
	}	
	fs << "]";
}


void DadaSegmenterForestParam::Load()
{
	string sname = m_param->modelName + string(".param.yml");
	Load(sname);
}

void DadaSegmenterForestParam::Load(string fname)
{
	FileStorage fs(fname, FileStorage::READ);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem());
	Load(fs);
	fs.release();
}

void DadaSegmenterForestParam::Load(FileStorage &fs)
{
	m_voteParam->Load(fs); 
	
	m_treeParams.clear();
	FileNode trees = fs["Trees"];
	for (FileNodeIterator it = trees.begin(); it != trees.end(); ++it) {
		m_treeParams.push_back(make_shared<DadaSegmenterTreeParam>(m_param)); 
		int id;
		(*it)["id"] >> id;
		m_treeParams[m_treeParams.size() - 1]->Load(it);
	}
}

DadaSegmenterForestParam::DadaSegmenterForestParam(std::shared_ptr<DadaParam> &param)	
	: m_logger(LOG_GET_LOGGER("DadaSegmenterForestParam")),
	  m_param(param)
{			
	m_voteParam = nullptr; 
	m_treeParams.clear();
}

DadaSegmenterForestParam::~DadaSegmenterForestParam()
{	
}

