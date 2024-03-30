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

#include "DadaSegmenterTreeParam.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaWSUtil.h"

void DadaSegmenterTreeParam::Init()
{
	LOG_TRACE_METHOD(m_logger, "Init");
	m_nodeType.clear();
	m_nodeParams.clear();
	m_rootNode = -1;
	m_children.clear();
	m_parent.clear(); 	
}

void DadaSegmenterTreeParam::AddRootNode(int nid, int D)
{
	AddChild(-1, nid, 0, D);	
}

void DadaSegmenterTreeParam::AddChild(int pid, int nid, int typei, int D)
{
	if (m_nodeParams.count(nid)) {
		LOG_INFO(m_logger, "Node " << nid << " already exists?");
		BOOST_THROW_EXCEPTION(Unexpected());
	}
	if (pid >= 0) {
		if (!m_nodeParams.count(pid)) {
			LOG_INFO(m_logger, "Where's parent?");
			BOOST_THROW_EXCEPTION(Unexpected());
		}
	}

	m_nodeType[nid] = typei;
	m_nodeParams[nid] = make_shared<DadaClassifierParam>(m_param);
	m_nodeParams[nid]->Init(D);
	
	vector<int> validFeatures; 
	DadaWSUtil::ChooseRandomFeatures(validFeatures, D, m_param->featureSubsetSize);
	//string temps;
	//VectorToString(validFeatures, temps);
	//LOG_INFO(m_logger, "From " << D << " features: " << temps);

	if (validFeatures.size() > 0) m_nodeParams[nid]->SetValidFeatures(validFeatures); 		

	m_parent[nid] = pid;	
	if (!m_children.count(pid)) {
		m_children[pid].clear();
	}
	m_children[pid].push_back(nid);	
}


void DadaSegmenterTreeParam::RemoveChild(int pid, int nid)
{
	if (!m_nodeParams.count(nid)) {
		BOOST_THROW_EXCEPTION(Unexpected());
	}
	int before = m_nodeParams.size();
	m_nodeParams.erase(nid); 
	m_nodeType.erase(nid); 

	int after = m_nodeParams.size();
	//cout << before << " and after " << after << "\n";

	m_parent.erase(nid); 
	m_children.erase(pid); 
}
std::shared_ptr<DadaClassifierParam> DadaSegmenterTreeParam::GetNode(int nid)
{
	if (!m_nodeParams.count(nid)) {
		BOOST_THROW_EXCEPTION(Unexpected());
		//BOOSTYreturn(nullptr);
	}
	return(m_nodeParams[nid]);
}

void DadaSegmenterTreeParam::CopyTo(std::shared_ptr<DadaSegmenterTreeParam> &dest)
{
	//dest->W() = m_weights; 
	//dest->T() = m_threshold; 
}

void DadaSegmenterTreeParam::Print()
{
	//LOG_INFO(m_logger, "W: " << m_weights);
	//LOG_INFO(m_logger, "T: " << m_threshold); 	
}

int DadaSegmenterTreeParam::GetNodeType(int nid) 
{
	if (m_nodeType.count(nid)) return(m_nodeType[nid]);
	BOOST_THROW_EXCEPTION(Unexpected());
}

void DadaSegmenterTreeParam::Save()
{
	string sname = m_param->modelName + string(".param.yml");
	Save(sname);
}

void DadaSegmenterTreeParam::Save(string fname)
{
	FileStorage fs(fname, FileStorage::WRITE);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem());
	Save(fs);
	fs.release();
}

void DadaSegmenterTreeParam::Save(FileStorage &fs) const
{	
	fs << "Nodes" << "[";
	for (auto it : m_nodeParams) {
		fs << "{";				
		fs << "id" << it.first;				
		fs << "type" << m_nodeType.at(it.first);
		it.second->Save(fs); 
		fs << "}";
	}
	fs << "]";
}


void DadaSegmenterTreeParam::Load()
{
	string sname = m_param->modelName + string(".param.yml");
	Load(sname);
}

void DadaSegmenterTreeParam::Load(string fname)
{
	FileStorage fs(fname, FileStorage::READ);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem());
	Load(fs);
	fs.release();
}

void DadaSegmenterTreeParam::Load(FileStorage &fs)
{
	m_nodeParams.clear(); 	
	m_nodeType.clear(); 
	// iterate through a sequence using FileNodeIterator
	FileNode features = fs["Nodes"];
	for (FileNodeIterator it = features.begin(); it != features.end(); ++it) {
		int id, typei;
		(*it)["id"] >> id;
		(*it)["type"] >> typei;
		m_nodeType[id] = typei;
		m_nodeParams[id] = make_shared<DadaClassifierParam>(m_param);
		m_nodeParams[id]->Load(it);
	}
}

void DadaSegmenterTreeParam::Load(FileNodeIterator &fs)
{
	m_nodeParams.clear();
	m_nodeType.clear();
	// iterate through a sequence using FileNodeIterator
	FileNode features = (*fs)["Nodes"];
	for (FileNodeIterator it = features.begin(); it != features.end(); ++it) {
		int id, typei;
		(*it)["id"] >> id;
		(*it)["type"] >> typei;
		m_nodeType[id] = typei;
		m_nodeParams[id] = make_shared<DadaClassifierParam>(m_param);
		m_nodeParams[id]->Load(it);
	}
}



DadaSegmenterTreeParam::DadaSegmenterTreeParam(std::shared_ptr<DadaParam> &param)	
	: m_logger(LOG_GET_LOGGER("DadaSegmenterTreeParam")),
	  m_param(param)
{			
}

DadaSegmenterTreeParam::~DadaSegmenterTreeParam()
{	
}

