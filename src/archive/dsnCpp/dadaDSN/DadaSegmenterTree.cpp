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

#include "DadaSegmenterTree.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaSegmenterTreeMerge.h"
#include "DadaSegmenterTreeSplit.h"
#include "DadaSegmenterTreeParam.h"
#include "DadaWS.h"
#include "Normalize.h"
#include "DadaWSUtil.h"


void DadaSegmenterTree::Apply(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg)
{
	LOG_TRACE_METHOD(m_logger, "Apply tree of depth " << m_treeNodes.size());
		

	m_treeNodes.clear();

	//LOG_INFO(m_logger, "Apply node 0");
	m_treeNodes[0] = make_shared<DadaSegmenterTreeMerge>(m_param);
	m_treeNodes[0]->Init(myGraph, fg); 
	DadaSegmenter::SetClassParam(m_treeParam->GetNode(0));
	DadaSegmenter::Apply(m_treeNodes[0]->GetGraphPtr(), m_treeNodes[0]->GetFG()); 

	if (m_numLabels < 2) {
		//LOG_INFO(m_logger, "=== Apply: Completely merged so exiting ===");
		return;
	}

	for (int nid = 1; nid < m_treeParam->GetNumNodes(); nid++) {
		//LOG_INFO(m_logger, "Apply node " << nid);
		if (m_treeParam->GetNodeType(nid) == 0) {
			m_treeNodes[nid] = make_shared<DadaSegmenterTreeMerge>(m_param);
		}
		else {
			m_treeNodes[nid] = make_shared<DadaSegmenterTreeSplit>(m_param);
		}
		m_treeNodes[nid]->Init(m_treeNodes[nid - 1]);

		DadaSegmenter::SetClassParam(m_treeParam->GetNode(nid));
		DadaSegmenter::Apply(m_treeNodes[nid]->GetGraphPtr(), m_treeNodes[nid]->GetFG()); 

		// used by split node
		m_treeNodes[nid]->Finalize(m_treeNodes[nid - 1]);

		if (m_numLabels < 2) {
			//LOG_INFO(m_logger, "=== Apply: Completely merged so exiting ===");
			break;
		}

	}	
}

void DadaSegmenterTree::Evaluate(std::shared_ptr<MamaGraph> &myGraph, DadaWSGT &groundTruth)
{
	m_treeNodes[0]->InitGT(groundTruth);
	for (int nid = 1; nid < m_treeParam->GetNumNodes(); nid++) {
		m_treeNodes[nid]->InitGT(m_treeNodes[nid-1]);
	}
	int lid = m_treeParam->GetNumNodes() - 1; 
	DadaSegmenter::Evaluate(m_treeNodes[lid]->GetGraphPtr(), m_treeNodes[lid]->GetGT()); 	
	//LOG_INFO(m_logger, "*** EVAL here is " << m_error.GetError());
}

void DadaSegmenterTree::Train(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg, DadaWSGT &groundTruth)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenterTree Train Tree");
	try {
		m_treeNodes.clear();

		LOG_INFO(m_logger, "=== Train Tree ===");

		m_treeNodes[0] = make_shared<DadaSegmenterTreeMerge>(m_param);
		m_treeNodes[0]->Init(myGraph, fg);		
		m_treeNodes[0]->InitGT(groundTruth);

		m_treeParam->AddRootNode(0, m_treeNodes[0]->GetFG()->D());

		DadaSegmenter::SetClassParam(m_treeParam->GetNode(0));
		//cout << m_treeParam->GetNodeType(0) << ": ";
		DadaSegmenter::Train(m_treeNodes[0]->GetGraphPtr(), m_treeNodes[0]->GetFG(), m_treeNodes[0]->GetGT());

		DadaError currentError = DadaSegmenter::GetError();

		DadaSegmenter::Apply(m_treeNodes[0]->GetGraphPtr(), m_treeNodes[0]->GetFG());

		if (m_numLabels < 2) {
			LOG_INFO(m_logger, "=== Train completely merged after first node");
			return; 
		}
		for (int i = 1; i < m_param->ensembleDepth; i++) {

			if (m_param->treeType == string("merge")) {
				AddNode(i - 1, i, "merge");
			}
			else if (m_param->treeType == string("split")) {
				AddNode(i - 1, i, "split");
			}
			else if (m_param->treeType == string("error")) {
				if (currentError.GetPosError() >= currentError.GetNegError()) {
					AddNode(i - 1, i, "merge");
				}
				else {
					AddNode(i - 1, i, "split");
				}
			}
			else if (m_param->treeType == string("alt")) {
				if (i % 2)		AddNode(i - 1, i, "split");
				else			AddNode(i - 1, i, "merge");
			}

			else BOOST_THROW_EXCEPTION(UnexpectedType("treeType needs to be merge, split or error"));

			//
			
			DadaSegmenter::SetClassParam(m_treeParam->GetNode(i));
			//cout << m_treeParam->GetNodeType(i) << ": ";

			DadaSegmenter::Train(m_treeNodes[i]->GetGraphPtr(), m_treeNodes[i]->GetFG(), m_treeNodes[i]->GetGT());
			DadaError nextError = DadaSegmenter::GetError();

			DadaSegmenter::Apply(m_treeNodes[i]->GetGraphPtr(), m_treeNodes[i]->GetFG()); 

			bool noImprove = (currentError.GetError() <= nextError.GetError()); 

			if (m_param->treeType == string("alt")) {
				if (i % 2) {
					noImprove = false; 
				}
			}
			if (noImprove || (i == (m_param->ensembleDepth-1))) {

				//LOG_INFO(m_logger, "No improvement at depth " << i);
				string temps = m_param->classifierLossType;
				m_param->classifierLossType = m_param->finalLossType;

				DadaSegmenter::Train(m_treeNodes[i]->GetGraphPtr(), m_treeNodes[i]->GetFG(), m_treeNodes[i]->GetGT());
				m_error = DadaSegmenter::GetError();
				DadaSegmenter::Apply(m_treeNodes[i]->GetGraphPtr(), m_treeNodes[i]->GetFG());
				m_param->classifierLossType = temps;
				//LOG_INFO(m_logger, "    Final error is " << m_error.GetError());
				// used by split node
				m_treeNodes[i]->Finalize(m_treeNodes[i - 1]);

				break;
			}

			// used by split node
			m_treeNodes[i]->Finalize(m_treeNodes[i - 1]);

			if (m_numLabels < 2) {
				LOG_INFO(m_logger, "=== Train completely merged, so exiting with depth " << i);
				break;
			}

			currentError = nextError;
		}
		//m_trainError = currentError;
	}
	catch (const std::exception& ex) {
		LOG_FATAL_EX(m_logger, "Tree train exception ", ex);
	}
}



void DadaSegmenterTree::InitLabelMap(std::shared_ptr<MamaGraph> &myGraph)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenterTree InitLabelMap with " << m_treeNodes.size());

	MamaGraph &gp = *(myGraph.get());
	m_labelMap.clear();
	m_vertexMap.clear();

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gp);

	for (nit = nstart; nit != nend; nit++) {
		MamaVId nidi = *nit;
		for (int i = 1; i < m_treeNodes.size(); i++) {
			nidi = m_treeNodes[i]->GetParentChildVertex()[nidi];
		}

		m_labelMap[*nit] = m_treeNodes[m_treeNodes.size() - 1]->GetGraph()[nidi].label;
		m_vertexMap[*nit] = nidi; 
	}

}


void DadaSegmenterTree::AddNode(int pid, int cid, string myType)
{
	LOG_TRACE_METHOD(m_logger, "AddNode " << myType);
	if (myType == string("merge")) {
		m_treeNodes[cid] = make_shared<DadaSegmenterTreeMerge>(m_param);
	}
	else {
		m_treeNodes[cid] = make_shared<DadaSegmenterTreeSplit>(m_param);
	}

	m_treeNodes[cid]->Init(m_treeNodes[pid]);	
	m_treeNodes[cid]->InitGT(m_treeNodes[pid]);
	int D = m_treeNodes[cid]->GetFG()->D();

	//cout << " D on add node is " << D << "\n";
	if (myType == string("merge")) {	
		m_treeParam->AddChild(pid, cid, 0, D);
	} else {		
		m_treeParam->AddChild(pid, cid, 1, D);
	}

}

void DadaSegmenterTree::RemoveNode(int pid, int cid)
{
	LOG_TRACE_METHOD(m_logger, "RemoveNode");

	m_treeParam->RemoveChild(pid, cid);
	m_treeNodes.erase(cid); 
}


void DadaSegmenterTree::Init(std::shared_ptr<DadaFeatureGenerator> &fg)
{
	LOG_TRACE_METHOD(m_logger, "Init");	
	m_treeParam = std::make_shared<DadaSegmenterTreeParam>(m_param);
	m_treeParam->Init();	
}

void DadaSegmenterTree::Save()
{
	m_treeParam->Save();
}

void DadaSegmenterTree::Load()
{
	m_treeParam->Load();
}

std::shared_ptr<DadaFeatureGenerator> DadaSegmenterTree::GetFeatureGenerator()
{
	if (m_treeNodes.size() > 0) {
		LOG_INFO(m_logger, "Returned feature generator for node " << m_treeNodes.size() - 1);
		return(m_treeNodes[m_treeNodes.size() - 1]->GetFG()); 
	}
	return(m_features);
}

DadaSegmenterTree::DadaSegmenterTree(std::shared_ptr<DadaParam> &param)
	: DadaSegmenter(param)	
{		
	m_logger.getInstance("DadaSegmenterTree");
}

DadaSegmenterTree::~DadaSegmenterTree()
{	
}

