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

#include "DadaSegmenterTreeNode.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"

void DadaSegmenterTreeNode::Init(std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr<DadaFeatureGenerator> &fg)
{
	LOG_TRACE_METHOD(m_logger, "Init");
	this->Clear();
	m_fg = fg; 
	m_graph = myGraph;

}

void DadaSegmenterTreeNode::InitGT(DadaWSGT &groundTruth)
{
	LOG_TRACE_METHOD(m_logger, "Init GT");
	m_gt = groundTruth;
}

void DadaSegmenterTreeNode::Init()
{
	LOG_TRACE_METHOD(m_logger, "Init");
	if (!m_parent)  BOOST_THROW_EXCEPTION(Unexpected());
	this->Clear();
	this->InitGraph(m_parent);
	this->InitFeatures(m_parent);	
}

void DadaSegmenterTreeNode::Init(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	LOG_TRACE_METHOD(m_logger, "Init");
	this->Clear();
	this->InitGraph(myParent); 
	this->InitFeatures(myParent);
	m_parent = myParent; 
}

void DadaSegmenterTreeNode::InitGT(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	LOG_TRACE_METHOD(m_logger, "InitGT");
}

void DadaSegmenterTreeNode::Finalize(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{

}

void DadaSegmenterTreeNode::LabelGraph(std::map<MamaVId, int> &myMap)
{
	MamaGraph &gc = *(m_graph.get());

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gc);
	
	for (nit = nstart; nit != nend; nit++) {
		if (!myMap.count(*nit)) BOOST_THROW_EXCEPTION(Unexpected()); 
		gc[*nit].label = myMap[*nit];
	}

}

void DadaSegmenterTreeNode::Clear()
{
	LOG_TRACE_METHOD(m_logger, "Clear");	
	m_gt.Clear(); 
	m_graph = nullptr;
	m_fg = nullptr; 
}


DadaSegmenterTreeNode::DadaSegmenterTreeNode(std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaSegmenterTreeNode")), m_param(param)
{	
}

DadaSegmenterTreeNode::~DadaSegmenterTreeNode()
{	
}

