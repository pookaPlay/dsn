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

#include "DadaSegmenterTreeMerge.h"
#include "MamaException.h"
#include "MamaDef.h"
#include "DadaFeatures.h"
using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

void DadaSegmenterTreeMerge::LabelParent(std::map<MamaVId, int> &myMap)
{
	MamaGraph &gc = *(m_graph.get());
	MamaGraph &gp = m_parent->GetGraph();
	for (auto &it : myMap) {
		MamaVId nid = m_parentChildVertex[it.first];
		it.second = gc[nid].label;
	}
}


void DadaSegmenterTreeMerge::InitGraph(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenterTreeMerge::InitGraph");

	m_graph = make_shared<MamaGraph>();
	m_graph->clear();

	MamaGraph &gp = myParent->GetGraph();
	MamaGraph &gc = *(m_graph.get());
	
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gp);

	
	for (nit = nstart; nit != nend; nit++) {
		MamaVId nid;
		if (!m_labelChild.count(gp[*nit].label)) {			
			nid = add_vertex(gc);
			m_labelChild[gp[*nit].label] = nid;			
		}
		else {
			nid = m_labelChild[gp[*nit].label];
		}
		m_parentChildVertex[*nit] = nid; 
	}

	// Now go through edges
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gp);
	int ei = 0;
	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, gp);
		MamaVId id2 = target(*eit, gp);
		int label1 = gp[id1].label;
		int label2 = gp[id2].label;

		if (label1 != label2) {
			// edge belongs to child 2
			MamaEId mid; bool newEdge;
			std::tie(mid, newEdge) = add_edge(m_labelChild[label1], m_labelChild[label2], gc);
			if (newEdge) {
				m_childParentEdge[mid].clear();
			}
			m_childParentEdge[mid].push_back(*eit);
			m_parentChildEdge[*eit] = mid;
		}
	}

	//LOG_INFO(m_logger, "M->Parent: " << num_vertices(gp) << " nodes with " << num_edges(gp) << " edges");
	//LOG_INFO(m_logger, "M->Child : " << num_vertices(gc) << " nodes with " << num_edges(gc) << " edges");	
}


void DadaSegmenterTreeMerge::InitFeatures(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenterTreeMerge::InitFeatures");

	m_fg = make_shared<DadaFeatureGenerator>(m_param);
	m_fg->SetBasins(myParent->GetFG()->GetBasins()); 
	m_fg->CalculateMergeFeatures(*(myParent->GetFG().get()), myParent->GetGraph(), *(m_graph.get()), m_labelChild, m_childParentEdge);

}

void DadaSegmenterTreeMerge::InitGT(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	LOG_TRACE_METHOD(m_logger, "DadaSegmenterTreeMerge::InitGT");

	MamaGraph &gp = myParent->GetGraph(); 
	m_gt.Clear(); 

	// for each node in parent
	for (auto &nit : myParent->GetGT().Labels()) {
		// get the child node
		int labeli = gp[nit.first].label;
		MamaVId nid = m_labelChild[labeli];
		// and pool
		for (auto &it : nit.second) {
			if (!m_gt.Labels()[nid].count(it.first)) {
				m_gt.Labels()[nid][it.first] = it.second;
			}
			else {
				m_gt.Labels()[nid][it.first] += it.second;
			}
		}
	}


	for (auto &nit : m_gt.Labels()) {

		double tsum = 0.0;
		double tssum = 0.0;

		for (auto &it : nit.second) {
			tssum += (it.second * it.second - it.second) / 2.0;
			tsum += it.second;
		}
		double myCount = (tsum*tsum - tsum) / 2.0;

		m_gt.ExtraPos() += tssum;
		m_gt.ExtraNeg() += (myCount - tssum);
		//m_gt[cid2].ErrorPos() += 0.0;
		//m_gt[cid2].ErrorNeg() += (myCount - tssum);
	}
	m_gt.ErrorPos() = 0.0;
	m_gt.ErrorNeg() = m_gt.ExtraNeg();
	//m_gt[cid2].ErrorNeg() += m_gt[pid].ErrorNeg();
	//m_gt[cid2].ErrorPos() += m_gt[pid].ErrorPos();
	//m_gt[cid2].ExtraPos() += m_gt[pid].ExtraPos();
	//m_gt[cid2].ExtraNeg() += m_gt[pid].ExtraNeg();

	m_gt.PosWeight() = myParent->GetGT().PosWeight(); 
	m_gt.NegWeight() = myParent->GetGT().NegWeight();
	//LOG_INFO(m_logger, "Parent EX P: " << myParent->GetGT().ExtraPos() << ", N: " << myParent->GetGT().ExtraNeg() << " ER P: " << myParent->GetGT().ErrorPos() << ", N: " << myParent->GetGT().ErrorNeg());
	//LOG_INFO(m_logger, "MChild EX P: " << m_gt.ExtraPos() << ", N: " << m_gt.ExtraNeg() << " ER P: " << m_gt.ErrorPos() << ", N: " << m_gt.ErrorNeg());
}


void DadaSegmenterTreeMerge::Clear()
{
	DadaSegmenterTreeNode::Clear();	
	m_labelChild.clear();
	m_parentChildVertex.clear();
}

DadaSegmenterTreeMerge::DadaSegmenterTreeMerge(std::shared_ptr<DadaParam> &param)
	: DadaSegmenterTreeNode(param)
{	
}

DadaSegmenterTreeMerge::~DadaSegmenterTreeMerge()
{	
}

