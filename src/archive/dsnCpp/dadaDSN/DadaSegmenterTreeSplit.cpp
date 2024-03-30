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

#include "MamaException.h"
#include "DadaSegmenterTreeSplit.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;


void DadaSegmenterTreeSplit::LabelParent(std::map<MamaVId, int> &myMap)
{
	MamaGraph &gc = *(m_graph.get());

	for (auto &it : myMap) {
		MamaVId nid = m_parentChildVertex[it.first];
		it.second = gc[nid].label;
	}
}


void DadaSegmenterTreeSplit::InitGraph(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	m_graph = make_shared<MamaGraph>();
	m_graph->clear(); 

	MamaGraph &gp = myParent->GetGraph(); 	
	MamaGraph &gc = *(m_graph.get());

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gp);

	// Add all vertices
	for (nit = nstart; nit != nend; nit++) {
		int labeli = gp[*nit].label;	
		MamaVId nid = add_vertex(gc);
		m_childParentVertex[nid] = *nit;
		m_parentChildVertex[*nit] = nid;		
	}


	
	// Add edges in sub graphs but not between
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gp);
	int ei = 0;
	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, gp);
		MamaVId id2 = target(*eit, gp);

		int label1 = gp[id1].label;
		int label2 = gp[id2].label;

		if (label1 == label2) {

		//if (gp[*eit].weight < 0.0) {

			if (!m_parentChildVertex.count(id1)) BOOST_THROW_EXCEPTION(Unexpected("no id1 in child 1"));
			if (!m_parentChildVertex.count(id2)) BOOST_THROW_EXCEPTION(Unexpected("no id2 in child 1"));

			MamaEId mid; bool newEdge;
			std::tie(mid, newEdge) = add_edge(m_parentChildVertex[id1], m_parentChildVertex[id2], gc);
			if (newEdge) {
				m_parentChildEdge[*eit] = mid;
				m_childParentEdge[mid] = *eit;
			}
			else {
				BOOST_THROW_EXCEPTION(Unexpected());
			}
		}
	}
	

	//LOG_INFO(m_logger, "S->Parent: " << num_vertices(gp) << " nodes with " << num_edges(gp) << " edges");
	//LOG_INFO(m_logger, "S->Child : " << num_vertices(gc) << " nodes with " << num_edges(gc) << " edges");	
	//LOG_INFO(m_logger, "S->Creating internal merge node"); 
	//m_mergeNode.InitGraph(myParent);
}

void DadaSegmenterTreeSplit::Finalize(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	MamaGraph myCopy;
	myCopy.clear(); 
	myCopy = *(m_graph.get());
	*(m_graph.get()) = myParent->GetGraph(); 
	MamaGraph &gc = *(m_graph.get());

	//LOG_INFO(m_logger, ">>>> On copy I have " << num_vertices(myCopy) << " nodes and " << num_edges(myCopy) << " edges"); 
	//LOG_INFO(m_logger, ">>>> New graph has " << num_vertices(gc) << " nodes and " << num_edges(gc) << " edges");
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(*(m_graph.get()));
	// copy label to new graph	
	for (nit = nstart; nit != nend; nit++) {
		if (!m_parentChildVertex.count(*nit)) BOOST_THROW_EXCEPTION(Unexpected("no parent child in split?"));
		MamaVId nid = m_parentChildVertex[*nit];
		gc[*nit].label = myCopy[nid].label;
		m_parentChildVertex[*nit] = *nit; 
	}
	m_fg = myParent->GetFG(); 
	m_gt = myParent->GetGT(); 
}

void DadaSegmenterTreeSplit::InitFeatures(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	LOG_TRACE_METHOD(m_logger, "InitFeatures");

	m_fg = make_shared<DadaFeatureGenerator>(m_param);	
	m_fg->SetBasins(myParent->GetFG()->GetBasins());
	m_fg->CalculateSplitFeatures(*(myParent->GetFG().get()), myParent->GetGraph(), *(m_graph.get()), m_parentChildVertex);
}

void DadaSegmenterTreeSplit::InitGT(std::shared_ptr<DadaSegmenterTreeNode> &myParent)
{
	LOG_TRACE_METHOD(m_logger, "InitGT");

	MamaGraph &gp = myParent->GetGraph();	
	MamaGraph &gc = *(m_graph.get());
	m_gt.Clear();

	// Need to get node negative 
	double extraPos = 0.0;
	double extraNeg = 0.0;
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(gc);
	for (nit = nstart; nit != nend; nit++) {
		m_gt.Labels()[*nit] = myParent->GetGT().Labels()[m_childParentVertex[*nit]];
		double tsum = 0.0;
		double tssum = 0.0;
		for (auto &it : m_gt.Labels()[*nit]) {
			tssum += (it.second * it.second - it.second) / 2.0;
			tsum += it.second;
		}
		double myCount = (tsum*tsum - tsum) / 2.0;

		extraPos += tssum;
		extraNeg += (myCount - tssum);
	}

	// need the merge graph for the split count	
	DadaWSGT tmpGT;
	tmpGT.Clear();
	// for each node in parent
	for (auto &nit : myParent->GetGT().Labels()) {
		// get the child node
		int labeli = gp[nit.first].label;
		MamaVId nid = (MamaVId) labeli; 
		// and pool
		for (auto &it : nit.second) {
			if (!tmpGT.Labels()[nid].count(it.first)) {
				tmpGT.Labels()[nid][it.first] = it.second;
			}
			else {
				tmpGT.Labels()[nid][it.first] += it.second;
			}
		}
	}

	// This is the positive error
	for (auto &nit1 : tmpGT.Labels()) {
		for (auto &nit2 : tmpGT.Labels()) {
			// each pair of clusters
			if (nit1.first < nit2.first) {
				// for each label
				double total1 = 0.0;
				double total2 = 0.0;
				double totalCommon = 0.0;

				for (auto &it : nit1.second) {
					// add the first 
					total1 += it.second;
					if (nit2.second.count(it.first)) {
						// add the second
						totalCommon += it.second * nit2.second[it.first];
					}
				}
				// get the left overs that don't match
				for (auto &it : nit2.second) {
					// add the first 
					total2 += it.second;
				}
				double myCount = (total1 * total2);

				m_gt.ExtraPos() += totalCommon;
				m_gt.ExtraNeg() += (myCount - totalCommon);
				//m_gt[cid1].ErrorPos() += 0.0;
				//m_gt[cid1].ErrorNeg() += m_gt[pid].ErrorNeg();

			}
		}
	}
	

	m_gt.ErrorPos() = m_gt.ExtraPos();
	m_gt.ErrorNeg() = extraNeg;
	m_gt.ExtraNeg() += extraNeg;
	m_gt.ExtraPos() += extraPos;

	m_gt.PosWeight() = myParent->GetGT().PosWeight();
	m_gt.NegWeight() = myParent->GetGT().NegWeight();

	//LOG_INFO(m_logger, "Parent EX P: " << myParent->GetGT().ExtraPos() << ", N: " << myParent->GetGT().ExtraNeg() << " ER P: " << myParent->GetGT().ErrorPos() << ", N: " << myParent->GetGT().ErrorNeg());
	//LOG_INFO(m_logger, "SChild EX P: " << m_gt.ExtraPos() << ", N: " << m_gt.ExtraNeg() << " ER P: " << m_gt.ErrorPos() << ", N: " << m_gt.ErrorNeg());	
}

void DadaSegmenterTreeSplit::Clear()
{
	DadaSegmenterTreeNode::Clear(); 
	m_childParentEdge.clear();
	m_parentChildEdge.clear();
	m_childParentVertex.clear();
	m_parentChildVertex.clear(); 
}

DadaSegmenterTreeSplit::DadaSegmenterTreeSplit(std::shared_ptr<DadaParam> &param)
	: DadaSegmenterTreeNode(param), m_mergeNode(param)
{	
}

DadaSegmenterTreeSplit::~DadaSegmenterTreeSplit()
{	
}

