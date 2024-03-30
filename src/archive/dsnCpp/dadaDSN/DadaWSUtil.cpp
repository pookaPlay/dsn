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

#include "DadaWSUtil.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"

#include <fstream>
#include <boost/graph/graphml.hpp>

Logger DadaWSUtil::m_logger(LOG_GET_LOGGER("DadaWSUtil"));

std::mt19937 DadaWSUtil::m_rgen(m_rd());

void DadaWSUtil::SetRandomSeed(int seed)
{
	m_rgen.seed(seed); 
	srand(seed); 
}

int DadaWSUtil::ThresholdLabelEdgeWeight(MamaGraph &myGraph, double threshold)
{
	LOG_TRACE_METHOD(m_logger, "ThresholdLabel");

	int segCount = 0;
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(myGraph);

	for (nit = nstart; nit != nend; nit++) {
		myGraph[*nit].label = -1;
	}

	for (nit = nstart; nit != nend; nit++) {
		if (myGraph[*nit].label < 0) {
			int compSize = DadaWSUtil::LabelVertices(*nit, segCount, threshold, myGraph);
			segCount++;
		}
	}	
	return(segCount); 
}

int DadaWSUtil::LabelVertices(MamaVId id, int label, double threshold, MamaGraph &myGraph)
{
	vector<MamaVId> newguy;
	newguy.clear();
	newguy.push_back(id);

	int ci = 0;

	while (newguy.size() > 0) {
		MamaVId nextId = newguy[newguy.size() - 1];
		newguy.pop_back();
		myGraph[nextId].label = label;
		ci++;
		// get neighbors
		MamaNeighborIt nit, nstart, nend;
		std::tie(nstart, nend) = adjacent_vertices(nextId, myGraph);
		for (nit = nstart; nit != nend; nit++) {
			if (myGraph[*nit].label < 0) {
				std::pair < MamaEId, bool > peid = edge(nextId, *nit, myGraph);
				if (!(peid.second)) BOOST_THROW_EXCEPTION(Unexpected());
				if (myGraph[peid.first].weight < threshold) {
					newguy.push_back(*nit);
				}
			}
		}
	}

	return(ci);
}


void DadaWSUtil::ApplyWatershed(MamaGraph &myGraph, map<MamaEId, MamaEId> &watershedNeighbors)
{
	LOG_TRACE_METHOD(m_logger, "ApplyWatershed");

	watershedNeighbors.clear();

	MamaEdgeIt eit, estart, eend;
	MamaNeighborEdgeIt neit, nestart, neend;
	int ii = 0;
	std::tie(estart, eend) = edges(myGraph);
	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, myGraph);
		MamaVId id2 = target(*eit, myGraph);

		double allReal1 = LARGEST_DOUBLE;
		double allReal2 = LARGEST_DOUBLE;
		MamaEId wIndex1, wIndex2;

		if (out_degree(id1, myGraph) > 1) {
			std::tie(nestart, neend) = out_edges(id1, myGraph);
			for (neit = nestart; neit != neend; neit++) {
				if (*neit != *eit) {
					if (myGraph[*neit].weight < allReal1) {
						allReal1 = myGraph[*neit].weight;
						wIndex1 = (MamaEId)(*neit);
					}
				}
			}
		}
		else allReal1 = SMALLEST_DOUBLE;

		if (out_degree(id2, myGraph) > 1) {
			std::tie(nestart, neend) = out_edges(id2, myGraph);
			for (neit = nestart; neit != neend; neit++) {
				if (*neit != *eit) {
					if (myGraph[*neit].weight < allReal2) {
						allReal2 = myGraph[*neit].weight;
						wIndex2 = (MamaEId)(*neit);
					}
				}
			}
		}
		else allReal2 = SMALLEST_DOUBLE;

		double allReal;
		MamaEId wIndex;
		if (allReal1 > allReal2) {
			allReal = allReal1;
			wIndex = wIndex1;
		}
		else {
			allReal = allReal2;
			wIndex = wIndex2;
		}
		watershedNeighbors[*eit] = wIndex;

		double threshVal = myGraph[*eit].weight - allReal;

		myGraph[*eit].wasWeight = myGraph[*eit].weight;
		myGraph[*eit].weight = threshVal;

	}
}


void DadaWSUtil::PrintLabelCount(std::map<MamaVId, int> &lmap)
{
	map<int, int> labelCount, frontc, backc;
	labelCount.clear();
	frontc.clear(); 
	backc.clear(); 

	int orig = lmap.size() / 2; 

	for (auto &it : lmap) {
		if (!labelCount.count(it.second)) {
			labelCount[it.second] = 0; 
		}
		labelCount[it.second] += 1;
		int index = static_cast<int>(it.first); 
		if (index < orig) {
			if (!frontc.count(it.second)) {
				frontc[it.second] = 0;
			}
			frontc[it.second] += 1;
		}
		else {
			if (!backc.count(it.second)) {
				backc[it.second] = 0;
			}
			backc[it.second] += 1;

		}

	}
	LOG_INFO(m_logger, "Base " << lmap.size() << ", " << labelCount.size() << " labels, " << frontc.size() << " front, " << backc.size() << "back"); 
	//for (auto &it : labelCount) {
	//	LOG_INFO(m_logger, "Label " << labelCount.size() << " has " << labelCount.size() << " labels ");
	//}
}



void DadaWSUtil::ExportSegmentWS(SegmentWS &ws, string fname, int saveBase)
{
	string sname = fname + ".ws.yml";
	FileStorage fs(sname, FileStorage::WRITE);
	
	fs << "w" << ws.GetW();
	fs << "h" << ws.GetH(); 

	if (saveBase) {
		sname = fname + ".base.xml";
		DadaWSUtil::ExportMamaGraph(ws.GetBasinGraph(), sname);
	}
	sname = fname + ".xml";
	DadaWSUtil::ExportMamaGraph(ws.GetGraph(), sname);
}

void DadaWSUtil::ExportSegmentWS(SegmentWS &ws, std::ofstream &fout)
{
	BOOST_THROW_EXCEPTION(NotImplemented());
	fout << ws.GetW() << "\n";
	fout << ws.GetH() << "\n";
	DadaWSUtil::ExportMamaGraph(ws.GetBasinGraph(), fout); 
	DadaWSUtil::ExportMamaGraph(ws.GetGraph(), fout);
}

void DadaWSUtil::ImportSegmentWS(SegmentWS &ws, string fname, int saveBase)
{
	string sname = fname + ".ws.yml";
	FileStorage fs(sname, FileStorage::READ);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem());
	int tempi;
	fs["w"] >> tempi;
	ws.SetW(tempi); 
	fs["h"] >> tempi;
	ws.SetH(tempi);

	if (saveBase) {
		sname = fname + ".base.xml";
		DadaWSUtil::ImportMamaGraph(ws.GetBasinGraph(), sname);
	}
	sname = fname + ".xml";
	DadaWSUtil::ImportMamaGraph(ws.GetGraph(), sname);
}

void DadaWSUtil::ImportSegmentWS(SegmentWS &ws, std::ifstream &fin)
{
	BOOST_THROW_EXCEPTION(NotImplemented()); 
	int tempi;
	fin >> tempi; 
	ws.SetW(tempi);
	fin >> tempi;
	ws.SetH(tempi); 
	LOG_INFO(m_logger, "Got " << ws.GetW() << " by " << ws.GetH());

	DadaWSUtil::ImportMamaGraph(ws.GetBasinGraph(), fin);
	DadaWSUtil::ImportMamaGraph(ws.GetGraph(), fin);
}


void DadaWSUtil::ExportMamaGraph(MamaGraph &m, string fname)
{	
	ofstream fout(fname); 
	DadaWSUtil::ExportMamaGraph(m, fout);
	fout.close();
}

void DadaWSUtil::ExportMamaGraph(MamaGraph &m, std::ofstream &fout)
{	
	dynamic_properties dp;
	dp.property("x", get(&MamaVertexProp::x, m));
	dp.property("y", get(&MamaVertexProp::y, m));
	dp.property("label", get(&MamaVertexProp::label, m));
	dp.property("weight", get(&MamaVertexProp::weight, m));
	dp.property("weight", get(&MamaEdgeProp::weight, m));
	dp.property("wasWeights", get(&MamaEdgeProp::wasWeight, m));
	dp.property("connected", get(&MamaEdgeProp::connected, m));
	boost::write_graphml(fout, m, dp, true);
}

void DadaWSUtil::ImportMamaGraph(MamaGraph &m, string fname)
{
	ifstream fin(fname);
	DadaWSUtil::ImportMamaGraph(m, fin);
	fin.close();
}

void DadaWSUtil::ImportMamaGraph(MamaGraph &m, std::ifstream &fin)
{
	dynamic_properties dp;
	dp.property("x", get(&MamaVertexProp::x, m));
	dp.property("y", get(&MamaVertexProp::y, m));
	dp.property("label", get(&MamaVertexProp::label, m));
	dp.property("weight", get(&MamaVertexProp::weight, m));
	dp.property("weight", get(&MamaEdgeProp::weight, m));
	dp.property("wasWeights", get(&MamaEdgeProp::wasWeight, m));
	dp.property("connected", get(&MamaEdgeProp::connected, m));
	boost::read_graphml(fin, m, dp);
}


void DadaWSUtil::ChooseRandomFeatures(vector<int> &fv, int outOf, int subsetSize)
{
	fv.clear();
	for (int i = 0; i < outOf; i++) fv.push_back(i);
	if (subsetSize < 1) return;

	shuffle(fv.begin(), fv.end(), m_rgen);
	if (subsetSize < fv.size()) {
		fv.erase(fv.begin() + subsetSize, fv.end());
	}
}

void DadaWSUtil::ChooseRandomEdges(vector<MamaVId> &origv, MamaGraph &gp, int N, vector< pair<MamaVId, MamaVId> > &newe)
{	
	newe.clear(); 

	for (int i = 0; i < N; i++) {
		int v1 = rand() % origv.size(); 
		int v2 = rand() % origv.size();
		while (v1 == v2) v2 = rand() % origv.size();
		MamaEId eid; bool myEdge;
		std::tie(eid, myEdge) = edge(origv[v1], origv[v2], gp);
		if (!myEdge) {
			newe.resize(newe.size() + 1);
			newe[newe.size() - 1].first = origv[v1];
			newe[newe.size() - 1].second = origv[v2];
		}
	}	
}

void DadaWSUtil::ChooseDegreeEdges(vector<MamaVId> &origv, MamaGraph &gp, int ND, vector< pair<MamaVId, MamaVId> > &newe)
{
	newe.clear();	

	
	for (int i = 0; i < origv.size(); i++) {		
		for (int d = 0; d < ND; d++) {
			int v1;
			MamaEId eid;
			bool currentEdge = 1;
			while (currentEdge) {
				v1 = rand() % origv.size();
				std::tie(eid, currentEdge) = edge(origv[i], origv[v1], gp);
			}
			newe.resize(newe.size() + 1); 
			newe[newe.size()-1].first = origv[i];
			newe[newe.size()-1].second = origv[v1];
		}
	}
}

DadaWSUtil::DadaWSUtil()
{			
}

DadaWSUtil::~DadaWSUtil()
{	
}

