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

#include "DadaIID.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"

Logger DadaIID::s_logger(LOG_GET_LOGGER("DadaIID"));

void DadaIID::TrainThreshold(double &thresh, MamaGraph &myM, DadaWSGT &gt, DadaError &err)
{
	LOG_TRACE(s_logger, "TrainThreshold");

	vector< pair< double, MamaEId > > myEdges;
	map<MamaEId, double> myLabels; 


	double totalPos = 0.0; 
	double totalNeg = 0.0;

	myLabels.clear(); 
	// sort the edges		
	myEdges.resize(num_edges(myM));

	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(myM);
	int i = 0;
	for (eit = estart; eit != eend; eit++) {
		pair< double, MamaEId > eid = make_pair(myM[*eit].weight, *eit);
		myEdges[i] = eid;
		i++;
		MamaVId id1 = source(*eit, myM);
		MamaVId id2 = target(*eit, myM);

		map<int, double> &l1 = gt.Labels()[id1];
		map<int, double> &l2 = gt.Labels()[id2];
		
		double posCount = 0.0; 		
		double count1 = 0.0; 
		double count2 = 0.0;

		for (auto &it1 : l1) {
			count1 += it1.second;
			if (l2.count(it1.first)) {
				posCount += (it1.second * l2[it1.first]);				
			}
		}

		for (auto &it2 : l2) {
			count2 += it2.second;
		}
		double negCount = count1 * count2 - posCount; 
		if (posCount > negCount) {
			myLabels[*eit] = 1.0; 
			totalPos += 1.0; 
		}
		else if (posCount < negCount) {
			myLabels[*eit] = -1.0;
			totalNeg += 1.0;
		}
	}
	// smallest to larget... everything below threshold is merged
	sort(myEdges.begin(), myEdges.end(), std::less< pair<double, MamaEId> >());

	// Now thrain threshold

	double posError, negError; 

	posError = totalPos; 
	negError = 0.0; 

	double bestThresh = myEdges[0].first - 1.0e-8;
	double bestError = (posError + negError) / (totalPos + totalNeg);
	double bestPosError = posError; 
	double bestNegError = negError;
	//int candi = 0;
	for (int i = 0; i < (myEdges.size() - 1); i++) {
		
		MamaEId eid = myEdges[i].second;
		if (myLabels.count(eid)) {
			if (myLabels[eid] > 0.0) {
				posError -= 1.0; 
			}
			else {
				negError += 1.0; 
			}
		}

		double val1 = myEdges[i].first;
		double val2 = myEdges[i+1].first;

		if (abs(val1 - val2) > 1.0e-8) {
			//candi++;
			double newError = (posError + negError) / (totalPos + totalNeg);
			if (newError < bestError) {
				bestError = newError;
				bestThresh = (val1 + val2) / 2.0;
				//bestPosError = localPos; 
				//bestNegError = localNeg;
				bestPosError = posError;
				bestNegError = negError;

			}
		}
	}

	//cout << "Train thresh has " << candi << " out of " << posCount.rows << "\n";
	// Now check final
	MamaEId eid = myEdges[myEdges.size()-1].second;
	if (myLabels.count(eid)) {
		if (myLabels[eid] > 0.0) {
			posError -= 1.0;
		}
		else {
			negError += 1.0;
		}
	}


	double newError = (posError + negError) / (totalPos + totalNeg);
	if (newError < bestError) {
		bestError = newError;
		bestThresh = myEdges[myEdges.size() - 1].first + 1.0e-8;
		bestPosError = posError;
		bestNegError = negError;
	}
	thresh = bestThresh;
	err.GetError() = bestError; 
	err.GetPosError() = bestPosError;
	err.GetNegError() = bestNegError;	

	//LOG_INFO(m_logger, "    T: " << bestThresh << " from " << this->mstEdges[0].first << "->" << this->mstEdges[this->mstEdges.size() - 1].first);
}


DadaIID::DadaIID() : m_logger(LOG_GET_LOGGER("DadaIID"))
{			
}

DadaIID::~DadaIID()
{	
}

