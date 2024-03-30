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

#include "DadaEval.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"

#include <boost/pending/disjoint_sets.hpp>
#include <boost/pending/property.hpp>

Logger DadaEval::s_logger(LOG_GET_LOGGER("DadaEval"));

void DadaEval::ComputeMaxMin(MamaGraph &myM2, map< MamaVId, map<int, double> > &vertexLabels)
{
	LOG_TRACE(m_logger, "ComputeMaxMin");
	
	// sort the edges		
	sortedEdges.resize( num_edges(myM2) );

	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(myM2); 
	int i = 0; 
	for(eit = estart; eit != eend; eit++) {			
		pair< double, MamaEId > eid = make_pair(myM2[*eit].weight, *eit);
		sortedEdges[i] = eid;
		i++;
	}
	// smallest to larget... everything below threshold is merged
	sort(sortedEdges.begin(), sortedEdges.end(), std::less< pair<double, MamaEId> >());

    //LOG_INFO(m_logger, "MaxMin sort is " << sortedEdges[0].first << " -> " << sortedEdges[sortedEdges.size()-1].first << "\n");
	//LOG_INFO(m_logger, "MaxMin: I have " << num_edges(myM) << " edges here\n"); 
    // Set up disjoint_sets.
    vector<SetElement> elements; elements.clear(); 
    vector<int> nodeIds( num_vertices( myM2 ) );     
    vector<int> rank( num_vertices( myM2 ) ); 
    vector<int> parent( num_vertices( myM2 ) ); 
    
    boost::disjoint_sets<int*, int*> dsets(&rank[0], &parent[0]);
	
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices( myM2 ); 		
	i = 0;
	for(nit = nstart; nit != nend; nit++) {
		elements.push_back(SetElement(i, vertexLabels[*nit]));
        nodeIds[i] = i; 
        dsets.make_set(i);
		i++;
    }
    	
	this->mstEdges.clear();
	this->posCounts.clear();
	this->negCounts.clear();
	this->m_totalPos = 0.0; 
	this->m_totalNeg = 0.0;
	int nc = 0;
	for (i = 0; i< num_edges(myM2); i++) {
        MamaEId eid = sortedEdges[i].second;
		MamaVId id1 = source(eid, myM2);
		MamaVId id2 = target(eid, myM2);

        int i1 = (int) id1; 
        int i2 = (int) id2; 
        
        // using disjoint_set instead
        int si1 = dsets.find_set(elements[i1].nodeID);
        int si2 = dsets.find_set(elements[i2].nodeID);
        
        if ( si1 != si2 ) {
            nc++;			            
            // Now for the maxmin path
            SetElement set1 = elements[si1];
            SetElement set2 = elements[si2];
            double labelAgreement = SetElement::DotProductLabels(set1.labelCount, set2.labelCount);
			
			this->mstEdges.push_back(sortedEdges[i]);
			this->posCounts.push_back(labelAgreement);
			this->negCounts.push_back(set1.GetNumberOfItems() * set2.GetNumberOfItems() - labelAgreement);
			this->m_totalPos += this->posCounts[this->posCounts.size() - 1];
			this->m_totalNeg += this->negCounts[this->negCounts.size() - 1];
            // Merge the sets
            dsets.link(si1, si2);
            elements[dsets.find_set(si1)].AddLabelCounts(set1.labelCount, set2.labelCount);						
        } 		        
	}    
		
	LOG_TRACE(m_logger, "ComputeMaxMin Done");
}

void DadaEval::TrainThreshold(double &thresh, DadaWSGT &gt, DadaError &err)
{
	//LOG_INFO(m_logger, "Train Thresh"); 

	double sumPosCount = 0.0, sumNegCount = 0.0;
	for (int i = 0; i < this->posCounts.size(); i++) {
		sumPosCount += posCounts[i];
		sumNegCount += negCounts[i];
	}
	m_totalPos = sumPosCount + gt.ExtraPos();
	m_totalNeg = sumNegCount + gt.ExtraNeg();

	double posError, negError; 

	posError = sumPosCount + gt.ErrorPos();
	negError = gt.ErrorNeg();

	double localPos, localNeg;
	localPos = sumPosCount; 
	localNeg = 0.0; 

	double bestThresh = this->mstEdges[0].first - 1.0e-8;
	double bestError = (posError + negError) / (m_totalPos + m_totalNeg);

	double bestPosError = localPos; 
	double bestNegError = localNeg;
	//int candi = 0;
	for (int i = 0; i < (this->posCounts.size() - 1); i++) {
		posError = posError - this->posCounts[i]; 
		negError = negError + this->negCounts[i];
		
		localPos = localPos - this->posCounts[i];
		localNeg = localNeg + this->negCounts[i];

		double val1 = this->mstEdges[i].first;
		double val2 = this->mstEdges[i+1].first;

		if (abs(val1 - val2) > 1.0e-8) {
			//candi++;
			double newError = (posError + negError) / (m_totalPos + m_totalNeg);
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
	posError = posError - this->posCounts[this->posCounts.size() - 1];
	negError = negError + this->negCounts[this->negCounts.size() - 1];
	double newError = (posError + negError) / (m_totalPos + m_totalNeg);
	if (newError < bestError) {
		bestError = newError;
		bestThresh = this->mstEdges[this->mstEdges.size() - 1].first + 1.0e-8;
		bestPosError = posError;
		bestNegError = negError;
	}
	thresh = bestThresh;
	m_error = bestError; 

	
	err.GetError() = bestError; 
	err.GetPosError() = bestPosError;
	err.GetNegError() = bestNegError;	

	//LOG_INFO(m_logger, "    T: " << bestThresh << " from " << this->mstEdges[0].first << "->" << this->mstEdges[this->mstEdges.size() - 1].first);
}


void DadaEval::TrainWeightedThreshold(double &thresh, DadaWSGT &gt, DadaError &err)
{
	double posWeight = gt.PosWeight();
	double negWeight = gt.NegWeight();

	//LOG_INFO(m_logger, "TrainWeighted Thresh has " << posWeight << ", " << negWeight);
	double sumPosCount = 0.0, sumNegCount = 0.0;
	for (int i = 0; i < this->posCounts.size(); i++) {
		sumPosCount += posCounts[i];
		sumNegCount += negCounts[i];
	}
	m_totalPos = sumPosCount + gt.ExtraPos();
	m_totalNeg = sumNegCount + gt.ExtraNeg();

	double posError, negError;

	posError = sumPosCount + gt.ErrorPos();
	negError = gt.ErrorNeg();
	
	double localPos, localNeg;
	localPos = sumPosCount;
	localNeg = 0.0;

	double bestThresh = this->mstEdges[0].first - 1.0e-8;
	double bestError = (posError + negError) / (m_totalPos + m_totalNeg);

	double bestPosError = localPos;
	double bestNegError = localNeg;
	//int candi = 0;
	for (int i = 0; i < (this->posCounts.size() - 1); i++) {
		posError = posError - this->posCounts[i];
		negError = negError + this->negCounts[i];

		localPos = localPos - this->posCounts[i];
		localNeg = localNeg + this->negCounts[i];

		double val1 = this->mstEdges[i].first;
		double val2 = this->mstEdges[i + 1].first;

		if (abs(val1 - val2) > 1.0e-8) {
			//candi++;
			double newError = ((posError * posWeight) + (negError* negWeight) ) / ( (m_totalPos * posWeight) + (m_totalNeg * negWeight));
			if (newError < bestError) {
				bestError = newError;
				bestThresh = (val1 + val2) / 2.0;
				bestPosError = localPos;
				bestNegError = localNeg;
			}
		}
	}

	//cout << "Train thresh has " << candi << " out of " << posCount.rows << "\n";
	// Now check final
	posError = posError - this->posCounts[this->posCounts.size() - 1];
	negError = negError + this->negCounts[this->negCounts.size() - 1];	
	double newError = ((posError * posWeight) + (negError* negWeight)) / ((m_totalPos * posWeight) + (m_totalNeg * negWeight));
	if (newError < bestError) {
		bestError = newError;
		bestThresh = this->mstEdges[this->mstEdges.size() - 1].first + 1.0e-8;
	}
	thresh = bestThresh;
	m_error = bestError;

	err.GetError() = bestError;
	err.GetPosError() = bestPosError;
	err.GetNegError() = bestNegError;

	//LOG_INFO(m_logger, "    T: " << bestThresh << " from " << this->mstEdges[0].first << "->" << this->mstEdges[this->mstEdges.size() - 1].first);
}

void DadaEval::RandMSTError(DadaWSGT &gt, DadaError &err)
{
	LOG_TRACE_METHOD(m_logger, "RandMSTError");

	double posError = 0.0;
	double negError = 0.0;
	double posTotal = 0.0;
	double negTotal = 0.0;
	//cout << result.rows << " rows in result\n";
	for (int i = 0; i< this->posCounts.size(); i++) {
		posTotal += this->posCounts[i]; 
		negTotal += this->negCounts[i];

		if (this->mstEdges[i].first < 0.0) {
			negError += this->negCounts[i];
		}
		else {
			posError += this->posCounts[i];
		}
	}

	err.GetError() = (negError + posError + gt.ErrorNeg() + gt.ErrorPos()) / (negTotal + posTotal + gt.ExtraNeg() + gt.ExtraPos());
	err.GetPosError() = posError + gt.ErrorPos(); 
	err.GetNegError() = negError + gt.ErrorNeg();

}

double DadaEval::GetCCGradient(MamaGraph &myM, std::map<MamaEId, cv::Mat> &features, double thresh, cv::Mat &gradient)
{
	LOG_TRACE_METHOD(m_logger, "GetCCGradients");

	double target1 = thresh;
	double target2 = thresh;

	vector< double > edgeE(mstEdges.size());		// local index so we can access mstEdges and wEdges
	vector< double > edgeK(mstEdges.size());		// local index so we can access mstEdges and wEdges
		
		
	double totalCount = 0.0;
	double totalError = 0.0;
	double totalPError = 0.0;
	double totalNError = 0.0;
	double totalPCount = 0.0;
	double totalNCount = 0.0;
	double totalPWeight = 0.0;
	double totalNWeight = 0.0;

	for (int i = 0; i < mstEdges.size(); i++) {
		double resultp = (double)mstEdges[i].first;
		// handle conflicts
		double pCount = this->posCounts[i];
		double nCount = this->negCounts[i];
		
		totalCount += (pCount+nCount);		
		totalPCount += pCount;
		totalNCount += nCount;

		edgeE[i] = 0.0;			
		edgeK[i] = abs(nCount - pCount);		

		if (nCount > pCount) {		// I do not want it merged
			if (resultp < target1) {		// its predicted merge						
				edgeE[i] = edgeK[i];
				totalError += nCount; 
				totalNError += nCount;
				totalNWeight += edgeK[i];
			}
		}
		if (pCount > nCount) {		// I want it merged
			if (resultp > target2) {		// its not going to be merged						
				edgeE[i] = -(edgeK[i]);
				totalError += pCount;
				totalPError += pCount;
				totalPWeight += edgeK[i];
			}
		}
	}

	int D = features[mstEdges[0].second].cols;
	//LOG_INFO(m_logger, "Estimating gradient of dimension " << D);

	Mat gradt, grads;
	gradt = Mat::zeros(D, 1, CV_64F);
	grads = Mat::zeros(D, 1, CV_64F);

	double totalVotes = 0.0;	

	for (int i = 0; i < mstEdges.size(); i++) {

		if (!CLOSE_ENOUGH(edgeE[i], 0.0)) {
			
			Mat grads = features[mstEdges[i].second].t();						
			
			Mat adje = grads * edgeE[i];
			
			gradt = gradt + adje;

			totalVotes += edgeK[i];

		}		
	}	
	// get average direction from the batch
	//gradient = gradt / totalVotes;	
	gradient = gradt / (m_totalPos + m_totalNeg);
	// This would normalize w.r.t. total error
	double myLoss = (totalError / totalCount);
	//gradient = gradient * myFracError; 
	return(myLoss); 	
}


double DadaEval::GetWSGradient(MamaGraph &myM, std::map<MamaEId, cv::Mat> &features, std::map<MamaEId, MamaEId> &wsNeighbor, double thresh, cv::Mat &gradient)
{
	LOG_TRACE_METHOD(m_logger, "GetWSGradient");
	
	double target1 = (thresh + 1.0);
	double target2 = (thresh - 1.0);

	vector< double > edgeE(mstEdges.size());		// local index so we can access mstEdges and wEdges
	vector< double > edgeK(mstEdges.size());		// local index so we can access mstEdges and wEdges
	double totalE = 0.0;
	double totalK = 0.0;
	double negError = 0.0;
	double posError = 0.0;
	double edgeCount = 0.0;
	double errorEdgeCount = 0.0;

	for (int i = 0; i < mstEdges.size(); i++) {
		double resultp = (double)mstEdges[i].first;
		// handle conflicts
		double pCount = this->posCounts[i];
		double nCount = this->negCounts[i];

		edgeE[i] = 0.0;
		edgeK[i] = abs(nCount - pCount);
		totalK += edgeK[i];

		if (nCount > pCount) {		// I do not want it merged
			if (resultp < target1) {		// its predicted merge						
				edgeE[i] = edgeK[i];
				negError += edgeK[i];
				totalE += edgeK[i];
				errorEdgeCount += 1.0;
			}
			edgeCount += 1.0;
		}
		if (pCount > nCount) {		// I want it merged
			if (resultp > target2) {		// its not going to be merged						
				edgeE[i] = -(edgeK[i]);
				posError += edgeK[i];
				totalE += edgeK[i];
				errorEdgeCount += 1.0;
			}
			edgeCount += 1.0;
		}
	}

	// Now go through and add up weight on edges
	map<MamaEId, double> combinedEdge;
	combinedEdge.clear();
	for (int i = 0; i < mstEdges.size(); i++) {
		if (!CLOSE_ENOUGH(edgeE[i], 0.0)) {		
			MamaEId eid = mstEdges[i].second;
			MamaEId nid = wsNeighbor[eid];
			if (combinedEdge.count(eid) == 0) combinedEdge[eid] = 0.0;
			if (combinedEdge.count(nid) == 0) combinedEdge[nid] = 0.0;

			combinedEdge[eid] = combinedEdge[eid] + edgeE[i];
			// then contribute to neighbor
			combinedEdge[nid] = combinedEdge[nid] - edgeE[i];
		}
	}

	int D = features[mstEdges[0].second].cols;
	LOG_INFO(m_logger, "Estimating gradient of dimension " << D);

	Mat gradt, grads;
	gradt = Mat::zeros(D, 1, CV_64F);
	grads = Mat::zeros(D, 1, CV_64F);
	double totalError = 0.0;
	double totalCount = 0.0;

	for (auto &it : combinedEdge) {

		if (!CLOSE_ENOUGH(it.second, 0.0)) {

			Mat grads = features[it.first];
			Mat adje = grads * it.second;
			totalError += abs(it.second);
			gradt = gradt + adje;
		}

	}

	gradient = gradt / totalK;
	return(0.0); 
}



double DadaEval::GetCCLogisticLoss(MamaGraph &myM, std::map<MamaEId, cv::Mat> &features, double thresh, cv::Mat &gradient, 
								   DadaWSGT &gt, double *gradientThresh)
{
	LOG_TRACE_METHOD(m_logger, "GetCCGradientLoss");

	int D = features.begin()->second.cols; 
	gradient = Mat::zeros(D, 1, CV_64F);
	if (gradientThresh) *gradientThresh = 0.0; 

	double totalPLoss = (m_totalPos + gt.ExtraPos()); 
	double totalNLoss = (m_totalNeg + gt.ExtraNeg());
	double totalContrib = 0.0; 
	//LOG_INFO(m_logger, "-----------------\nTotal loss " << totalPLoss << " and " << totalNLoss);

	double totalLoss = 0.0; 
	for (int i = 0; i < mstEdges.size(); i++) {
		
		double nLoss, nProb, pLoss, pProb;
		
		double resultp = (double)mstEdges[i].first;
		Mat grads = features[mstEdges[i].second].t();		

		// less than zero means merge
		if (gradientThresh == nullptr) resultp = resultp - thresh;		
		
		double resultn = -resultp; 

		if (resultp < -30) {
			nLoss = -resultp;
			nProb = 0;
		}
		else if (resultp > 30) {
			nLoss = 0;
			nProb = 1;
		}
		else {
			double temp = 1.0 + exp(-resultp);
			nLoss = log(temp);
			nProb = 1.0 / temp;
		}
		if (resultn < -30) {
			pLoss = -resultn;
			pProb = 0;
		}
		else if (resultn > 30) {
			pLoss = 0;
			pProb = 1;
		}
		else {
			double temp = 1.0 + exp(-resultn);
			pLoss = log(temp);
			pProb = 1.0 / temp;
		}

		//double nWeight = 1.0; 
		//double pWeight = 1.0; 
		double nWeight = this->negCounts[i]; 
		double pWeight = this->posCounts[i]; 

		double tLoss = ((pLoss*pWeight) + (nLoss*nWeight));		
		totalLoss += tLoss; 

		double nContrib = (1.0 - nProb) * nWeight;
		double pContrib = (1.0 - pProb) * pWeight;
		double tContrib = pContrib - nContrib; 
		totalContrib += tContrib; 
		
		gradient += (grads * tContrib);
		if (gradientThresh) *gradientThresh -= (tContrib);

	}
	
	totalLoss = totalLoss / (m_totalPos + m_totalNeg);
	gradient = gradient / (m_totalPos + m_totalNeg);
	
	if (gradientThresh) *gradientThresh = (*gradientThresh) / (m_totalPos + m_totalNeg);

	return(totalLoss);
}


void DadaEval::MatRandError(Mat &seg1, Mat &seg2, DadaError &error)
{
	double err, dr, fa, pc, nc;

	double tpc = 0.0; 
	double fnc = 0;
	pc = 0.0; 
	nc = 0.0;

	if (seg1.rows != seg2.rows) BOOST_THROW_EXCEPTION(UnexpectedSize());
	if (seg1.cols != seg2.cols) BOOST_THROW_EXCEPTION(UnexpectedSize());

	for (int n1i = 0; n1i < (seg1.rows*seg1.cols); n1i++) {
		for (int n2i = 0; n2i < (seg1.rows*seg1.cols); n2i++) {
			if (n1i < n2i) {
				float *s1p1 = (float *)seg1.ptr();  s1p1 += n1i;
				float *s1p2 = (float *)seg1.ptr();  s1p2 += n2i;
				float *s2p1 = (float *)seg2.ptr();  s2p1 += n1i;
				float *s2p2 = (float *)seg2.ptr();  s2p2 += n2i;

				if (CLOSE_ENOUGH(*s1p1, *s1p2)) {
					pc += 1.0;
					if (CLOSE_ENOUGH(*s2p1, *s2p2)) tpc += 1.0;
				}
				else {
					nc += 1.0;
					if (CLOSE_ENOUGH(*s2p1, *s2p2)) fnc += 1.0;
				}
			}
		}
	}

	err = (pc - tpc + fnc) / (pc + nc);
	dr = tpc / pc;
	fa = fnc / nc;
	error.GetError() = err;
	error.GetPosError() = pc - tpc; 
	error.GetNegError() = fnc; 
}


void DadaEval::MatRandTest(Mat &seg1, Mat &seg2, DadaError &error)
{
	if (seg1.rows != seg2.rows) BOOST_THROW_EXCEPTION(UnexpectedSize());
	if (seg1.cols != seg2.cols) BOOST_THROW_EXCEPTION(UnexpectedSize());

	map< int, map<int, double> > nodeLabels1;
	nodeLabels1.clear(); 	

	// pool seg1 as prediction, seg2 as ground truth
	for (int j = 0; j < seg1.rows; j++) {
		for (int i = 0; i < seg1.cols; i++) {
			int val1 = static_cast<int>(seg1.at<float>(j, i)); 
			int val2 = static_cast<int>(seg2.at<float>(j, i));

			if (!nodeLabels1.count(val1)) nodeLabels1[val1].clear(); 			

			if (!nodeLabels1[val1].count(val2)) {
				nodeLabels1[val1][val2] = 0.0; 
			}
			nodeLabels1[val1][val2] += 1.0; 
		}
	}

	double totalNodes = static_cast<double>(seg1.rows*seg1.cols);
	double totalPairs = (totalNodes * (totalNodes - 1)) / 2.0; 
	
	// estimate err1
	// for each node
	double totalNegCount = 0.0; 
	for (auto &it : nodeLabels1) {
		// for each label
		double nodeNegCount = 0.0; 
		for (auto &it1 : it.second) {
			for (auto &it2 : it.second) {
				// negcounts
				if (it1.first < it2.first) {
					nodeNegCount += (it1.second * it2.second);
				}
			}
		}
		totalNegCount += nodeNegCount; 
	}
	// Now poss error
	double totalPosCount = 0.0;
	for (auto &it1 : nodeLabels1) {
		for (auto &it2 : nodeLabels1) {
			if (it1.first < it2.first) {
				double nodePosCount = 0.0;
				for (auto &it11 : it1.second) {
					if (it2.second.count(it11.first)) {
						nodePosCount += (it11.second * it2.second[it11.first]);
					}
				}
				totalPosCount += nodePosCount; 
			}
		}
	}
	double totalError = (totalNegCount + totalPosCount) / totalPairs; 
	LOG_INFO(s_logger, "E: " << totalError << " from P: " << totalPosCount << ", N: " << totalNegCount);
	error.GetError() = totalError; 
	error.GetPosError() = totalPosCount; 
	error.GetNegError() = totalNegCount;
	//LOG_INFO(s_logger, "Rand Mat"); 
	//DadaEval::MatRandError(seg1, seg2, error);
	//LOG_INFO(s_logger, "R: " << error.GetError() << " from P: " << error.GetPosError() << ", N: " << error.GetNegError());


	//double err, dr, fa, pc, nc;
	//double tpc = 0.0;
	//double fnc = 0;
	//pc = 0.0;
	//nc = 0.0;
	//err = (pc - tpc + fnc) / (pc + nc);
	//dr = tpc / pc;
	//fa = fnc / nc;
	//error.GetError() = err;
	//error.GetPosError() = pc - tpc;
	//error.GetNegError() = fnc;
}

DadaEval::DadaEval() : m_logger(LOG_GET_LOGGER("DadaEval"))
{			
}

DadaEval::~DadaEval()
{	
}

DadaError::DadaError() 
{
}

DadaError::~DadaError()
{
}

