#include "SegmentEval.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"

#include <boost/pending/disjoint_sets.hpp>
#include <boost/pending/property.hpp>

void SegmentEval::InitGroundTruth(MamaGraph &myM, MamaGraph &myM2, cv::Mat &seg)
{
	this->InitBaseGroundTruth(myM, seg);
	this->InitSuperGroundTruth(myM, myM2);
}

void SegmentEval::InitBaseGroundTruth(MamaGraph &myM, cv::Mat &seg)
{
	this->myVertexLabels.clear();
	
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(myM);
	for (nit = nstart; nit != nend; nit++) {		
		float label = seg.at<float>(myM[*nit].y, myM[*nit].x);		
		this->myVertexLabels[*nit] = boost::numeric_cast<int>(label);
	}
}

void SegmentEval::InitSuperGroundTruth(MamaGraph &myM, MamaGraph &myM2)
{
	LOG_TRACE(m_logger, "InitBaseGroundTruth\n");
	this->mySuperLabels.clear();
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(myM2);
	for (nit = nstart; nit != nend; nit++) {
		this->mySuperLabels[*nit].clear(); 		
	}

	int totalCount = 0;
	int negCount = 0;
	std::tie(nstart, nend) = vertices(myM);
	for (nit = nstart; nit != nend; nit++) {
		int nid = myM[*nit].label;
		int gt = this->myVertexLabels[*nit];
		if (gt >= 0) {
			if (this->mySuperLabels[nid].count(gt) == 0) this->mySuperLabels[nid][gt] = 1;
			else this->mySuperLabels[nid][gt] = this->mySuperLabels[nid][gt] + 1;
			totalCount++;
		}
		else negCount;
	}
	
	// get error already iccured at lower level
	this->extraNeg = 0.0;
	this->extraPos = 0.0;

	for (nit = nstart; nit != nend; nit++) {
		double tsum = 0.0;
		double tssum = 0.0;
		// multiply matched entries in a and b and add them
		map<int, double>::iterator it;

		for (it = this->mySuperLabels[*nit].begin(); it != this->mySuperLabels[*nit].end(); ++it) {
			tssum += (it->second * it->second - it->second) / 2.0;
			tsum += it->second;
		}
		double totalCount = (tsum*tsum - tsum) / 2.0;
		this->extraPos += tssum; 
		this->extraNeg += (totalCount - tssum);
	}
	LOG_TRACE(m_logger, "   GROUND TRUTH " << totalCount << " labels and " << negCount << " unlabeled and extra costs " << this->extraPos << "+ and " << this->extraNeg << "-\n");
	LOG_TRACE(m_logger, "InitBaseGroundTruth Done\n");

}

void SegmentEval::ComputeMaxMin(MamaGraph &myM2)
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
		elements.push_back( SetElement(i, this->mySuperLabels[*nit]) );
        nodeIds[i] = i; 
        dsets.make_set(i);
		i++;
    }
    	
	this->mstEdges.clear();
	this->posCounts.clear();
	this->negCounts.clear();

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
				            
            // Merge the sets
            dsets.link(si1, si2);
            elements[dsets.find_set(si1)].AddLabelCounts(set1.labelCount, set2.labelCount);						
        } 		        
	}    
		
	LOG_TRACE(m_logger, "ComputeMaxMin Done");
}

double SegmentEval::TrainThreshold(double &thresh)
{
	//LOG_TRACE(m_logger, "PreSortedTrain");	
	double sumPosCount = 0.0, sumNegCount = 0.0;
	for (int i = 0; i < this->posCounts.size(); i++) {
		sumPosCount += posCounts[i];
		sumNegCount += negCounts[i];
	}
	double posTotal = sumPosCount + this->extraPos;
	double negTotal = sumNegCount + this->extraNeg;

	double posError = sumPosCount;
	double negError = this->extraNeg;

	double bestThresh = this->mstEdges[0].first - 1.0e-8;
	double bestError = (posError + negError) / (posTotal + negTotal);
	//int candi = 0;
	for (int i = 0; i < (this->posCounts.size() - 1); i++) {
		posError = posError - this->posCounts[i]; 
		negError = negError + this->negCounts[i];

		double val1 = this->mstEdges[i].first;
		double val2 = this->mstEdges[i+1].first;

		if (abs(val1 - val2) > 1.0e-8) {
			//candi++;
			double newError = (posError + negError) / (posTotal + negTotal);
			if (newError < bestError) {
				bestError = newError;
				bestThresh = (val1 + val2) / 2.0;
			}
		}
	}

	//cout << "Train thresh has " << candi << " out of " << posCount.rows << "\n";
	// Now check final
	posError = posError - this->posCounts[this->posCounts.size() - 1];
	negError = negError + this->negCounts[this->negCounts.size() - 1];
	double newError = (posError + negError) / (posTotal + negTotal);
	if (newError < bestError) {
		bestError = newError;
		bestThresh = this->mstEdges[this->mstEdges.size() - 1].first + 1.0e-8;
	}
	thresh = bestThresh;
	return(bestError);
}


void SegmentEval::GetMSTCounts( MamaGraph &myM, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount)
{			
	result = Mat::zeros(mstEdges.size(), 1, CV_64F); 
	posCount = Mat::zeros(mstEdges.size(), 1, CV_64F); 
	negCount = Mat::zeros(mstEdges.size(), 1, CV_64F); 
	
	for(int i=0; i< mstEdges.size(); i++) {
		result.at<double>(i) = this->mstEdges[i].first;
		posCount.at<double>(i) = this->posCounts[i]; 
		negCount.at<double>(i) = this->negCounts[i]; 
	}
}


double SegmentEval::RandMSTTrain(double &thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos, double extraNeg)
{
	//LOG_TRACE(m_logger, "PreSortedTrain");	
	double sumPosCount = 0.0, sumNegCount = 0.0;
	for (int i = 0; i < posCount.rows; i++) {
		sumPosCount += posCount.at<double>(i);
		sumNegCount += negCount.at<double>(i);
	}
	double posTotal = sumPosCount + extraPos;
	double negTotal = sumNegCount + extraNeg;

	double posError = sumPosCount;
	double negError = extraNeg;

	double bestThresh = result.at<double>(0) - 1.0e-8;
	double bestError = (posError + negError) / (posTotal + negTotal);
	//int candi = 0;
	for (int i = 0; i < (posCount.rows - 1); i++) {
		posError = posError - posCount.at<double>(i);
		negError = negError + negCount.at<double>(i);

		double val1 = result.at<double>(i);
		double val2 = result.at<double>(i + 1);

		if (abs(val1 - val2) > 1.0e-8) {
			//candi++;
			double newError = (posError + negError) / (posTotal + negTotal);
			if (newError < bestError) {
				bestError = newError;
				bestThresh = (val1 + val2) / 2.0;
			}
		}
	}

	//cout << "Train thresh has " << candi << " out of " << posCount.rows << "\n";
	// Now check final
	posError = posError - posCount.at<double>(posCount.rows - 1);
	negError = negError + negCount.at<double>(negCount.rows - 1);
	double newError = (posError + negError) / (posTotal + negTotal);
	if (newError < bestError) {
		bestError = newError;
		bestThresh = result.at<double>(result.rows - 1) + 1.0e-8;
	}
	thresh = bestThresh;
	return(bestError);
}

double SegmentEval::RandMSTError(double thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos, double extraNeg)
{
	//LOG_TRACE(m_logger, "RandMSTError");
	double posError = 0.0;
	double negError = 0.0;
	double posTotal = 0.0;
	double negTotal = 0.0;
	//cout << result.rows << " rows in result\n";
	for (int i = 0; i< result.rows; i++) {
		posTotal += (double)posCount.at<double>(i);
		negTotal += (double)negCount.at<double>(i);
		if (result.at<double>(i) < thresh) {
			negError += (double)negCount.at<double>(i);
		}
		else {
			posError += (double)posCount.at<double>(i);
		}
	}
	//cout << posTotal << " and " << negTotal << " errors\n";
	//double ri = (negError+posError) / (negTotal + posTotal); 
	double ri = (negError + posError + extraNeg) / (negTotal + posTotal + extraPos + extraNeg);

	//if (totalPos) *totalPos = posTotal + extraPos;
	//if (totalNeg) *totalNeg = negTotal + extraNeg;

	return(ri);
	//LOG_TRACE(m_logger, "RandMSTError Done");
}

SegmentEval::SegmentEval() : m_logger(LOG_GET_LOGGER("SegmentEval"))
{			
}

SegmentEval::~SegmentEval()
{	
}

