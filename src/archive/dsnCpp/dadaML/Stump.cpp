#include "Stump.h"
#include "Info.h"

using namespace std;
static Info info;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaDef.h"
#include "MamaException.h"

#define FLATZONE_TOLERANCE	1.0e-10

double Stump::RandMSTTrain(double &thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos, double extraNeg)
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
	posError = posError - posCount.at<double>(posCount.rows-1);
	negError = negError + negCount.at<double>(negCount.rows-1);
	double newError = (posError + negError) / (posTotal + negTotal);
	if (newError < bestError) {
		bestError = newError;
		bestThresh = result.at<double>(result.rows-1) + 1.0e-8; 
	}
	thresh = bestThresh;	
	return(bestError);
}

double Stump::RandMSTError(double thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos, double extraNeg, double *totalPos, double *totalNeg)
{
	//LOG_TRACE(m_logger, "RandMSTError");
	double posError = 0.0;
	double negError = 0.0;
	double posTotal = 0.0;
	double negTotal = 0.0;
	//cout << result.rows << " rows in result\n";
	for(int i=0; i< result.rows; i++) {				
		posTotal += (double) posCount.at<double>(i);
		negTotal += (double) negCount.at<double>(i);
		if ( result.at<double>(i) < thresh ) {
			negError += (double)negCount.at<double>(i);			
		} else {
			posError += (double)posCount.at<double>(i);
		}
	}
	//cout << posTotal << " and " << negTotal << " errors\n";
	//double ri = (negError+posError) / (negTotal + posTotal); 
	double ri = (negError + posError + extraNeg) / (negTotal + posTotal + extraPos + extraNeg); 	

	if (totalPos) *totalPos = posTotal + extraPos;
	if (totalNeg) *totalNeg = negTotal + extraNeg;

	return( ri ); 
	//LOG_TRACE(m_logger, "RandMSTError Done");
}

void Stump::TrainThreshold(double &thresh, cv::Mat &result, cv::Mat &labels, cv::Mat &weights)
{
	int N = result.rows;

	
	vector< pair<double, double> > ds;
	ds.clear(); ds.resize( N ); 

	// lets take a look	
	for(int ni=0; ni< N; ni++) {						
		ds[ni].first = result.at<double>(ni); 		
		ds[ni].second = weights.at<double>(ni) * labels.at<double>(ni); 
	}

	//sort(ds.begin(), ds.end(), std::greater< pair<double, int> >());
	sort(ds.begin(), ds.end());
	//info(1, "Threshold weights are %f -> %f\n", ds[0].first, ds[ds.size()-1].first);

	double myAccWeight = 0.0f;
	double myBestWeight = 0.0f;
	int myBestIndex = -1;

	for(int ni=0; ni< N; ni++) {				
		//info(1, "%f for %f\n", ds[ni].second, myAccWeight); 
		myAccWeight += ds[ni].second;
		if (myAccWeight < myBestWeight) {
			myBestWeight = myAccWeight;
			myBestIndex  = ni;
		}
	}
	
	thresh = 0.0f;
	
	if (myBestIndex > -1) {
		if (myBestIndex < N-1 )  {
			thresh = (ds[myBestIndex].first + ds[myBestIndex+1].first)/2.0f;
			//info(1, "Yip yah %f!\n", this->myThresh);
		}
		else  thresh = ds[myBestIndex].first + 0.001f;
	} else thresh = ds[0].first - 0.001f;
	
	//info(10, "--------------->%f from [%f->%f]\n", thresh , ds[0].first, ds[ ds.size() - 1].first);	
	//info(100, "LearnNodeGraph::FindThreshold Done with %f\n", thresh );
}

double Stump::EvalThreshold(double &thresh, cv::Mat &result, cv::Mat &labels, cv::Mat &weights, double &dr, double &far)
{
	int N = result.rows;
	double pc = 0.0, nc = 0.0, tp = 0.0, fa = 0.0;

	for(int ni=0; ni< N; ni++) {				
		if ( labels.at<double>(ni) > 0.0f) {
			pc += weights.at<double>(ni);
			if ( result.at<double>(ni) - thresh > 0.0) {
				tp += weights.at<double>(ni);
			} 
		} else {
			nc += weights.at<double>(ni);
			if ( result.at<double>(ni) - thresh > 0.0) {
				fa += weights.at<double>(ni);
			}
		}
	}

	dr = (double) (tp / pc);
	fa = (double) (fa / nc); 

	double err = (pc - tp + fa) / (pc + nc);
	
	return( (double) err);
}


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
Stump::Stump()  : m_logger(LOG_GET_LOGGER("Stump"))
{	
}

Stump::~Stump()
{	
}




/*  
BACKUP: Pre extras

double Stump::RandMSTTrain(double &thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos, double extraNeg)
{
	//LOG_TRACE(m_logger, "PreSortedTrain");	
	vector<double> cresult; 
	vector<double> cpos, cneg;
	cresult.clear(); cpos.clear(); cneg.clear();

	double posTotal = 0.0;
	double negTotal = 0.0;
	
	int vali = 0;
	double valf = result.at<double>(0);
	cresult.push_back( result.at<double>(0) );
	cpos.push_back( (double) posCount.at<double>(0) );
	cneg.push_back( (double) negCount.at<double>(0) );
	
	for(int i=1; i < posCount.rows; i++) {		
		double nextf = result.at<double>(i);

		if ( fabs(valf - nextf) <  FLATZONE_TOLERANCE) {
			cpos[vali] += (double) posCount.at<double>(i);
			cneg[vali] += (double) negCount.at<double>(i);
		} else {
			cresult.push_back( nextf );
			cpos.push_back( (double) posCount.at<double>(i) );
			cneg.push_back( (double) negCount.at<double>(i) );
			vali = cneg.size()-1;
			valf = nextf;
		}
		posTotal += (double) posCount.at<double>(i);
		negTotal += (double) negCount.at<double>(i);
	}
	
	double posError = posTotal;
	double negError = 0.0;
	
	//double ri = (negError + posError + extraNeg) / (negTotal + posTotal + extraPos + extraNeg); 	

	double myRun = posError + negError;	
	double myBestPos = posError; 
	double myBestNeg = negError;	
	double myBest = myRun; 
	int myBestIndex = -1;
	//cout << "Was " << result.rows << " now " << cresult.size() << "\n";
	for(int i=0; i < cresult.size(); i++) {		
		posError -= cpos[i];
		negError += cneg[i]; 
		myRun = posError + negError;
		//double ri = myRun / (posTotal + negTotal);
		//cout << cresult[i] << " -> " << myRun << "(" << negError << "-ve, " << posError << "+ve\n";
		if (myRun < myBest) {
			myBest = myRun; 
			myBestPos = posError; 
			myBestNeg = negError; 
			myBestIndex = i;
		}
	}

	thresh = 0.0f;
	
	if (myBestIndex > -1) {
		if (myBestIndex < cresult.size() )  {
			thresh = (cresult[myBestIndex] + cresult[myBestIndex+1])/2.0f;			
		} else  thresh = cresult[myBestIndex] + 0.001f;
	} else thresh = cresult[0] - 0.001f;
		
	//cout << "Thresh " << thresh << " from " << cresult[0] << " -> " << cresult[cresult.size()-1] << "\n";	
	//LOG_INFO(m_logger, this->myThresh << " from " << myBestPos << "+ve, " << myBestNeg << "-ve\n"); 
	//LOG_TRACE(m_logger, "PreSortedTrain Done");

	return(thresh);
}

*/