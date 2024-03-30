#if !defined(DadaEval_H__)
#define DadaEval_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "SegmentWS.h"
#include "Logger.h"
#include "SetElement.h"
#include <boost/tuple/tuple.hpp>
#include "DadaWSGT.h"

class DadaError
{
public:
	double & GetError() { return(m_error); }; 
	double & GetPosError() { return(m_posError); };
	double & GetNegError() { return(m_negError); };

	double m_error;
	double m_posError; 
	double m_negError;

	DadaError(); 
	virtual ~DadaError();
};

class DadaEval 
{
	public:		

		void ComputeMaxMin(MamaGraph &myM, map< MamaVId, map<int, double> > &vertexLabels);
				
		void TrainThreshold(double &thresh, DadaWSGT &gt, DadaError &err);		
		void TrainWeightedThreshold(double &thresh, DadaWSGT &gt, DadaError &err);

		void RandMSTError(DadaWSGT &gt, DadaError &err);
		static void MatRandError(cv::Mat &seg1, cv::Mat &seg2, DadaError &error);
		static void MatRandTest(cv::Mat &seg1, cv::Mat &seg2, DadaError &error);

		double GetCCGradient(MamaGraph &myM, std::map<MamaEId, cv::Mat> &features, double thresh, cv::Mat &gradient);		
		double GetCCLogisticLoss(MamaGraph &myM, std::map<MamaEId, cv::Mat> &features, double thresh, cv::Mat &gradient, DadaWSGT &gt, double *gradientThresh = nullptr);
		double GetWSGradient(MamaGraph &myM, std::map<MamaEId, cv::Mat> &features, std::map<MamaEId, MamaEId> &wsNeighbor, double thresh, cv::Mat &gradient);

		double GetTotalPos() { return(m_totalPos); }; 
		double GetTotalNeg() { return(m_totalNeg); };
		double GetError() { return(m_error); };		

		DadaEval();
		virtual ~DadaEval();
	private:				
		vector< pair< double, MamaEId > > sortedEdges;
		vector< pair< double, MamaEId > > mstEdges;
		vector<double> posCounts, negCounts;

		double m_totalPos; 
		double m_totalNeg;
		double m_error;
		Logger m_logger;	
		static Logger s_logger;

}; 

#endif 

