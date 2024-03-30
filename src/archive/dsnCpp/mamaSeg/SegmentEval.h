#if !defined(SegmentEval_H__)
#define SegmentEval_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "SegmentWS.h"
#include "Logger.h"
#include "SetElement.h"
#include <boost/tuple/tuple.hpp>


class SegmentEval 
{
	public:
		map< MamaVId, int > myVertexLabels; 
		map< MamaVId, map<int, double> > mySuperLabels;
		
		vector< pair< double, MamaEId > > sortedEdges; 
		vector< pair< double, MamaEId > > mstEdges; 
		
		vector<double> posCounts, negCounts;
		double extraNeg, extraPos;

		void InitGroundTruth(MamaGraph &myM, MamaGraph &myM2, cv::Mat &seg);
		void InitBaseGroundTruth(MamaGraph &myM, cv::Mat &seg);
		void InitSuperGroundTruth(MamaGraph &myM, MamaGraph &myM2);

		double TrainThreshold(double &thresh);

		void ComputeMaxMin(MamaGraph &myM);
		void GetMSTCounts( MamaGraph &myM, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount);		

		static double RandMSTError(double thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos = 0.0, double extraNeg = 0.0);
		static double RandMSTTrain(double &thresh, cv::Mat &result, cv::Mat &posCount, cv::Mat &negCount, double extraPos = 0.0, double extraNeg = 0.0);

		SegmentEval();
		virtual ~SegmentEval();
	private:				

		Logger m_logger;				
}; 

#endif 

