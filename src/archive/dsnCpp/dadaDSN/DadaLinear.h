#if !defined(WSLinearTrain_H__)
#define WSLinearTrain_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "WSGraph.h"
#include "WSTrain.h"
#include "Logger.h"


class WSLinearTrain : public WSTrain
{
	public:
		cv::Mat weights, gradient; 

		void TrainInit(WSTrainParam &param);
		void TrainIteration(WSMaxMin &minMax, WSGraph &aGraph, WSTrainParam &param);
		void TrainFinalize(WSTrainParam &param);

		void ApplyInit(WSTrainParam &param);
		void Apply(WSGraph &aGraph, WSTrainParam &param);		

		void EstimateWSGradient(map<MergeEId, double> &trainEdges, WSGraph &aGraph, WSTrainParam &param, double totalK);
		//void EstimateWSGradient(map<MergeEId, pair<double, double> > &trainEdges, WSGraph &aGraph, WSTrainParam &param);
		void EstimateGradient(WSMaxMin &myMaxMin, WSGraph &aGraph, WSTrainParam &param);
		void UpdateWeight(WSMaxMin &myMaxMin, WSGraph &aGraph, WSTrainParam &param);
		double NormalizeWeights(WSTrainParam &param);		

		void MakeCopy(boost::shared_ptr<WSTrain> &dest);
		void Print(WSTrainParam &param);

		void Save(WSTrainParam &param);
		void Load(WSTrainParam &param);

		WSLinearTrain();
		virtual ~WSLinearTrain();
	private:				

		Logger m_logger;				
}; 


#endif 

