#include "SegmentTrain.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "VizMat.h"
#include "DadaDef.h"
#include "GdalMat.h"
#include "SegmentEval.h"

double SegmentTrain::FindBestParameter(cv::Mat &img, cv::Mat &seg, SegmentParameter &param)
{
	SegmentWS segmenter;
	SegmentEval evaluator;

	int preTypeUpto = 4;
	int preSizeUpto = 5;	
	int postTypeUpto = 4;
	int postSizeUpto = 5;
	int gradSize = 1;
	double scaleFactor = 1.0f;
	Mat segp;

	double bestError = LARGEST_FLOAT;
	SegmentParameter bestParam;

	for (int preType = 0; preType < preTypeUpto; preType++) {
		if (preType > 0) preSizeUpto = 5;
		else preSizeUpto = 1;
		for (int preSize = 1; preSize <= preSizeUpto; preSize+=2) {
			for (int gradType = 1; gradType <= 2; gradType++) {
				for (int postType = 0; postType < postTypeUpto; postType++) {
					if (postType > 0) postSizeUpto = 5;
					else postSizeUpto = 1;
					for (int postSize = 1; postSize <= postSizeUpto; postSize+=2) {
											
						for (int scaleType = 0; scaleType < 4; scaleType++) {
							if (scaleType == 0) {
								scaleFactor = 1.0;
							}
							else if (scaleType == 1) {
								scaleFactor = 0.75;
							}
							else if (scaleType == 2) {
								scaleFactor = 0.66;
							}
							else if (scaleType == 3) {
								scaleFactor = 0.5;
							}

							param.scaleFactor = scaleFactor; 
							param.preType = preType;
							param.preSize = preSize;
							param.postType = postType;
							param.postSize = postSize;
							param.gradType = gradType;
							param.gradSize = gradSize;
							param.absoluteThreshold = 1;
								segmenter.Init(img, param);
							segmenter.UpdateThreshold(param);
								evaluator.InitGroundTruth(segmenter.GetBasinGraph(), segmenter.GetGraph(), seg);
							evaluator.ComputeMaxMin(segmenter.GetGraph());

							double err = evaluator.TrainThreshold(param.threshold);
							//segmenter.UpdateThreshold(param);
							//segmenter.GetLabels(segp);
							//VizMat::DisplayColorSeg(segp, "current", 1, 1.0f);
							if (err < bestError) {
								bestError = err;
								bestParam = param;
								//segmenter.UpdateThreshold(param);
								//segmenter.GetLabels(segp);
								//VizMat::DisplayColorSeg(segp, "best", 1, 1.0f);
								//cout << "Best so far " << err << "\n";
								//param.Print();
								//param.Save("bestShip.yml");
								//FitsMat::Write2DFitsAsFloat("bestShip.fits", segp);
							}
						}
					}
				}
			}
		}
	}

	param = bestParam;
	return(bestError);
}

double SegmentTrain::FindBestThreshold(cv::Mat &img, cv::Mat &seg, SegmentParameter &param)
{
	SegmentWS segmenter;

	param.threshold = 0.5;
	param.absoluteThreshold = 1;

	segmenter.Init(img, param);
	segmenter.UpdateThreshold(param);

	SegmentEval evaluator;
	evaluator.InitGroundTruth(segmenter.GetBasinGraph(), segmenter.GetGraph(), seg);
	evaluator.ComputeMaxMin(segmenter.GetGraph());
	double err = evaluator.TrainThreshold(param.threshold);
	return(err);
}

SegmentTrain::SegmentTrain() : m_logger(LOG_GET_LOGGER("SegmentTrain"))
{			
}

SegmentTrain::~SegmentTrain()
{	
}

