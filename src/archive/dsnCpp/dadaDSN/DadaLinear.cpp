#include "WSLinearTrain.h"
#include "Info.h"

using namespace std;
static Info info;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "WSLinearTrain.h"
#include "MamaDef.h"
#include "Stump.h"

void WSLinearTrain::TrainInit(WSTrainParam &param)
{
	LOG_TRACE(m_logger, "TrainInit");
	param.treep.threshold = 0.0f;
	
	if (param.initType == WS_INIT_UNIFORM) {		// uniform
		this->weights = Mat::zeros(param.numFeatures, 1, CV_64F);
		for (int i = 0; i < this->weights.rows; i++) {
			this->weights.at<double>(i) = (1.0 / boost::numeric_cast<double>(this->weights.rows));
		}
	} 
	else if (param.initType == WS_INIT_RANDOM) {	// random
		this->weights = Mat::zeros(param.numFeatures, 1, CV_64F);
		for (int i = 0; i < this->weights.rows; i++) {
			double rf = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
			double weightRange = 0.1; 
			this->weights.at<double>(i) = ((weightRange * rf) - weightRange/2.0);
		}
	}
	else BOOST_THROW_EXCEPTION(NotImplemented());
	
	LOG_TRACE(m_logger, "TrainInit Done");
}

void WSLinearTrain::TrainIteration(WSMaxMin &myMaxMin, WSGraph &aGraph, WSTrainParam &param)
{
	LOG_TRACE(m_logger, "TrainIteration");

	// Get the errors at current threshold
	if (param.segType == WS_WATER) {
		map<MergeEId, double> trainEdges;
		myMaxMin.GetWSErrors(aGraph.GetGraph(), param.treep.threshold, trainEdges);
		this->EstimateWSGradient(trainEdges, aGraph, param, myMaxMin.totalK);
	} 
	else {
		myMaxMin.GetMSTErrors(aGraph.GetGraph(), param.treep.threshold, param.segType, param.zeroThresh);
		this->EstimateGradient(myMaxMin, aGraph, param);
	}	
	
	this->UpdateWeight(myMaxMin, aGraph, param);

	LOG_TRACE(m_logger, "TrainIteration Done");
}

void WSLinearTrain::EstimateWSGradient(map<MergeEId, double> &trainEdges, WSGraph &aGraph, WSTrainParam &param, double totalK)
{
	LOG_TRACE(m_logger, "EstimateWSGradient");

	Mat gradt, grads;
	gradt = Mat::zeros(param.numFeatures, 1, CV_64F);
	grads = Mat::zeros(param.numFeatures, 1, CV_64F);
	double totalError = 0.0;	

	for (map<MergeEId, double>::iterator it = trainEdges.begin(); it != trainEdges.end(); it++) {

		if (!CLOSE_ENOUGH(it->second, 0.0)) {

			aGraph.GetFeatureVector(it->first, grads);
			//if (totalError < 0.1) {
			//cout << "------------------------------------\n";
			//cout << grads.t() << "\n";
			//cout << "------------------------------------\n";
			//}
			totalError += abs(it->second);
			//cout << "ERROR " << it->second.first << "\n";
			Mat adje = grads * it->second;
			gradt = gradt + adje;
		}
		
	}

	this->gradient = gradt / totalK;

	// for display purposes
	param.nWeight = totalError / totalK;
	param.pWeight = totalK;

	LOG_TRACE(m_logger, "EstimateWSGradient Done");
}

void WSLinearTrain::EstimateGradient(WSMaxMin &myMaxMin, WSGraph &aGraph, WSTrainParam &param)
{
	LOG_TRACE(m_logger, "EstimateGradient");
	
	Mat gradt, grads;
	gradt = Mat::zeros(param.numFeatures, 1, CV_64F);
	grads = Mat::zeros(param.numFeatures, 1, CV_64F);
	double totalError = 0.0;
	double totalCount = 0.0;

	for (int i = 0; i < myMaxMin.edgeError.size(); i++) {

		if (!CLOSE_ENOUGH(myMaxMin.edgeError[i], 0.0)) {
			
			// get the data
			//if (param.segType == WS_WATER) aGraph.GetDifferenceVector(myMaxMin.GetMST()[i].second, myMaxMin.GetWSI()[i], grads);
			//if (param.segType == WS_WATER) aGraph.GetDifferenceVector(myMaxMin.GetWSI()[i], myMaxMin.GetMST()[i].second, grads);
			//else 			
			//if (totalError < 0.1) {
				//cout << "------------------------------------\n";
				//cout << grads.t() << "\n";
				//cout << "------------------------------------\n";

			//}
			//if (param.zeroThresh) aGraph.GetDifferenceVector(myMaxMin.GetMST()[i].second, myMaxMin.GetWSI()[i], grads);
			
			aGraph.GetFeatureVector(myMaxMin.GetMST()[i].second, grads);

			totalError += abs(myMaxMin.edgeError[i]);
			
			//cout << "ERROR " << myMaxMin.edgeError[i] << "\n";
			Mat adje = grads * myMaxMin.edgeError[i];
			gradt = gradt + adje;
		}

		totalCount += myMaxMin.edgeK[i];		
	}

	this->gradient = gradt / totalCount; 
	
	// for display purposes
	param.nWeight = totalError / totalCount;
	param.pWeight = totalCount;
	
	LOG_TRACE(m_logger, "EstimateGradient Done");
}

void WSLinearTrain::UpdateWeight(WSMaxMin &myMaxMin, WSGraph &aGraph, WSTrainParam &param)
{	
	LOG_TRACE(m_logger, "UpdateWeight");
	
	if (this->gradient.empty()) {
		LOG_INFO(m_logger, "!");
		return;
	}

	double lrate, irate; 

	if (param.updateMode == WS_UPDATE_SVM) {
		//lrate = param.alpha / (1.0 + param.alpha * param.lambda * ((double)param.currentIter + 1.0));
		lrate = 1.0 / (param.lambda * ((double)param.currentIter + 1.0));
		irate = 1.0 - (1.0 / ((double)param.currentIter + 1.0));
	}
	else {
		lrate = param.alpha / (1.0 + param.alpha * param.lambda * ((double)param.currentIter + 1.0));
		//lrate = param.lambda;
		//irate = 1.0 - param.lambda;
		irate = 1.0 - lrate; 
	}

	Mat orig = this->weights.clone();
	Mat adj = this->gradient * lrate;	
	Mat adjw = orig * irate;
	this->weights = adjw + adj;
	
	//Mat orig = this->weights.clone();
	//Mat adj = this->gradient * param.lambda;
	
	//double minVal, maxVal;
	//minMaxLoc(orig, &minVal, &maxVal); //
	//double minGrad, maxGrad;
	//minMaxLoc(adj, &minGrad, &maxGrad); //
	//double myMult = 1.0;	
	//if (minGrad < 0.0) {
	//	double abNeg = abs(minGrad); 
	//	if (abNeg > 0.9*minVal) {	// need to attenuate negatives
	//		double myTarget = 0.9*minVal;
	//		myMult = myTarget / abNeg;
	//	}
	//}
	//adj = adj * myMult;
	//this->weights = orig + adj;
	//double minYada, maxYada;
	//minMaxLoc(this->weights, &minYada, &maxYada); //

	//cout << "UPDATEEEE\n";
	//cout << "Min weight " << minVal << " and grad " << minGrad << " let to mult " << myMult << " and new min " << minYada << "\n";

	//double myMult = 1.0f; 	
	//if (param.normalizeWeights > 0) 
	//myMult = this->NormalizeWeights(param);
	//LOG_INFO(m_logger, " R: " << irate << ", " << lrate << " and Norm: " << myMult << "\n");

	
	if (param.printUpdate) {
		LOG_INFO(m_logger, "------------------------------------------\n");
		LOG_INFO(m_logger, "Rates: " << irate << ", " << lrate); // << " and Norm: " << myMult << "\n");
		LOG_INFO(m_logger, " O        G         AW         AG       FF     DIFF\n");
		for (int i = 0; i < weights.rows; i++) {
			LOG_INFO(m_logger, std::fixed << std::setprecision(5) <<
				orig.at<double>(i) << " " <<
				gradient.at<double>(i) << " " <<
				adjw.at<double>(i) << " " <<
				adj.at<double>(i) << "  |  " <<
				weights.at<double>(i) << "  |  " << 
				(weights.at<double>(i) - orig.at<double>(i)) << "\n");
		}
		LOG_INFO(m_logger, "------------------------------------------\n");
	}
	

	LOG_TRACE(m_logger, "UpdateWeight Done");
}

double WSLinearTrain::NormalizeWeights(WSTrainParam &param)
{
	LOG_TRACE(m_logger, "NormalizeMag");

	double total = 0.0;
	for (int i = 0; i < this->weights.rows; i++) {
		total += (this->weights.at<double>(i) * this->weights.at<double>(i));
	}
	total = sqrt(total);
	if (total > 0.0) {
		this->weights = this->weights / total;
	}
	LOG_TRACE(m_logger, "NormalizeMag Done");
	return(total);
/*
	double total = 0.0;
	for (int i = 0; i < this->weights.rows; i++) {
		total += (this->weights.at<double>(i) * this->weights.at<double>(i));
	}
	total = sqrt(total);
	double temp = 1.0 / sqrt(param.lambda);
	temp = temp / total; 
	double myMult = std::min(1.0, temp);

	this->weights = this->weights * myMult;
	
	LOG_TRACE(m_logger, "NormalizeMag Done");
	return(myMult);
	*/
}

void WSLinearTrain::TrainFinalize(WSTrainParam &param)
{
}

void WSLinearTrain::ApplyInit(WSTrainParam &param)
{
}

void WSLinearTrain::Apply(WSGraph &aGraph, WSTrainParam &param)
{
	LOG_TRACE(m_logger, "Apply");
	//cout << " I have " << this->weights.rows << " weights: " << this->weights.at<double>(0) << "\n";
	if (aGraph.GetNumFeatures() != this->weights.rows) BOOST_THROW_EXCEPTION(UnexpectedSize());
	MergeEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(aGraph.GetGraph());

	for (eit = estart; eit != eend; eit++) {
		double total = 0.0;
		for (int i = 0; i < aGraph.GetNumFeatures(); i++) {
			total += this->weights.at<double>(i) * (double)aGraph.GetGraph()[*eit].features[i];
		}

		aGraph.GetGraph()[*eit].valf = boost::numeric_cast<float>(total);
	}

	aGraph.Update(param.treep);	
	
	LOG_TRACE(m_logger, "Apply Done");
}


void WSLinearTrain::MakeCopy(boost::shared_ptr<WSTrain> &dest)
{
	WSLinearTrain *temp = (WSLinearTrain *) dest.get(); 
	temp->weights = this->weights.clone(); 
}

void WSLinearTrain::Print(WSTrainParam &param)
{
	LOG_INFO(m_logger, "segType: " << param.treep.segType << "\n");
	LOG_INFO(m_logger, "T: " << param.treep.threshold << "\n");
	LOG_INFO(m_logger, "W: " << this->weights << "\n");		
}

void WSLinearTrain::Save(WSTrainParam &param)
{
	string sname = param.expName + string(".weights");
	FileStorage fs(sname, FileStorage::WRITE);
	fs << "weights" << this->weights;
	fs << "thresh" << param.treep.threshold;
	fs << "segType" << param.treep.segType;
	fs.release();
}

void WSLinearTrain::Load(WSTrainParam &param)
{
	string sname = param.expName + string(".weights");
	FileStorage fs(sname, FileStorage::READ);
	fs["weights"] >> this->weights;
	fs["thresh"] >> param.treep.threshold;
	fs["segType"] >> param.treep.segType;
	fs.release();
}

WSLinearTrain::WSLinearTrain() : m_logger(LOG_GET_LOGGER("WSLinearTrain"))
{
}

WSLinearTrain::~WSLinearTrain()
{
}

