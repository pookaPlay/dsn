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

#include "DadaClassifierLinear.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaEval.h"
#include "DadaWSGT.h"

void DadaClassifierLinear::TrainInit()
{
	m_param->currentIteration = 0;
	
	m_bestError = LARGEST_DOUBLE;
	m_bestIteration = -1;
	m_bestThreshold = 0.0;		
	m_sawImprovment = 0;
	m_trainDone = 0;
}

void DadaClassifierLinear::TrainIteration(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &gt)
{
	LOG_TRACE_METHOD(m_logger, "TrainIteration");

	m_param->currentIteration = m_param->currentIteration + 1; 
	m_classParam->T() = 0.0; 
	
	DadaClassifier::Apply(myGraph, fg); 
	m_eval.ComputeMaxMin(myGraph, gt.Labels());

	DadaError error; 
	m_eval.TrainThreshold(m_classParam->T(), gt, error);
	//double err = m_eval.RandMSTError(gt);
	double err = error.GetError(); 
	double allPos = m_eval.GetTotalNeg() / (m_eval.GetTotalNeg() + m_eval.GetTotalPos());
	double allNeg = m_eval.GetTotalPos() / (m_eval.GetTotalNeg() + m_eval.GetTotalPos());
	double myTarget = min(m_eval.GetTotalNeg(), m_eval.GetTotalPos()) / (m_eval.GetTotalNeg() + m_eval.GetTotalPos());


	LOG_INFO(m_logger, "I: " << m_param->currentIteration << " T: " << m_classParam->T() << " had E: " << err);

	if (err < m_bestError) {
		m_bestError = err;
		m_bestThreshold = m_classParam->T();
		m_bestW = m_classParam->W().clone(); 
		m_bestIteration = m_param->currentIteration;
		m_sawImprovment = 1;
	}

	if (m_param->currentIteration >= m_param->maxTrainIterations) {
		m_trainDone = 1;
		return;
	}
	Mat gradient;
	BOOST_THROW_EXCEPTION(NotImplemented());
	/*
	if (m_param->segmentationType == string("ws")) {
		err = m_eval.GetWSGradient(myGraph, features, m_watershedNeighbors, m_classParam->T(), gradient);
	} 
	else {		
		if (m_param->classifierLossType == string("log")) {
			err = m_eval.GetCCLogisticLoss(myGraph, features, m_classParam->T(), gradient, gt);
		}
		else {
			err = m_eval.GetCCGradient(myGraph, features, m_classParam->T(), gradient);
		}
	}
		
	this->UpdateWeight(gradient, err);
	*/
}


void DadaClassifierLinear::UpdateWeight(cv::Mat &gradient, double loss)
{
	LOG_TRACE_METHOD(m_logger, "UpdateWeight");

	if (gradient.rows != m_classParam->W().rows) BOOST_THROW_EXCEPTION(UnexpectedSize(gradient.rows, m_classParam->W().rows));
	double lrate, irate; 
	if (m_param->classifierUpdateType == string("svm")) {
		lrate = 1.0 / (m_param->lrate * ((double)m_param->currentIteration + 1.0));
		irate = 1.0 - (1.0 / ((double)m_param->currentIteration + 1.0));
	}
	else if (m_param->classifierUpdateType == string("l1")) {
		lrate = loss / (loss + 1.0);
		irate = 1.0 / (loss + 1.0);
	}
	else {
		lrate = m_param->lrate;
		irate = 1.0; 
	}

	Mat orig = m_classParam->W().clone();
	Mat adj = gradient * lrate;	
	Mat adjw = orig * irate;
	m_classParam->W() = adjw + adj;

	if (m_param->classifierNormalizeWeight) {
		this->NormalizeWeight();
	}

	/*
	if (m_param->classifierUpdateType == string("l1")) {
		double l1sum = 0.0;
		for (int i = 0; i < m_classParam->W().rows; i++) {
			l1sum += abs(m_classParam->W().at<double>(i));
		}
		double l1clip = (l1sum / (m_classParam->W().rows));
		//LOG_INFO(m_logger, "L1 Clip: " << l1clip);
		for (int i = 0; i < m_classParam->W().rows; i++) {
			//if (abs(m_classParam->W().at<double>(i)) < l1clip) m_classParam->W().at<double>(i) = 0.0; 
			if (abs(m_classParam->W().at<double>(i)) < l1clip) m_classParam->W().at<double>(i) = m_classParam->W().at<double>(i) / 2.0; 
		}
	}
	*/
	if (m_param->printUpdate) {
		cout << "------------------------------------------\n";
		cout << "Rates: " << irate << ", " << lrate << "\n"; // << " and Norm: " << myMult << "\n");
		cout << " O        G         AW         AG       FF     DIFF\n";
		for (int i = 0; i < m_classParam->W().rows; i++) {
			cout << std::fixed << std::setprecision(5) <<
				orig.at<double>(i) << " " <<
				gradient.at<double>(i) << " " <<
				adjw.at<double>(i) << " " <<
				adj.at<double>(i) << "  |  " <<
				m_classParam->W().at<double>(i) << "  |  " <<
				(m_classParam->W().at<double>(i) -orig.at<double>(i)) << "\n";
		}
		cout << "------------------------------------------\n";
		if (m_param->stepUpdate) waitKey(0); 
	}	
}

void DadaClassifierLinear::NormalizeWeight()
{
	LOG_TRACE_METHOD(m_logger, "UpdateWeight");
	
	if (m_param->classifierUpdateType == string("svm")) {
		double total = 0.0;
		for (int i = 0; i < m_classParam->W().rows; i++) {
			total += (m_classParam->W().at<double>(i) * m_classParam->W().at<double>(i));
		}
		total = sqrt(total);
		double temp = 1.0 / sqrt(m_param->lrate);
		temp = temp / total;
		double myMult = std::min(1.0, temp);

		m_classParam->W() = m_classParam->W() * myMult;
	}
	else {
		double total = 0.0;
		for (int i = 0; i < m_classParam->W().rows; i++) {
			total += (m_classParam->W().at<double>(i) * m_classParam->W().at<double>(i));
		}
		total = sqrt(total);
		if (total > 0.0) {
			m_classParam->W() = m_classParam->W() / total;
		}
	}
}

void DadaClassifierLinear::TrainFinalize()
{
}

double DadaClassifierLinear::ApplyNoThreshold(MamaEId &mid, DadaFeatureGenerator &fg)
{
	BOOST_THROW_EXCEPTION(NotImplemented()); 
	/*
	if (feature.cols != m_classParam->W().rows) BOOST_THROW_EXCEPTION(UnexpectedSize()); 
	double total = 0.0;
	for (int i = 0; i < feature.cols; i++) {
		total += m_classParam->W().at<double>(i) * feature.at<double>(i); 
	}	
	return(total);
	*/
}

double DadaClassifierLinear::ApplyThreshold(double &feature)
{
	return(feature - m_classParam->T());
}

DadaClassifierLinear::DadaClassifierLinear(std::shared_ptr<DadaParam> param, std::shared_ptr<DadaClassifierParam> classParam)
	: DadaClassifier(param, classParam)
{	
}

DadaClassifierLinear::~DadaClassifierLinear()
{	
}

