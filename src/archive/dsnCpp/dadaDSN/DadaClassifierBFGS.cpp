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

#include "DadaClassifierBFGS.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaEval.h"

#include "OWLQN.h"

double SegObjective::Eval(const DblVec& input, DblVec& gradient)
{	
	return(problem->EstimateGradient(input, gradient));	
}

double DadaClassifierBFGS::EstimateGradient(const std::vector<double> &current, std::vector<double> &gradient)
{
	LOG_TRACE_METHOD(m_logger, "EstimateGradient");
	BOOST_THROW_EXCEPTION(NotImplemented());
	/*
	int D = current.size();
	int FD = m_classParam->W().rows;

	if (m_param->classifierGradientThreshold) {
		if (D != FD + 1) BOOST_THROW_EXCEPTION(UnexpectedSize(D, FD + 1)); 
	}
	else {
		if (D != FD) BOOST_THROW_EXCEPTION(UnexpectedSize(D, FD));
	}


	for (int i = 0; i < FD; i++) {
		m_classParam->W().at<double>(i) = current[i];
	}
	if (m_param->classifierGradientThreshold) m_classParam->T() = current[D-1];

	//cout << "Current: " << m_classParam->W().t() << "\n";

	double loss = this->EstimateGradient(*m_myGraph, *m_features, *m_gt); 

	//cout << "Gradient: " << this->m_gradient.t() << "\n";
	gradient.resize(D); 
	for (int i = 0; i < FD; i++) {
		gradient[i] = this->m_gradient.at<double>(i); 
	}
	if (m_param->classifierGradientThreshold) {
		gradient[D - 1] = this->m_gradientThresh;
		//LOG_INFO(m_logger, "\nGradient on thereshold is " << this->m_gradientThresh);
	}

	return(loss); 
	*/
	return(0.0); 
}

double DadaClassifierBFGS::EstimateGradient(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &gt)
{
	LOG_TRACE_METHOD(m_logger, "EstimateGradient Inner");
	BOOST_THROW_EXCEPTION(NotImplemented());
	/*
	m_param->currentIteration = m_param->currentIteration + 1;
	DadaClassifier::Apply(myGraph, features);
	m_eval.ComputeMaxMin(myGraph, gt.Labels());

	//double err = m_eval.TrainThreshold(m_classParam->T(), extraPos, extraNeg);
	//LOG_INFO(m_logger, "I: " << m_param->currentIteration << " T: " << m_classParam->T() << " had E: " << err);

	double loss;

	if (m_param->classifierGradientThreshold) {
		loss = m_eval.GetCCLogisticLoss(myGraph, features, m_classParam->T(), m_gradient, gt, &m_gradientThresh);
	}
	else {
		loss = m_eval.GetCCLogisticLoss(myGraph, features, m_classParam->T(), m_gradient, gt);
	}
	return(loss);
	*/
}

void DadaClassifierBFGS::TrainInit()
{
	LOG_TRACE_METHOD(m_logger, "TrainItit");

	m_param->currentIteration = 0;
	
	m_bestError = LARGEST_DOUBLE;
	m_bestIteration = -1;
	m_bestThreshold = 0.0;		
	m_sawImprovment = 0;
	m_trainDone = 0;
	m_classParam->T() = 0.0; 

	m_obj = (SegObjective *) new SegObjective; 
	m_obj->problem = this; 

}

void DadaClassifierBFGS::TrainIteration(MamaGraph &myGraph, DadaFeatureGenerator &fg, DadaWSGT &gt)
{
	LOG_TRACE_METHOD(m_logger, "TrainIteration");
	BOOST_THROW_EXCEPTION(NotImplemented());
	/*
	int FD = features.begin()->second.cols; 	
	int D = FD; 
	if (m_param->classifierGradientThreshold) D = FD + 1;

	m_myGraph = &myGraph; 
	m_features = &features; 
	m_gt = &gt; 
	
	DblVec init(D), ans(D);

	for (int i = 0; i < FD; i++) {
		init[i] = m_classParam->W().at<double>(i);
	}
	if (m_param->classifierGradientThreshold) init[D - 1] = m_classParam->T(); 

	
	OWLQN opt;
	opt.Minimize(*m_obj, init, ans, m_param->alpha, 1e-6, 10); 
	//opt.Minimize(*m_obj, init, ans, 0.0);
	
	for (int i = 0; i < FD; i++) {
		m_classParam->W().at<double>(i) = -ans[i];
		cout << "   " << m_classParam->W().at<double>(i) << "\n";
	}
	if (m_param->classifierGradientThreshold) {
		m_classParam->T() = -ans[D - 1];
		cout << "   " << m_classParam->T() << "\n";
	}

	m_sawImprovment = 1;
	m_trainDone = 1;
	*/
}


void DadaClassifierBFGS::TrainFinalize()
{
	BOOST_THROW_EXCEPTION(NotImplemented());
	/*
	DadaClassifier::Apply(*m_myGraph, *m_features);
	m_eval.ComputeMaxMin(*m_myGraph, m_gt->Labels()); 
	DadaError error; 

	if (m_param->classifierGradientThreshold) {
		m_eval.RandMSTError(*m_gt, error);
		LOG_INFO(m_logger, "FINAL T: " << m_classParam->T() << " had E: " << error.GetError());
	}
	else {
		m_eval.TrainThreshold(m_classParam->T(), *m_gt, error);
		LOG_INFO(m_logger, "Train T: " << m_classParam->T() << " had E: " << error.GetError());
	}

	delete m_obj; 
	*/
}

double DadaClassifierBFGS::ApplyNoThreshold(MamaEId &mid, DadaFeatureGenerator &fg)
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
	return(0.0); 
}

double DadaClassifierBFGS::ApplyThreshold(double &feature)
{
	return(feature - m_classParam->T());
}

DadaClassifierBFGS::DadaClassifierBFGS(std::shared_ptr<DadaParam> param, std::shared_ptr<DadaClassifierParam> classParam)
	: DadaClassifier(param, classParam)
{	
}

DadaClassifierBFGS::~DadaClassifierBFGS()
{	
}

