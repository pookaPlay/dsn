#include "SvmDiscriminant.h"
#include "Info.h"

using namespace std;
static Info info;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"

#define MY_BIAS -1.0

void SvmDiscriminant::Train(Mat &mlData, Mat &labels, Mat &weights, float regularize)
{	
	if (this->myModel) {
		free_and_destroy_model(&(this->myModel));
	}
	
	this->Mat2SvmProblem(mlData, labels, weights, this->myProb);	
	this->DefaultSvmParameters(this->myParam);	
	if (regularize > 0.0) this->myParam.C = regularize;

	//info(1, "Regularize %f\n", this->myParam.C );
	this->myModel = train(&(this->myProb), &(this->myParam));	

	this->myWeights = Mat::zeros(mlData.cols, 1, CV_32F);	
	this->myThresh = 0.0f;
	for(int i=0; i< mlData.cols; i++) {
		this->myWeights.at<float>(i) = (float) *(this->myModel->w+i);
	}
	if (MY_BIAS >= 0.0) {
		this->myThresh = -1.0f * (float) *(this->myModel->w + mlData.cols);
	}
	this->setParam = 0;
	this->CleanUp();
	//info(1, "Yah in SVM TRAIN: "); 
	//for(int i=0; i< mlData.rows; i++) {
	//	cout << labels.at<float>(i) << ", " << weights.at<float>(i) << " [";
	//	cout << mlData.row(i) << "]\n";
	//}
	//cout << this->myWeights.t() << "\n";
}

void SvmDiscriminant::CleanUp()
{
	destroy_param(&(this->myParam));
	free(this->myProb.y);
	free(this->myProb.x);
	free(this->myProb.W);
	free(this->x_space);
	this->x_space = 0;
}

/*
void SvmDiscriminant::Apply(cv::Mat &mlData, cv::Mat &result)
{	
	info(10, "SvmDiscriminant::Apply\n");
	if (this->setParam) {
		Discriminant::Apply(mlData, result);
		return;
	}
	double bias = MY_BIAS;
	result = Mat::zeros(mlData.rows, 1, CV_32F); 
	int D = mlData.cols + 1;	// includes bias
	int N = mlData.rows;
	int mySpace = D;
	if (bias >= 0.0) mySpace = D+1;
	struct feature_node *x = (struct feature_node *) SvmMalloc(struct feature_node, mySpace);
	double answers[2];

	for (int ni = 0; ni < mlData.rows; ni++) {
		for (int di = 0; di < (D-1); di++) {
			x[di].index = di+1; 
			x[di].value = (double) mlData.at<float>(ni, di); 
		}
		if (bias < 0.0) {
			x[D-1].index = -1;
		} else {
			x[D-1].index = D;
			x[D-1].value = bias;
			x[D].index = -1;
		}
		double label=predict_values(this->myModel, x, answers); 
		result.at<float>(ni) = (float) (answers[0] * this->mySign ); 
		//info(1, "Here %f from %f\n", label, result.at<float>(ni));
	}

	free(x);
  info(10, "SvmDiscriminant::Apply Done\n");
}
*/

void SvmDiscriminant::Mat2SvmProblem(Mat &mlData, Mat &labels, Mat &weights, SvmProblem &prob)
{

	prob.bias = MY_BIAS; 
	int i;
	long int elements, j;
	
	int D = mlData.cols + 1;	// includes bias
	int N = mlData.rows;

	//info(1, "Allocating %i by %i\n", N, D);
	elements = D*N; 
	prob.l = N;

	char *endptr;
	char *idx, *val, *label;
	
	prob.y = SvmMalloc(double,prob.l);
	prob.x = SvmMalloc(struct feature_node *,prob.l);
	prob.W = SvmMalloc(double,prob.l);
	x_space = SvmMalloc(struct feature_node,elements+prob.l);
	
	j=0;
	int di;
	int upto = 0;

	this->mySign = labels.at<float>(0);		// weird libsvm thing 

	for(i=0;i<prob.l;i++)	{		
		prob.x[i] = &(x_space[upto]);
		prob.y[i] = (double) labels.at<float>(i); 
		
		for(di=0; di < (D-1); di++) {			// D includes bias 
			x_space[upto].index = di+1;
			x_space[upto].value = (double)mlData.at<float>(i, di); 
			upto++;
		}		
		if (prob.bias < 0.0) {
			x_space[upto].index = -1;
		} else {
			x_space[upto].index = D;
			x_space[upto].value = prob.bias;
			upto++;
			x_space[upto].index = -1; 
		}
		upto++;
	}

	if (prob.bias < 0.0) {
		prob.n=D-1;
	} else {
		prob.n=D;
	}
	
	double minVal, maxVal; 	
	minMaxLoc( weights, &minVal, &maxVal); //, &minLoc, &maxLoc );
	//cout << "    Weights min: " << minVal << " max: " << maxVal << "\n";
	int c1 = 0, c0 = 0;
	double w1 = 0.0, w0 = 0.0;
	for(i=0;i<prob.l;i++) {
		prob.W[i] = (double) (weights.at<float>(i)) / maxVal; 
		if (prob.y[i] > 0.0) {
			w1 += prob.W[i];
			c1++;
		} else {
			w0 += prob.W[i];
			c0++;
		}
	}
//	info(1, "SVM Prob: %i, %f +ve, %i, %f -ve\n", c1, w1, c0, w0);
}


void SvmDiscriminant::DefaultSvmParameters(SvmParameters &param)
{
	// default values
	//param.solver_type = L2R_L2LOSS_SVC_DUAL;  // enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */
	//param.solver_type = L1R_LR; 
	param.solver_type = L2R_LR; 
	//param.C = 1.0e6; 	
	param.C = 1;	
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.eps = 0.1;
	void (*print_func)(const char*) = NULL;	// default printing to stdout
	
	set_print_string_function(no_print_out);
	//set_print_string_function(print_func);
}

void no_print_out(const char *s) 
{
	//printf("%s", s);
}

void SvmDiscriminant::SaveModel(string fname)
{
	string sname = fname + ".svm";
	if(save_model(sname.c_str(), this->myModel)) {		
		BOOST_THROW_EXCEPTION(FileIOProblem("Writing " + sname));  
	}
	sname = fname + ".model.yml";
	FileStorage fs(sname, FileStorage::WRITE);
	fs << "weights" << this->myWeights;
	fs << "thresh" << this->myThresh;
	fs << "sign" << this->mySign;
	fs.release();
}

void SvmDiscriminant::LoadModel(string fname)
{
	string sname = fname + ".svm";
	if((this->myModel = load_model(sname.c_str()))==0)	{		
		BOOST_THROW_EXCEPTION(FileIOProblem("Reading " + sname));  		
	}	

	sname = fname + ".model.yml";
	FileStorage fs(sname, FileStorage::READ);
	fs["weights"] >> this->myWeights;
	fs["thresh"] >> this->myThresh;
	fs["sign"] >> this->mySign;
	fs.release();    
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
SvmDiscriminant::SvmDiscriminant() 
{	
	Clear();
	this->myModel = 0;
	this->x_space = 0;
	this->mySign = 1.0f;
}

SvmDiscriminant::~SvmDiscriminant()
{	
	if (this->myModel) {
		free_and_destroy_model(&(this->myModel));
	}
	if (this->x_space) {
		free(this->x_space);
	}
}
