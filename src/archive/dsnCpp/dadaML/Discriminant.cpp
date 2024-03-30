#include "Discriminant.h"
#include "Info.h"

using namespace std;
static Info info;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"

#define MAG_TOLERANCE	1.0e-10

void Discriminant::Print(int endLine)
{
	cout << "W: " << this->myWeights.t();
	cout << " T: " << this->myThresh;
	if (endLine) cout << "\n";
}

void Discriminant::Train(cv::Mat &mlData, cv::Mat &labels, cv::Mat &weights, float regularize)
{
	BOOST_THROW_EXCEPTION( NotImplemented() ); 
}

void Discriminant::InitUniform(int D)
{
	this->myWeights = Mat::ones(D, 1, CV_32F);
	this->myWeights = this->myWeights / (float) D;
	this->myThresh = 0.0f;
	this->setParam = 1;
}

void Discriminant::InitRandom(int D)
{
	this->myWeights = Mat::ones(D, 1, CV_32F);

	for(int i=0; i < D; i++) {
		float rf = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);					
		this->myWeights.at<float>(i) = ((2.0f * rf) - 1.0f);
	}
	
	this->myThresh = 0.0f;
	this->setParam = 1;
}

void Discriminant::InitZero(int D)
{
	this->myWeights = Mat::zeros(D, 1, CV_32F);

	this->myThresh = 1.0e-6; 
	this->setParam = 1;
}

void Discriminant::NormalizeMag()
{
	LOG_TRACE(m_logger, "NormalizeMag");
	float total = 0.0f;
	for(int i=0; i < this->myWeights.rows; i++) {
		total += (this->myWeights.at<float>(i) * this->myWeights.at<float>(i));
	}
	total = sqrt(total);
	if (fabs(total) > MAG_TOLERANCE) {
		this->myWeights = this->myWeights / total;
		this->myThresh = this->myThresh / total;
	}
	LOG_TRACE(m_logger, "NormalizeMag Done");
}

void Discriminant::Apply(cv::Mat &mlData, cv::Mat &result)
{	
  //info(10, "Discriminant::Apply\n");

  result = Mat::zeros(mlData.rows, 1, CV_32F); 
  
  for (int ni = 0; ni < mlData.rows; ni++) {
	  result.at<float>(ni) = mlData.row(ni).dot(this->myWeights.t());
  }

  //info(10, "Discriminant::Apply Done\n");
}

void Discriminant::TrainThreshold(cv::Mat &result, cv::Mat &labels, cv::Mat &weights)
{
	int N = result.rows;

	info(100, "LearnNodeGraph::FindThreshold\n");
	vector< pair<float, float> > ds;
	ds.clear(); ds.resize( N ); 

	// lets take a look	
	for(int ni=0; ni< N; ni++) {						
		ds[ni].first = result.at<float>(ni); 		
		ds[ni].second = weights.at<float>(ni) * labels.at<float>(ni); 
	}

	//sort(ds.begin(), ds.end(), std::greater< pair<float, int> >());
	sort(ds.begin(), ds.end());
	//info(1, "Threshold weights are %f -> %f\n", ds[0].first, ds[ds.size()-1].first);

	float myAccWeight = 0.0f;
	float myBestWeight = 0.0f;
	int myBestIndex = -1;

	for(int ni=0; ni< N; ni++) {				
		//info(1, "%f for %f\n", ds[ni].second, myAccWeight); 
		myAccWeight += ds[ni].second;
		if (myAccWeight < myBestWeight) {
			myBestWeight = myAccWeight;
			myBestIndex  = ni;
		}
	}
	
	this->myThresh = 0.0f;
	
	if (myBestIndex > -1) {
		if (myBestIndex < N-1 )  {
			this->myThresh = (ds[myBestIndex].first + ds[myBestIndex+1].first)/2.0f;
			//info(1, "Yip yah %f!\n", this->myThresh);
		}
		else  this->myThresh = ds[myBestIndex].first + 0.001f;
	} else this->myThresh = ds[0].first - 0.001f;
	
	info(10, "--------------->%f from [%f->%f]\n", this->myThresh, ds[0].first, ds[ ds.size() - 1].first);	
	info(100, "LearnNodeGraph::FindThreshold Done with %f\n", this->myThresh);
}

void Discriminant::ApplyThreshold(cv::Mat &data, cv::Mat &result, int hard)
{
	result = data - this->myThresh; 

	if (hard) {
		for(int i=0; i< result.rows; i++) {
			if ( result.at<float>(i) >= 0.0f) result.at<float>(i) = 1.0f; 
			else result.at<float>(i) = -1.0f; 
		}
	}
}

void Discriminant::Clear()
{	
	this->myWeights = Mat(); 
	this->myThresh = 0.0f;
	this->setParam = 1;
}



void Discriminant::SaveModel(string fname)
{
	string sname = fname + ".discriminant.model.yml";
	FileStorage fs(sname, FileStorage::WRITE);
	fs << "weights" << this->myWeights;
	fs << "thresh" << this->myThresh;
	fs.release();
}

void Discriminant::LoadModel(string fname)
{
	string sname = fname + ".discriminant.model.yml";
	FileStorage fs(sname, FileStorage::READ);
	fs["weights"] >> this->myWeights;
	fs["thresh"] >> this->myThresh;
	fs.release();    
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
Discriminant::Discriminant()  : m_logger(LOG_GET_LOGGER("Discriminant"))
{	
	Clear();
}

Discriminant::~Discriminant()
{	
}
