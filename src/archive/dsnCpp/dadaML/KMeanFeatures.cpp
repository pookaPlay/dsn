#include "KMeanFeatures.h"
#include "MamaDef.h"
#include "Info.h"

static Info info;

#include <limits>
//#include "HistFuncs.h"
#include "MamaException.h"

using namespace cv;

#define KMEANS_ITER	1000
#define KMEANS_EPS  1.0

void KMeanFeatures::SaveModel(string fname)
{
	white.SaveModel(fname);		
	string sname = fname + ".kmeans.yml";
	FileStorage fs(sname, FileStorage::WRITE);
	fs << "whiten" << this->myUseWhite;
	fs << "dict" << this->dict; 
	fs.release();
}
	
void KMeanFeatures::LoadModel(string fname)
{
	white.LoadModel(fname);
	this->dict = Mat();
	string sname = fname + ".kmeans.yml";
	FileStorage fs(sname, FileStorage::READ);
	fs["whiten"] >> this->myUseWhite;
	fs["dict"] >> this->dict; 
	fs.release();    
}

void KMeanFeatures::LearnFeatures(Mat &myNodeData, int numFeatures)
{	
	Mat nodeLabels;
	kmeans(myNodeData, numFeatures, nodeLabels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, KMEANS_ITER, KMEANS_EPS), 3, KMEANS_PP_CENTERS, this->dict);	
	info(1, "KMeans calculated %i, %i features from %i, %i data\n", this->dict.rows, this->dict.cols, myNodeData.rows, myNodeData.cols);
}


void KMeanFeatures::GenerateFeatures(Mat &myNodeData, Mat *myOut)
{	
	/*
	info(1, "KMeanFeatures::GenerateFeatures\n"); 
	vector<float> tempf;
	int patchSize = (winSize*2 + 1)*(winSize*2 + 1);
	tempf.clear(); tempf.resize(patchSize, 0.0f);
	
	features.clear();
	int numFeatures = this->dict.rows;
	for(int f=0; f< numFeatures; f++) {
		Mat nf = Mat::zeros( img.rows, img.cols, CV_32F); 
		features.push_back(nf);
	}
	vector<float> myDistances;
	myDistances.clear();
	myDistances.resize(numFeatures, 0.0f);

	for(int j=0; j< img.rows; j++) {
	for(int i=0; i< img.cols; i++) {
		
		int upto = 0;		
		for(int jj=-winSize; jj< winSize; jj++) {
		for(int ii=-winSize; ii< winSize; ii++) {
			int newj = j + jj;
			int newi = i + ii;
			if ((newi >= 0)	&& (newi < img.cols) && (newj >= 0) && (newj < img.rows)) {
				tempf[upto] = img.at<float>(newj, newi); 
			} else tempf[upto] = 0.0f;
			upto++;
		}
		}
		
		Mat arow = Mat(tempf);		
		float closest = std::numeric_limits<float>::max();
		int closestK = -1;
		float meanVal = 0.0f;		
		float myVal = 0.0f;

		for(int d=0; d< numFeatures; d++) {
			float dist = norm(arow.t() - this->dict.row(d));
			meanVal += dist;
			if (dist < closest) {
				closest = dist;
				closestK = d;
			}
			myDistances[d] = dist;
		}
		meanVal = meanVal / numFeatures;
		
		for(int d=0; d< numFeatures; d++) {
			if (myType == string("allMap")) {
				myVal = -myDistances[d] + meanVal;
			} else if (myType == string("oneMap")) {
				if (d == closestK) myVal = 255.0f;
				else myVal = 0.0f;
			} else if (myType == string("triMap")) {
				if (myDistances[d] < meanVal) {
					myVal = (meanVal - myDistances[d]) / meanVal;
				} else {
					myVal = 0.0f;
				}
			} else {
				info(1, "Unknown map\n");
				return;
			}
			features[d].at<float>(j,i) = myVal;
		}
	}
	}	
	*/
}

void KMeanFeatures::LearnFeatures(Mat &img, int numFeatures, int winSize, int useWhite)
{	
	Mat myNodeData = Mat();
	vector<float> tempf;
	tempf.clear();
	int patchSize = (winSize*2 + 1)*(winSize*2 + 1);
	tempf.resize(patchSize, 0.0f);
	this->myUseWhite = useWhite;

	//info(1, "Concating spatial neighborhood\n"); 
	for(int j=winSize; j< img.rows-winSize; j++) {
	for(int i=winSize; i< img.cols-winSize; i++) {
		int upto = 0;
		for(int jj=-winSize; jj<= winSize; jj++) {
		for(int ii=-winSize; ii<= winSize; ii++) {
			tempf[upto] = img.at<float>( j + jj, i + ii);
			upto++;
		}
		}
		Mat m = Mat(tempf);		
		Mat mt = m.t();
		myNodeData.push_back( mt ); 
	}
	}
	//info(1, "Here data is %i by %i\n", myNodeData.rows, myNodeData.cols); 
	//cout << myNodeData << "\n";
	if (myUseWhite) {
		
		white.Estimate(myNodeData); 
		//cout << white.myMean.t();
		white.Apply(myNodeData); 

	}
	


	Mat nodeLabels;
	kmeans(myNodeData, numFeatures, nodeLabels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, KMEANS_ITER, KMEANS_EPS), 3, KMEANS_PP_CENTERS, this->dict);	
	//info(10, "KMeans calculated %i, %i features from %i, %i data\n", this->dict.rows, this->dict.cols, myNodeData.rows, myNodeData.cols);
}

void KMeanFeatures::GenerateFeatures(Mat &img, vector< Mat > &features, string myType, int winSize)
{	
	//info(10, "KMeanFeatures::GenerateFeatures\n"); 
	try {
		vector<float> tempf;
		int patchSize = (winSize*2 + 1)*(winSize*2 + 1);
		tempf.clear(); tempf.resize(patchSize, 0.0f);
	
		features.clear();
		int numFeatures = this->dict.rows;
		for(int f=0; f< numFeatures; f++) {
			Mat nf = Mat::zeros( img.rows, img.cols, CV_32F); 
			features.push_back(nf);
		}

		vector<float> myDistances;
		myDistances.clear();
		myDistances.resize(numFeatures, 0.0f);
		//info(1, "Img here is %i by %i\n", img.rows, img.cols);
		for(int j=0; j< img.rows; j++) {
		for(int i=0; i< img.cols; i++) {
		
			int upto = 0;		
			for(int jj=-winSize; jj<= winSize; jj++) {
			for(int ii=-winSize; ii<= winSize; ii++) {
				int newj = j + jj;
				int newi = i + ii;
				if ((newi >= 0)	&& (newi < img.cols) && (newj >= 0) && (newj < img.rows)) {
					tempf[upto] = img.at<float>(newj, newi); 
				} else tempf[upto] = 0.0f;
				upto++;
			}
			}
		
			Mat arow = Mat(tempf);		
			Mat arowt = arow.t();
			//info(1, "Pre white: %i, %i\n", arow.rows, arow.cols); 
			if (myUseWhite) {								
				white.Apply(arowt); 				
			} 

			float closest = std::numeric_limits<float>::max();
			int closestK = -1;
			float meanVal = 0.0f;		
			float myVal = 0.0f;

			for(int d=0; d< numFeatures; d++) {
				float dist = norm(arowt - this->dict.row(d));
				meanVal += dist;
				if (dist < closest) {
					closest = dist;
					closestK = d;
				}
				myDistances[d] = dist;
			}
			meanVal = meanVal / numFeatures;
		
			for(int d=0; d< numFeatures; d++) {
				if (myType == string("allMap")) {
					myVal = -myDistances[d] + meanVal;
				} else if (myType == string("distance")) {
					myVal = myDistances[d]; 					
				} else if (myType == string("oneMap")) {
					if (d == closestK) myVal = 255.0f;
					else myVal = 0.0f;
				} else if (myType == string("triMap")) {
					if (myDistances[d] < meanVal) {
						myVal = (meanVal - myDistances[d]) / meanVal;
					} else {
						myVal = 0.0f;
					}
				} else {
					info(1, "Unknown map\n");
					return;
				}
				features[d].at<float>(j,i) = myVal;
			}
		}
		}	
	} catch(...) {
		BOOST_THROW_EXCEPTION( Unexpected() ); 
	}
}

void KMeanFeatures::NormalizeOutputFeatures(vector< cv::Mat > &features)
{
	Scalar meanMat, stdMat;
	double minVal, maxVal; 	

	//info(1, "Feature image %i, %i\n", features[0].cols, features[0].rows); 
	Mat tempf = Mat::zeros(features[0].rows, features[0].cols, CV_32F);
	
	for(int f=0; f< features.size(); f++) {
		meanStdDev(features[f], meanMat, stdMat);
		//info(1, "Mean size is %i, %i\n", meanMat.rows, meanMat.cols);
		//info(1, "Std  size is %i, %i\n", stdMat.rows, stdMat.cols);
		//info(1, "M: %f and S: %f\n", meanMat.val[0], stdMat.val[0]); 
		subtract(features[f], meanMat, tempf);
		if (stdMat.val[0] > 1.0e-6) {			
			divide(tempf, stdMat, features[f]);
		}		
		minMaxLoc( features[f], &minVal, &maxVal); //, &minLoc, &maxLoc );		
	}

}

void KMeanFeatures::VizFeatures(vector< cv::Mat > &features)
{
	Mat picy;
	for(int i=0; i< features.size(); i++) {
		double minVal, maxVal; 	
		minMaxLoc( features[i], &minVal, &maxVal); //, &minLoc, &maxLoc );
		cout << "min: " << minVal << " max: " << maxVal << "\n";
		float alpha = 255.0f / (maxVal - minVal);
		float beta = -(minVal * alpha);
		features[i].convertTo(picy, CV_8U, alpha, beta); 
		namedWindow( "Output Image", 1 );
		imshow( "Output Image", picy);
		waitKey(0);
	}
}


KMeanFeatures::KMeanFeatures(){	
}

KMeanFeatures::~KMeanFeatures(){
}

