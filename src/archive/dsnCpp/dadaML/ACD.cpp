#include "ACD.h"


using namespace std;

#include "MamaException.h"

#include "opencv2/opencv.hpp"
using namespace cv;

void ACD::Train(cv::Mat &x, cv::Mat &y)
{	
	int dx = x.cols;
	int dy = y.cols;
	int DD = x.cols + y.cols;
	int D = x.cols;
	int N = x.rows;
	int nx = x.rows;
	int ny = y.rows;
	if (dx != dy) BOOST_THROW_EXCEPTION(UnexpectedSize()); 
	if (nx != ny) BOOST_THROW_EXCEPTION(UnexpectedSize());
	
	
	Mat cx, cy, mx, my;
	calcCovarMatrix(x, cx, mx, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
	calcCovarMatrix(y, cy, my, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
	//cout << "X\n" << cx << "\n";
	//cout << "Y\n" << cy << "\n";
	//cout << "I have x " << x.rows << " by " << x.cols << "\n";
	//cout << "I have mean " << mx.rows << " by " << mx.cols << "\n";
	Mat cc = Mat::zeros(DD, DD, CV_64F);
	Mat ccx = cc(Range(0, D), Range(0, D));
	Mat ccy = cc(Range(D, DD), Range(D, DD));
	Mat cxy, icxy, icc;		
	cx.copyTo(ccx); 
	cy.copyTo(ccy); 
	//cout << "XY\n" << cc << "\n";

	invert(cc, icc, DECOMP_SVD);
	//cout << "IXY\n" << icc << "\n";
	
	Mat newxy;
	
	newxy = Mat::zeros(N, DD, CV_64F);
	//cout << "WA\n";
	//cout << "I have xy " << newxy.rows << " by " << newxy.cols << "\n";
	Mat xside = newxy(Range(0, N), Range(0, D));
	Mat yside = newxy(Range(0, N), Range(D, DD));
	x.copyTo(xside); 
	y.copyTo(yside); 
	calcCovarMatrix(newxy, cxy, this->mxy, CV_COVAR_NORMAL + CV_COVAR_ROWS, CV_64F);
	invert(cxy, icxy, DECOMP_SVD);

	this->Q = icxy - icc;
	cout << "Q: " << Q << "\n";
}

void ACD::Apply(cv::Mat &x, cv::Mat &y, cv::Mat &result)
{
	if ((x.cols+y.cols) != this->mxy.cols) BOOST_THROW_EXCEPTION(UnexpectedSize());
	if (x.rows != y.rows) BOOST_THROW_EXCEPTION(UnexpectedSize());
	Mat txy = Mat::zeros(1, x.cols + y.cols, CV_64F);
	Mat xside = txy(Range(0,1), Range(0, x.cols));
	Mat yside = txy(Range(0,1), Range(x.cols, y.cols+x.cols));
	
	result = Mat::zeros(x.rows, 1, CV_64F);

	for (int i = 0; i < x.rows; i++) {
		x.row(i).copyTo(xside); 
		y.row(i).copyTo(yside); 
		Mat zxy = (txy - this->mxy);
		Mat zxyt = zxy.t();
		Mat nv = zxy*this->Q;
	
		Mat fv = nv*zxyt;
		
		result.at<double>(i) = fv.at<double>(0, 0);
	}
	//this->mx = Mat::zeros(1, dx, CV_64F);
	//this->my = Mat::zeros(1, dx, CV_64F);

}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
ACD::ACD() : m_logger(LOG_GET_LOGGER("ACD"))
{		
}

ACD::~ACD()
{	
}
