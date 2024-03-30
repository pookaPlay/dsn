#include "ThreshSegment_Int.h"

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "DadaException.h"
#include "VizMat.h"
#include "DadaDef.h"


#define		THRESHOLD_RESOLUTION	1000

int localThreshold;

extern ThreshSegment_Int myISeg;

static void myTrackbar(int value, void*)
{
	float myThreshVal = (float) localThreshold / (float) THRESHOLD_RESOLUTION;
	myISeg.Update(myThreshVal);
	myISeg.Render();
}

static void onMouse(int eid, int x, int y, int flags, void*)
{	
	if (eid == EVENT_LBUTTONDOWN) {		// EVENT_MOUSEMOVE
		cout << "Click at " << x << ", " << y << "\n";
	}	
}

void ThreshSegment_Int::Run()
{	
	namedWindow( "Threshold", 1 );
	createTrackbar( "Slider", "Threshold", &localThreshold, THRESHOLD_RESOLUTION, myTrackbar);	
	int kb = 1;
	while ( kb != 'q' ) {
		//if (kb == 's') {
//			FitsMat::Write2DFitsAsFloat("temp.fits", this->seg);
	//	}
		/*
		if (kb == 'o') {
			param.preType = (param.preType + 1) % 5;
			LOG_INFO(m_logger, "Scale type " << param.scaleType << "\n");
			LOG_INFO(m_logger, "Pre type " << param.preType << "\n");
			LOG_INFO(m_logger, "Post type " << param.postType << "\n");
			this->Init(this->img);
			this->Update(param.threshold);
			this->Render();			
		}
		else if (kb == 's') {
			param.scaleType = (param.scaleType + 1) % 5;
			LOG_INFO(m_logger, "Scale type " << param.scaleType << "\n");
			LOG_INFO(m_logger, "Pre type " << param.preType << "\n");
			LOG_INFO(m_logger, "Post type " << param.postType << "\n");			
			this->Init(this->img);
			this->Update(param.threshold);
			this->Render();			
		}
		else if (kb == 'p') {
			param.postType = (param.postType + 1) % 5;
			LOG_INFO(m_logger, "Scale type " << param.scaleType << "\n");
			LOG_INFO(m_logger, "Pre type " << param.preType << "\n");
			LOG_INFO(m_logger, "Post type " << param.postType << "\n");			
			this->Init(this->img);
			this->Update(param.threshold);
			this->Render();			
		}
		*/
		kb = waitKey(0);
	}
}

void ThreshSegment_Int::Render()
{
	int magFactor = 1;
	namedWindow( "Main", 1 );		
	setMouseCallback("Main", onMouse, 0);
	picy = Mat::zeros(img.rows, img.cols, CV_8UC3);	
	this->RenderSeg(img, seg, picy);
	Mat bigPicy;
	Size mySize(picy.rows*magFactor, picy.cols*magFactor);
	resize(picy, bigPicy, mySize);
	imshow( "Main", bigPicy); 
}

void ThreshSegment_Int::Init(cv::Mat &imgf, int runSeg)	
{	
	LOG_INFO(m_logger, "Init with " << imgf.rows << " and " << imgf.cols << "\n");

	this->img = imgf.clone();
	this->seg = Mat::zeros(this->img.rows, this->img.cols, CV_32F);

	localThreshold = THRESHOLD_RESOLUTION / 2;
	param.threshold = 0.5;
	param.waterfall = 0;
	if (runSeg) segment.Init(this->img, param);
	
}

void ThreshSegment_Int::Update(float threshold)
{
	param.threshold = threshold;	
	//if (this->baseMode) {
		//segment.UpdateBaseThreshold(param);
		//segment.GetBasinLabels(this->seg);
	//}
	//else {
		segment.UpdateThreshold(param);
		segment.GetLabels(this->seg);
	//}
	
}

void ThreshSegment_Int::RenderSeg(Mat &img, Mat &seg, Mat &picy)
{
	int w = picy.cols; 
	int h = picy.rows;	

	float imageContrast = 0.75f;
	float segColorContrast = 0.5f;
		
	for(int j=0; j< picy.rows; j++) {
	for(int i=0; i< picy.cols; i++) {
		picy.at<Vec3b>(j, i)[0] = static_cast<uchar>(img.at<float>(j,i) * imageContrast);
		picy.at<Vec3b>(j, i)[1] = static_cast<uchar>(img.at<float>(j,i) * imageContrast);
		picy.at<Vec3b>(j, i)[2] = static_cast<uchar>(img.at<float>(j,i) * imageContrast);
	}
	}
		/*
	//if (myIO.showSeg) {
		for(int j=0; j< seg.rows; j++) {
		for(int i=0; i< seg.cols; i++) {	
			int segid = (int) seg.at<float>(j, i);			
			int cii = segid % 64;
			int col1 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 0] * 255.0);
			int col2 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 1] * 255.0);
			int col3 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 2] * 255.0);
			picy.at<Vec3b>(j, i)[0] = col1 * segColorContrast; 
			picy.at<Vec3b>(j, i)[1] = col2 * segColorContrast; 
			picy.at<Vec3b>(j, i)[2] = col3 * segColorContrast; 
			//if (myIO.showSel) {
			//	if (FindIn(myIO.GetSelections(), segid)) picy.at<Vec3b>(j, i)[2] = 255; 
			//}
		}
		}
	//}
	//*/
		
	// segmentation edges	
	if ((seg.rows == h) && (seg.cols == w)) {		
		for (int j = 0; j < seg.rows; j++) {
			for (int i = 0; i < seg.cols; i++) {
				int vali = (int)seg.at<float>(j, i);
				int boundary = 0;
				if (j > 0) {
					int vali2 = (int)seg.at<float>(j - 1, i);
					if (vali != vali2) boundary = 1;
				}
				if (j < h - 1) {
					int vali2 = (int)seg.at<float>(j + 1, i);
					if (vali != vali2) boundary = 1;
				}
				if (i > 0) {
					int vali2 = (int)seg.at<float>(j, i - 1);
					if (vali != vali2) boundary = 1;
				}
				if (i < w - 1) {
					int vali2 = (int)seg.at<float>(j, i + 1);
					if (vali != vali2) boundary = 1;
				}
				if (boundary) {
					picy.at<Vec3b>(j, i)[1] = 255;
				}
			}
		}
	}
	
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
ThreshSegment_Int::ThreshSegment_Int() : m_logger(LOG_GET_LOGGER("ThreshSegment_Int"))
{		
	baseMode = 0;
}

ThreshSegment_Int::~ThreshSegment_Int()
{	
}

