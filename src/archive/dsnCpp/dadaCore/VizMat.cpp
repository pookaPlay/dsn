#include "VizMat.h"
#include "DadaDef.h"
#include <boost/numeric/conversion/cast.hpp>

using namespace cv;
using namespace std;
using boost::numeric_cast;

Logger VizMat::m_logger(LOG_GET_LOGGER("VizMat"));

std::vector<int> globalColMap = boost::assign::list_of(41)(9)(12)(29)(33)(35)(58)(10)(34)(53)(45)(13)(49)(61)(25)(16)(5)(21)(11)(55)(27)(48)(60)(1)(36)(44)(43)(6)(54)(14)(37)(26)(51)(39)(31)(23)(19)(38)(8)(20)(63)(47)(4)(52)(57)(28)(2)(3)(18)(62)(17)(0)(46)(40)(50)(30)(59)(7)(56)(24)(22)(15)(42)(32);

void VizMat::DisplaySeg(Mat &in, string id, int waitVal, float mag)
{
	Mat picy = Mat::zeros(in.rows, in.cols, CV_32F);
	map<int, int> vals;
	vals.clear();
	int upto = 1;
	for(int j=0; j < in.rows; j++) {
	for(int i=0; i < in.cols; i++) {
		int val = (int) in.at<float>(j, i);
		if (vals.count(val)) {
			picy.at<float>(j, i) = (float) vals[val];
		} else {
			vals[val] = upto;
			picy.at<float>(j, i) = (float) vals[val];
			upto++;
		}
	}
	}
	VizMat::DisplayFloat(picy, id, waitVal, mag);
}

void VizMat::DisplayEdgeSeg(Mat &img, Mat &seg, string id, int waitVal, float mag)
{
	Mat picy = Mat::zeros(img.rows, img.cols, CV_8UC3);

	int w = picy.cols; 
	int h = picy.rows;	

	float imageContrast = 0.5f;
	float segColorContrast = 0.5f;
		
	for(int j=0; j< picy.rows; j++) {
	for(int i=0; i< picy.cols; i++) {
		picy.at<Vec3b>(j, i)[0] = static_cast<unsigned char>(img.at<float>(j,i) * imageContrast);
		picy.at<Vec3b>(j, i)[1] = static_cast<unsigned char>(img.at<float>(j, i) * imageContrast);
		picy.at<Vec3b>(j, i)[2] = static_cast<unsigned char>(img.at<float>(j, i) * imageContrast);
	}
	}

	// segmentation edges	
	int sel = 0;
	for(int j=0; j< seg.rows; j++) {
	for(int i=0; i< seg.cols; i++) {	
		int vali = (int) seg.at<float>(j,i); 
		int boundary = 0; 
		if (j > 0) {					
			int vali2 = (int) seg.at<float>(j-1,i);
			if (vali != vali2) boundary = 1;
		}
		if (j < h-1) {	
			int vali2 = (int) seg.at<float>(j+1,i);
			if (vali != vali2) boundary = 1;
		}
		if (i > 0) {	
			int vali2 = (int) seg.at<float>(j,i-1);
			if (vali != vali2) boundary = 1;
		}		
		if (i < w-1) {	
			int vali2 = (int) seg.at<float>(j,i+1);
			if (vali != vali2) boundary = 1;
		}		
		if (boundary) {
			picy.at<Vec3b>(j, i)[1] = 255; 
		} 
	}
	}	
	namedWindow(id); 
	imshow( id, picy);
	if (waitVal >= 0) waitKey(waitVal);
}


void VizMat::DisplayFloat(Mat &in, string id, int waitVal, float mag)
{
	Mat picy;
	VizMat::FloatToByte(in, picy, mag);
	namedWindow(id); 
	imshow( id, picy);
	if (waitVal >= 0) waitKey(waitVal);
}

void VizMat::DisplayByte(Mat &in, string id, int waitVal, float mag)
{
	Mat picy;
	
	Size sz = Size(static_cast<int>(in.cols*mag), static_cast<int>(in.rows*mag));
	picy = Mat::zeros(sz, in.type());
	resize(in, picy, sz);	

	namedWindow(id); 
	imshow( id, picy);
	if (waitVal >= 0) waitKey(waitVal);
}

void VizMat::DisplayFloat(cv::Mat &in, vector< cv::Point2f > &pts, std::string id, int waitVal, float mag)
{
	Mat pic;
	VizMat::FloatToByte(in, pic, mag);

	Mat picy = Mat::zeros(pic.rows, pic.cols, CV_8UC3);
	vector<Mat> channels;
	channels.clear();
	channels.push_back(pic);channels.push_back(pic);channels.push_back(pic);
	merge(channels, picy);
	
	for(int i=0; i < pts.size(); i++) {
		int rr = ((i+1) % 2) ? 255 : 0; 
		int rg = (((i+1) % 4) > 1) ? 255 : 0;
		int rb = (((i+1) % 8) > 3) ? 255 : 0;
		rr = 255; 
		rg = 0; 
		rb = 0; 
		Point2f magp = pts[i];
		magp.x = magp.x * mag;
		magp.y = magp.y * mag;
		circle(picy, magp, 10, CV_RGB(rr, rg, rb), 1); 					
	}

	namedWindow(id); 
	imshow( id, picy);
	waitKey(waitVal);
}

void VizMat::DisplayFloat(cv::Mat &in, vector< cv::Point3f > &pts, std::string id, int waitVal, float mag)
{
	Mat pic;
	VizMat::FloatToByte(in, pic, mag);

	Mat picy = Mat::zeros(pic.rows, pic.cols, CV_8UC3);
	vector<Mat> channels;
	channels.clear();
	channels.push_back(pic); channels.push_back(pic); channels.push_back(pic);
	merge(channels, picy);

	for (int i = 0; i < pts.size(); i++) {
		int rr = ((i + 1) % 2) ? 255 : 0;
		int rg = (((i + 1) % 4) > 1) ? 255 : 0;
		int rb = (((i + 1) % 8) > 3) ? 255 : 0;
		rr = 255;
		rg = 0;
		rb = 0;
		Point2f magp; 
		magp.x = pts[i].x;
		magp.y = pts[i].y;
		magp.x = magp.x * mag;
		magp.y = magp.y * mag;
		circle(picy, magp, 10, CV_RGB(rr, rg, rb), 1);
	}

	namedWindow(id);
	imshow(id, picy);
	waitKey(waitVal);
}

void VizMat::FloatToByte(Mat &in, Mat &out, float mag)
{
	//Point minLoc; Point maxLoc;
	double minVal, maxVal; 	
	minMaxLoc( in, &minVal, &maxVal); //, &minLoc, &maxLoc );
	//cout << "min: " << minVal << " max: " << maxVal << "\n";

	Mat bytein;
	double alpha = 255.0f / (maxVal - minVal);
	double beta = -(minVal * alpha);
	in.convertTo(bytein, CV_8U, alpha, beta); 

	Size sz = Size(static_cast<int>(in.cols*mag), static_cast<int>(in.rows*mag));
	out = Mat::zeros(sz, CV_8U);
	resize(bytein, out, sz);	
	
	//minMaxLoc( out, &minVal, &maxVal); //, &minLoc, &maxLoc );
	//cout << "now -> min: " << minVal << " max: " << maxVal << "\n";

}
 
void VizMat::FloatTo8UC3(Mat &in, Mat &out, float mymaxVal)
{
	if (in.channels() == 1) {
		//Point minLoc; Point maxLoc;
		double minVal, maxVal; 	
		minMaxLoc( in, &minVal, &maxVal); //, &minLoc, &maxLoc );
		//cout << "min: " << minVal << " max: " << maxVal << "\n";

		Mat bytein;
		double alpha = mymaxVal / (maxVal - minVal);
		double beta = -(minVal * alpha);
		in.convertTo(bytein, CV_8U, alpha, beta); 

		Mat picy = Mat::zeros(in.rows, in.cols, CV_8UC3);
		vector<Mat> channels; channels.clear();
		channels.push_back(bytein);channels.push_back(bytein);channels.push_back(bytein);
		merge(channels, out);	
	} else {
		//Point minLoc; Point maxLoc;
		double minVal, maxVal; 	
		minMaxLoc( in, &minVal, &maxVal); //, &minLoc, &maxLoc );
		//cout << "min: " << minVal << " max: " << maxVal << "\n";

		Mat bytein;
		double alpha = mymaxVal / (maxVal - minVal);
		double beta = -(minVal * alpha);
		in.convertTo(out, CV_8UC3, alpha, beta); 

	}
}
 
void VizMat::Mosaic(vector< cv::Mat > &in, int nw, int nh, cv::Mat &out)
{	
	int nf = in.size();
	
	int ih= in[0].rows; 
	int iw = in[0].cols; 

	int h= ih * nh;
	int w = iw * nw;

	out = Mat::zeros(h, w, CV_8U);
	int upto = 0;
	for(int hi=0; hi < nh; hi++) {
		for(int wi=0; wi < nw; wi++) {
			if (upto >= in.size()) return;
			Mat outl  = out(Range(hi*ih,(hi+1)*ih),Range(wi*iw,(wi+1)*iw));	// This is just a header	
			in[upto].copyTo(outl);
			upto++;
		}
	}

}


int VizMat::DisplayColorSeg(Mat &in, string id, int waitVal, float mag)
{
	// First compress
	Mat comp = Mat::zeros(in.rows, in.cols, CV_32F);
	map<int, int> vals;
	vals.clear();
	int upto = 1;
	for(int j=0; j < in.rows; j++) {
	for(int i=0; i < in.cols; i++) {
		int val = (int) in.at<float>(j, i);
		if (vals.count(val)) {
			comp.at<float>(j, i) = (float) vals[val];
		} else {
			vals[val] = upto;
			comp.at<float>(j, i) = (float) vals[val];
			upto++;
		}
	}
	}
	// random color map
	//vector<int> colMap;
	//RandPerm(64, colMap);

	Mat picy = Mat::zeros(in.rows, in.cols, CV_8UC3);

	for(int j=0; j < in.rows; j++) {
	for(int i=0; i < in.cols; i++) {
		int val = static_cast<int>(comp.at<float>(j, i));
		int cii = globalColMap[ val % 64 ];
		int col1 = static_cast<int>(NEAREST_INT(jetcolormap[cii * 3 + 0] * 255.0));
		int col2 = static_cast<int>(NEAREST_INT(jetcolormap[cii * 3 + 1] * 255.0));
		int col3 = static_cast<int>(NEAREST_INT(jetcolormap[cii * 3 + 2] * 255.0));
		picy.at<Vec3b>(j, i)[0] = col1; 
		picy.at<Vec3b>(j, i)[1] = col2; 
		picy.at<Vec3b>(j, i)[2] = col3; 
	}
	}

	Size sz = Size(static_cast<int>(in.cols*mag), static_cast<int>(in.rows*mag));
	Mat out = Mat::zeros(sz, CV_8UC3);
	resize(picy, out, sz);	

	namedWindow(id); 
	imshow( id, out);
	if (waitVal >= 0) waitKey(waitVal);
	return(upto);
}


void VizMat::GenerateEdgeMaskFromSeg(Mat &seg, Mat &msk)
{
	msk = Mat::zeros(seg.rows, seg.cols, CV_32F);

	int w = seg.cols;
	int h = seg.rows;

	// segmentation edges	
	int sel = 0;
	for (int j = 0; j< seg.rows; j++) {
		for (int i = 0; i< seg.cols; i++) {
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
				msk.at<float>(j, i) = 255.0f;
			}
		}
	}
}

VizMat::VizMat(){	
}

VizMat::~VizMat(){
}

