#include "ConnectedComponents.h"
#include "DadaDef.h"

#include "opencv2/opencv.hpp"
using namespace cv;

#include <boost/tuple/tuple.hpp>


void ConnectedComponents::GetComponentLocations(cv::Mat &img, float thresh, std::vector< cv::Point2f > &myCentroids)
{
	Mat imgt;
	threshold(img, imgt, thresh, 1.0, 0);
	int numcc = ConnectedComponents::Label(imgt);
	int numcen = ConnectedComponents::Centroids(imgt, myCentroids);	
}

void ConnectedComponents::GetComponentLocations(cv::Mat &img, cv::Mat &mask, std::vector< cv::Point3f > &myCentroids)
{
	Mat imgt;
	threshold(mask, imgt, 0.5, 1.0, 0);
	int numcc = ConnectedComponents::Label(imgt);
	int numcen = ConnectedComponents::Centroids(imgt, img, myCentroids);
}

int ConnectedComponents::Label(Mat &label_image)
{
	assert(label_image.type() == CV_32F); 
	
	double minVal, maxVal; 	
	minMaxLoc( label_image, &minVal, &maxVal); //, &minLoc, &maxLoc );
	if (maxVal > 1.1f) {
		LOG_ERROR(m_logger, "Expects binary image with values 0,1"); 
		return(-1);
	}

	int label_count = 2; // starts at 2 because 0,1 are used already
 	
    for(int y=0; y < label_image.rows; y++) {
        for(int x=0; x < label_image.cols; x++) {
            if(label_image.at<float>(y,x) == 1.0f) {
				Rect rect;
				floodFill(label_image, cv::Point(x,y), cv::Scalar((double) label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);
				label_count ++;
			}
        }
    }
	label_count = label_count - 2;
	return(label_count);
}

int ConnectedComponents::Centroids(cv::Mat &label_image, vector< Point2f > &myCentroids)
{
	assert(label_image.type() == CV_32F); 

	typedef boost::tuple<int, int, int> Centroid;
	
	map<int, Centroid> blobs;
    
	for(int y=0; y < label_image.rows; y++) {
        for(int x=0; x < label_image.cols; x++) {
			int labeli = (int) label_image.at<float>(y,x); 
			if (labeli > 0) {
				if (blobs.count(labeli) == 0) {
					blobs[labeli] = boost::make_tuple(x, y, 1);				
				} else {
					blobs[labeli].get<0>() += x;
					blobs[labeli].get<1>() += y;
					blobs[labeli].get<2>()++;
				}
			}
		}
	}

	myCentroids.clear();
	for (map<int, Centroid>::iterator it = blobs.begin(); it != blobs.end(); it++) {
		Point2f p; 
		p.x = (float) it->second.get<0>() / (float) it->second.get<2>();
		p.y = (float) it->second.get<1>() / (float) it->second.get<2>();
		myCentroids.push_back(p);
	}
	return(myCentroids.size()); 
}

int ConnectedComponents::Centroids(cv::Mat &label_image, cv::Mat &conf_image, vector< Point3f > &myCentroids)
{
	assert(label_image.type() == CV_32F); 

	typedef boost::tuple<float, float, float, float> Centroid;

	map<int, Centroid> blobs;

	for (int y = 0; y < label_image.rows; y++) {
		for (int x = 0; x < label_image.cols; x++) {
			int labeli = (int)label_image.at<float>(y, x);
			float probf = conf_image.at<float>(y, x);
			if (labeli > 0) {
				if (blobs.count(labeli) == 0) {
					blobs[labeli] = boost::make_tuple( ((float) x) * probf, ((float) y ) * probf, probf, 1.0f);
				}
				else {
					blobs[labeli].get<0>() += ((float) x) * probf;
					blobs[labeli].get<1>() += ((float) y) * probf;
					blobs[labeli].get<2>() += probf;
					blobs[labeli].get<3>() += 1.0f;
				}
			}
		}
	}

	myCentroids.clear();
	for (map<int, Centroid>::iterator it = blobs.begin(); it != blobs.end(); it++) {
		Point3f p;
		p.x = it->second.get<0>() / it->second.get<2>();
		p.y = it->second.get<1>() / it->second.get<2>();
		p.z = it->second.get<2>() / it->second.get<3>(); 
		myCentroids.push_back(p);
	}
	return(myCentroids.size());
}

void ConnectedComponents::ComputeGradientBins(cv::Mat& img, cv::Mat& grad, cv::Mat& qangle, cv::Size paddingTL, cv::Size paddingBR, int nbins)
{
    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );

    Size gradsize(img.cols + paddingTL.width + paddingBR.width,
                  img.rows + paddingTL.height + paddingBR.height);
    grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation
    Size wholeSize;
    Point roiofs;
    img.locateROI(wholeSize, roiofs);

    int i, x, y;
    int cn = img.channels();

    Mat_<float> _lut(1, 256);
    const float* lut = &_lut(0,0);
	
	int gammaCorrection = 1;
    if( gammaCorrection )
        for( i = 0; i < 256; i++ )
            _lut(0,i) = std::sqrt((float)i);
    else
        for( i = 0; i < 256; i++ )
            _lut(0,i) = (float)i;

    AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradsize.width + 2;

    const int borderType = (int)BORDER_REFLECT_101;

    for( x = -1; x < gradsize.width + 1; x++ )
        xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
                        wholeSize.width, borderType) - roiofs.x;
    for( y = -1; y < gradsize.height + 1; y++ )
        ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
                        wholeSize.height, borderType) - roiofs.y;

    // x- & y- derivatives for the whole row
    int width = gradsize.width;
    AutoBuffer<float> _dbuf(width*4);
    float* dbuf = _dbuf;
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    int _nbins = nbins;
    float angleScale = (float)(_nbins/CV_PI);

    for( y = 0; y < gradsize.height; y++ )
    {
        const uchar* imgPtr  = img.data + img.step*ymap[y];
        const uchar* prevPtr = img.data + img.step*ymap[y-1];
        const uchar* nextPtr = img.data + img.step*ymap[y+1];

        float* gradPtr = (float*)grad.ptr(y);
        uchar* qanglePtr = (uchar*)qangle.ptr(y);

        if( cn == 1 )
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x];

				dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
                dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);

            }
        }
        else
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x]*3;
                float dx0, dy0, dx, dy, mag0, mag;

                const uchar* p2 = imgPtr + xmap[x+1]*3;
                const uchar* p0 = imgPtr + xmap[x-1]*3;

                dx0 = lut[p2[2]] - lut[p0[2]];
                dy0 = lut[nextPtr[x1+2]] - lut[prevPtr[x1+2]];
                mag0 = dx0*dx0 + dy0*dy0;

                dx = lut[p2[1]] - lut[p0[1]];
                dy = lut[nextPtr[x1+1]] - lut[prevPtr[x1+1]];
                mag = dx*dx + dy*dy;

                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dx = lut[p2[0]] - lut[p0[0]];
                dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
                mag = dx*dx + dy*dy;
 
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dbuf[x] = dx0;
                dbuf[x+width] = dy0;
            }
        }

        cartToPolar( Dx, Dy, Mag, Angle, false );

		for( x = 0; x < width; x++ )
        {
            float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
            int hidx = cvFloor(angle);
            angle -= hidx;
            gradPtr[x*2] = mag*(1.f - angle);
            gradPtr[x*2+1] = mag*angle;

			if( hidx < 0 )
                hidx += _nbins;
            else if( hidx >= _nbins )
                hidx -= _nbins;
            assert( (unsigned)hidx < (unsigned)_nbins );

            qanglePtr[x*2] = (uchar)hidx;
            hidx++;
            hidx &= hidx < _nbins ? -1 : 0;
            qanglePtr[x*2+1] = (uchar)hidx;
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ConnectedComponents::ConnectedComponents()
{
}

ConnectedComponents::~ConnectedComponents()
{	
}


/*


void SegGraph::PopulateGradientBins(Mat &grad, Mat &angle, int nbins)
{
	//grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    //qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation
	this->numBins = nbins;
	// Process nodes
	//info(1, "Nodes\n"); 
	for(uint n=0; n< this->vNodes.size(); n++) {
		vNodes[n].hist.clear(); 
		vNodes[n].hist.resize(nbins, 0.0f); 
		vNodes[n].bhist.clear(); 
		vNodes[n].bhist.resize(nbins, 0.0f); 

		Mat *chipp, *bounp;
		
		Rect roi = this->GetNodeROIfromIndex(n);
		chipp = this->GetErodedChipPointer(n);
		bounp = this->GetBoundaryChipPointer(n);
		
		if (this->expType < 2) {
			chipp = this->GetOrigChipPointer(n);
			bounp = this->GetOrigChipPointer(n);
		} else {
			chipp = this->GetErodedChipPointer(n);
			bounp = this->GetBoundaryChipPointer(n);
		}

		for(uint j=0; j< roi.height; j++) {
		for(uint i=0; i< roi.width; i++) {
			if (chipp->at<uchar>(j, i) > 0) {
				uint ind1 = (uint) angle.at<Vec2b>(roi.y+j, roi.x+i)[0];
				float mag1 = grad.at<Vec2f>(roi.y+j, roi.x+i)[0];
				uint ind2 = (uint) angle.at<Vec2b>(roi.y+j, roi.x+i)[1];
				float mag2 = grad.at<Vec2f>(roi.y+j, roi.x+i)[1];

				vNodes[n].hist[ind1] += mag1;
				vNodes[n].hist[ind2] += mag2;
			}
			
			if (bounp->at<uchar>(j, i) > 0) {
				uint ind1 = (uint) angle.at<Vec2b>(roi.y+j, roi.x+i)[0];
				float mag1 = grad.at<Vec2f>(roi.y+j, roi.x+i)[0];
				uint ind2 = (uint) angle.at<Vec2b>(roi.y+j, roi.x+i)[1];
				float mag2 = grad.at<Vec2f>(roi.y+j, roi.x+i)[1];

				vNodes[n].bhist[ind1] += mag1;
				vNodes[n].bhist[ind2] += mag2;			
			}
		}
		}
		//info(1, "HIST: "); 
		//PrintVector(vNodes[n].hist);
		//info(1, "\n"); 
	}
	// Process edges
	//info(1, "Edges\n"); 
	for(uint n=0; n< this->vEdges.size(); n++) {
		if (! (vEdges[n].longRangeEdge)) {
			vEdges[n].hist.clear(); 
			vEdges[n].hist.resize(nbins, 0.0f); 

			Rect roi = this->GetEdgeROIfromIndex(n);
			Mat &chip = this->GetEdgeChip(n);

			for(uint j=0; j< roi.height; j++) {
			for(uint i=0; i< roi.width; i++) {
				if (chip.at<uchar>(j, i) > 0) {
					uint ind1 = (uint) angle.at<Vec2b>(roi.y+j, roi.x+i)[0];
					float mag1 = grad.at<Vec2f>(roi.y+j, roi.x+i)[0];
					uint ind2 = (uint) angle.at<Vec2b>(roi.y+j, roi.x+i)[1];
					float mag2 = grad.at<Vec2f>(roi.y+j, roi.x+i)[1];

					vEdges[n].hist[ind1] += mag1;
					vEdges[n].hist[ind2] += mag2;

				}
			}
			}
		}
	}
	
	// Process the combined
	//info(1, "Combined\n"); 
	for(uint n=0; n< this->vEdges.size(); n++) {
		pair<int, int> id = this->getEdgeId(n);
		// Now we look at combined histograms
		SegEdge &e1 = this->getSegEdgeIndex(n);
		SegNode &s1 = this->getSegNodeId(id.first);
		SegNode &s2 = this->getSegNodeId(id.second);

		vEdges[n].combinedHist.clear();
		for(uint i=0; i< s1.hist.size(); i++) {
			vEdges[n].combinedHist.push_back( s1.hist[i] + s2.hist[i]); 
		}
		vEdges[n].combinedBHist.clear();
		if (vEdges[n].longRangeEdge) {
			for(uint i=0; i< s1.bhist.size(); i++) {
				vEdges[n].combinedBHist.push_back( s1.bhist[i] + s2.bhist[i]); 
			}
		} else {			
			for(uint i=0; i< s1.bhist.size(); i++) {
				vEdges[n].combinedBHist.push_back( s1.bhist[i] + s2.bhist[i] - 2.0f * vEdges[n].hist[i]); 
			}
		}
	}
}

*/
