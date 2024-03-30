#include "SegmentVectorize.h"
#include "DadaException.h"
#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
#include <vector>
using namespace std; 

#include <boost/lexical_cast.hpp>

#define PAD_SIZE 10

void SegmentVectorize::Convert(cv::Mat &rseg, float scale)
{
	LOG_TRACE(m_logger, "Convert"); 
    
    this->Clear();
    this->m_scale = scale; 
    cv::Mat seg;
    resize(rseg, seg, Size(), scale, scale, INTER_NEAREST);
	this->m_imgw = seg.cols; 
	this->m_imgh = seg.rows;
	// Get bounding boxes
	for (int j = 0; j < seg.rows; j++) {
		for (int i = 0; i < seg.cols; i++) {
			int segi = static_cast<int>(seg.at<float>(j, i));
			if (segi >= 0) {
				if (m_count.count(segi)) {
					if (i > m_maxx[segi])  m_maxx[segi] = i;
					if (i < m_minx[segi])  m_minx[segi] = i;
					if (j > m_maxy[segi])  m_maxy[segi] = j;
					if (j < m_miny[segi])  m_miny[segi] = j;
					m_count[segi]++;
				}
				else {
					m_minx[segi] = i;
					m_maxx[segi] = i;
					m_miny[segi] = j;
					m_maxy[segi] = j;
					m_count[segi] = 1;
				}
			}
		}
	}		

	for (auto &it : m_count) {
        //LOG_INFO(m_logger, "Has " << it.second);
		//LOG_INFO(m_logger, "Chip " << it.first << ": " << m_minx[it.first] << ", " << m_miny[it.first] << " -> " << m_maxx[it.first] << ", " << m_maxy[it.first]);
		int minx = std::max(m_minx[it.first] - PAD_SIZE, 0);
		int maxx = std::min(m_maxx[it.first] + PAD_SIZE, seg.cols-1); 
		int miny = std::max(m_miny[it.first] - PAD_SIZE, 0);
		int maxy = std::min(m_maxy[it.first] + PAD_SIZE, seg.rows-1);

		Range rowRange = Range(miny, maxy + 1); 
		Range colRange = Range(minx, maxx + 1); 
				
		m_msk[it.first] = Mat::zeros(maxy-miny+1, maxx-minx+1, CV_8U);
		//LOG_INFO(m_logger, "  mask size " << m_msk[it.first].cols << ", " << m_msk[it.first].rows);
		for (int j = rowRange.start; j < rowRange.end; j++) {
			for (int i = colRange.start; i < colRange.end; i++) {
				int segi = static_cast<int>(seg.at<float>(j, i));
				if (segi == it.first) {
					m_msk[it.first].at<unsigned char>(j - rowRange.start, i - colRange.start) = 1;					
				}
			}
		}
	}

	/////////////////////////
	// Iterate over the chips	
	for(auto &it : m_msk) {		

		int segId = it.first; 				
		double pixelCount = static_cast<double>(m_count[it.first]); 
		
        try {    		
            Mat tempMat = it.second.clone(); 

		    vector< vector<Point> > allCons;
		    allCons.clear();			
			cv::findContours(tempMat, allCons, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); 

		    if (allCons.size() < 1) {
			    BOOST_THROW_EXCEPTION(Unexpected("No contours returned"));
		    }
		    else {
			    this->m_contours[segId] = allCons[0];
		    }
	    }   
	    catch (...) {
            LOG_INFO(m_logger, "Problem with seg " << it.first); 
	    }        
	} 

	LOG_TRACE(m_logger, "Convert Done");	
}

void SegmentVectorize::Export(string fname)
{
	ofstream fout(fname); 

    fout << "{\"type\":\"FeatureCollection\",\"features\":[";
    int numSegs = this->m_contours.size(); 
    LOG_INFO(m_logger, "Vectorizing " << numSegs); 
    int count = 0; 
	int dud = 0; 
    for(auto &it : this->m_contours) {		
		int goodone = 1; 
		for (int i = 0; i < it.second.size(); i++) {
			if ((it.second[i].x > this->m_imgw) || (it.second[i].y > this->m_imgh) ||
				(it.second[i].x < 0) || (it.second[i].y < 0)) {
				goodone = 0;
				//cout << it.second[i].x << " and " << it.second[i].y << "\n";
			}
		}
		if (goodone) {

			fout << "{\"type\":\"Feature\",\"id\":";
			fout << boost::lexical_cast<std::string>(it.first);
			fout << ",\"geometry\":{";
			fout << "\"type\":\"Polgon\",";
			fout << "\"coordinates\":[";

			int minx = std::max(m_minx[it.first] - PAD_SIZE, 0);
			int miny = std::max(m_miny[it.first] - PAD_SIZE, 0);
			//LOG_INFO(m_logger, "minx " << minx << ", " << miny << "\n");

			for (int i = 0; i < it.second.size(); i++) {
				int cordx = static_cast<int>(static_cast<float>(minx + it.second[i].x) / this->m_scale);
				int cordy = static_cast<int>(static_cast<float>(miny + it.second[i].y) / this->m_scale);
				//LOG_INFO(m_logger, "Point " << cordx << ", " << cordy << "\n");
				fout << "[" << boost::lexical_cast<std::string>(cordx) << ",";
				fout << boost::lexical_cast<std::string>(cordy) << "]";
				if (i < it.second.size() - 1) fout << ",";
			}
			fout << "]}}";
			if (count < numSegs - 1) fout << ",";
			fout << "\n";
			count = count + 1;
		}
		else {
			//LOG_INFO(m_logger, "Got a dud"); 
			count = count + 1;
			dud++; 
		}
    }
    fout << "]}\n";
    fout.close(); 
	LOG_INFO(m_logger, "Vectorization had problems with " << dud); 
}

/*
{
  "type": "FeatureCollection",
  "features": [
    {
type: feature
geometry 
{
    "coordinates": [
        [
            [
                68.52823492039879,
                11.969604492187502
            ],
            [
                74.4964131169431,
                46.77429199218751
            ],
            [
                67.87554134672945,
                47.82897949218751
            ]
        ]
    ],
    "type": "Polygon"
}
*/

void SegmentVectorize::Clear()
{
	m_minx.clear();
	m_maxx.clear();
	m_miny.clear();
	m_maxy.clear();
	m_msk.clear();
	m_count.clear();
    m_scale = 1.0; 
}

SegmentVectorize::SegmentVectorize() : m_logger(LOG_GET_LOGGER("SegmentVectorize"))
{	
}

SegmentVectorize::~SegmentVectorize()
{	
}
