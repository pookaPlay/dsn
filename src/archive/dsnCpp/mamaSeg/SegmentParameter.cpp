#include "SegmentParameter.h"
#include "MamaException.h"

#include "opencv2/opencv.hpp"
using namespace cv;


void SegmentParameter::Save(string fname)
{
	LOG_TRACE(m_logger, "Save");
	
	string sname = fname + ".segment.yml";
	FileStorage fs(sname, FileStorage::WRITE);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem()); 
	this->Save(fs); 
	fs.release();

	LOG_TRACE(m_logger, "Save Done");
}

void SegmentParameter::Save(FileStorage &fs) const
{
	fs << "version" << 1;
	fs << "scale" << this->scaleFactor;
	fs << "preType" << this->preType;
	fs << "preSize" << this->preSize;
	fs << "postType" << this->postType;
	fs << "postSize" << this->postSize;
	fs << "gradType" << this->gradType;
	fs << "gradSize" << this->gradSize;
	fs << "threshold" << this->threshold;
	fs << "absolute" << this->absoluteThreshold;
	fs << "waterfall" << this->waterfall;
}


void SegmentParameter::Load(string fname)
{
	LOG_TRACE(m_logger, "Load");
	
	string sname = fname + ".segment.yml";
	FileStorage fs(sname, FileStorage::READ);
	if (!fs.isOpened()) BOOST_THROW_EXCEPTION(FileIOProblem());
	this->Load(fs); 
	fs.release();
	
	LOG_TRACE(m_logger, "Load Done");
}

void SegmentParameter::Load(FileStorage &fs)
{
	fs["scale"] >> this->scaleFactor;
	fs["preType"] >> this->preType;
	fs["preSize"] >> this->preSize;
	fs["postType"] >> this->postType;
	fs["postSize"] >> this->postSize;
	fs["gradType"] >> this->gradType;
	fs["gradSize"] >> this->gradSize;
	fs["threshold"] >> this->threshold;
	fs["absolute"] >> this->absoluteThreshold;
	fs["waterfall"] >> this->waterfall;
}

void SegmentParameter::Load(FileNodeIterator &fs)
{
	(*fs)["scale"] >> this->scaleFactor;
	(*fs)["preType"] >> this->preType;
	(*fs)["preSize"] >> this->preSize;
	(*fs)["postType"] >> this->postType;
	(*fs)["postSize"] >> this->postSize;
	(*fs)["gradType"] >> this->gradType;
	(*fs)["gradSize"] >> this->gradSize;
	(*fs)["threshold"] >> this->threshold;
	(*fs)["absolute"] >> this->absoluteThreshold;
	(*fs)["waterfall"] >> this->waterfall;
}

void SegmentParameter::Print()
{
	LOG_INFO(m_logger, "Segment Parameters");
	LOG_INFO(m_logger, "Scale: " << this->scaleFactor);
	LOG_INFO(m_logger, "Pre Type: " << this->preType);
	LOG_INFO(m_logger, "Pre Scale: " << this->preSize);
	LOG_INFO(m_logger, "Post Type: " << this->postType);
	LOG_INFO(m_logger, "Post Scale: " << this->postSize);
	LOG_INFO(m_logger, "Grad Type: " << this->gradType);
	LOG_INFO(m_logger, "Grad Scale: " << this->gradSize);
	LOG_INFO(m_logger, "Threshold: " << this->threshold);
	LOG_INFO(m_logger, "AbsThresh: " << this->absoluteThreshold);
	LOG_INFO(m_logger, "Waterfall: " << this->waterfall);
}

int SegmentParameter::CompareTo(SegmentParameter &p2)
{
	if (this->scaleFactor != p2.scaleFactor) return(1); 
	if (this->preType != p2.preType) return(1);
	if (this->postType != p2.postType) return(1);
	if (this->gradType != p2.gradType) return(1);
	if (this->preSize != p2.preSize) return(1);
	if (this->postSize != p2.postSize) return(1);
	if (this->gradSize != p2.gradSize) return(1);
	if (this->threshold != p2.threshold) return(1);
	if (this->absoluteThreshold != p2.absoluteThreshold) return(1);
	if (this->waterfall != p2.waterfall) return(1);
	return(0);
}

SegmentParameter::SegmentParameter(double aScaleType, int aPreType, int aPreSize, int aPostType, int aPostSize) : m_logger(LOG_GET_LOGGER("SegmentParameter"))
{
	scaleFactor = aScaleType;
	preType = aPreType;
	postType = aPostType;
	preSize = aPreSize;
	postSize = aPostSize;
	gradSize = 1;
	gradType = 1;
	threshold = 0.5;
	absoluteThreshold = 0;
	waterfall = 0;
}

SegmentParameter::SegmentParameter(double aScaleType, int aPreType, int aPostType) : m_logger(LOG_GET_LOGGER("SegmentParameter"))
{
	scaleFactor = aScaleType;
	preType = aPreType;
	postType = aPostType;
	preSize = 1;
	postSize = 1;
	gradSize = 1;
	gradType = 1;
	threshold = 0.5;
	absoluteThreshold = 0;
	waterfall = 0;
}

SegmentParameter::SegmentParameter() : m_logger(LOG_GET_LOGGER("SegmentParameter"))
{	
	scaleFactor = 1.0;
	preType = 0; 
	preSize = 1;
	postType = 1; 
	postSize = 1;
	gradSize = 1;
	gradType = 1;
	threshold = 0.5; 
	absoluteThreshold = 0;
	waterfall = 0;
}

SegmentParameter::~SegmentParameter()
{	
}