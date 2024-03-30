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

#include <exception>
#include <vector>
#include <iostream>
#include <string>
using namespace std;


#include "Logger.h"

#include "boost/program_options.hpp" 
namespace po = boost::program_options;
#include <boost/filesystem.hpp>

#include "GdalMat.h"
#include "VizMat.h"
#include "Logger.h"
#include "SegmentVectorize.h"

#include "opencv2/opencv.hpp"
using namespace cv;

#include <fstream>

// should be replaced with connected components
int MyLabeler(Mat &label_image);

static void printUsage(po::options_description& desc)
{
	cout << "mamaVectorize options  \n";
	cout << desc;
}

int main(int argc, char**argv)
{
	if (!boost::filesystem::exists("log4cplus.properties")) LOG_CONFIGURE_BASIC();
	else LOG_CONFIGURE("log4cplus.properties");
	
	Logger logger(LOG_GET_LOGGER("main"));
	LOG_TRACE(logger, "Application started");

	po::options_description desc("Options");
	desc.add_options()
		("inputLabels,i", po::value<string>()->default_value(""), "Label file")
		("inputSeg,j", po::value<string>()->default_value(""), "Segmentation file")
		("output,o", po::value<string>()->default_value("temp.json"), "Output name")
		("label,l", po::value<double>()->default_value(-1.0), "Segment label to use")
		("scale,s", po::value<double>()->default_value(1.0), "Below 1.0 for less detail")
		("help,h", "Print help messages");

	// Parse options
	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);

		if (vm.count("help")) {
			printUsage(desc);			
			return(0);
		}
		po::notify(vm); // throws on error, so do after help in case 

	}
	catch (po::error& e) {
		LOG_INFO_EX(logger, "Problem parsing command line ", e);
		return 0;
	}
	LOG_TRACE(logger, "Command line parsed");

	try
	{
	
		string labelFileName = vm["inputLabels"].as<string>(); 
		string segFileName = vm["inputSeg"].as<string>();
		string outFileName = vm["output"].as<string>();
		double scale = vm["scale"].as<double>();
		double labeli = vm["label"].as<double>();

		if ((segFileName == "") && (labelFileName == "")) {
			LOG_INFO(logger, "You need to provide the label file or segmentation, or ideally both!");
			printUsage(desc);			
			return(0);
		}
		cv::Mat label, seg;

		if ((segFileName == "") && (labelFileName != ""))
		{			
			//GdalMat::Read2DAsFloat(segFileName, seg);
			GdalMat::Read2DAsFloat(labelFileName, label);

			seg = Mat::zeros(label.rows, label.cols, CV_32F);
			for (int j = 0; j < seg.rows; j++) {
				for (int i = 0; i < seg.cols; i++) {
					if (label.at<float>(j, i) > 0.0f) seg.at<float>(j, i) = 1.0f; 
				}
			}
			int numc = MyLabeler(seg); 
			LOG_INFO(logger, "Found " << numc << " components");
			SegmentVectorize vecSeg; 
			vecSeg.Clear(); 
			LOG_INFO(logger, "Vectorizing with scale " << scale); 
			vecSeg.Convert(seg, scale); 
			vecSeg.Export(outFileName);			

			LOG_INFO(logger, "Success!");
			return(0); 
		}

		if ((segFileName != "") && (labelFileName == ""))
		{
			GdalMat::Read2DAsFloat(segFileName, seg);
			
			SegmentVectorize vecSeg;
			vecSeg.Clear();
			LOG_INFO(logger, "Vectorizing with scale " << scale);
			vecSeg.Convert(seg, scale);
			vecSeg.Export(outFileName);

			LOG_INFO(logger, "Success!");
			return(0);
		}

		if ((segFileName != "") && (labelFileName != ""))
		{
			GdalMat::Read2DAsFloat(segFileName, seg);
			GdalMat::Read2DAsFloat(labelFileName, label);
			LOG_INFO(logger, "Vectorizing segments with label " << labeli);
			//seg = Mat::zeros(label.rows, label.cols, CV_32F);
			for (int j = 0; j < seg.rows; j++) {
				for (int i = 0; i < seg.cols; i++) {
					if ( fabs(label.at<float>(j, i) - labeli) > 0.1f) seg.at<float>(j, i) = -1.0f;
				}
			}
			SegmentVectorize vecSeg;
			vecSeg.Clear();
			LOG_INFO(logger, "Vectorizing with scale " << scale);
			vecSeg.Convert(seg, scale);
			vecSeg.Export(outFileName);

			LOG_INFO(logger, "Success!");
			return(0);
		}


	}
	catch (const std::exception& ex) {
		LOG_FATAL_EX(logger, "Unhandled exception in main", ex);
	}

	LOG_TRACE(logger, "Application exited");
	return(0);
}



int MyLabeler(Mat &label_image)
{
	int label_count = 2; // starts at 2 because 0,1 are used already

	try {
		// RBP 
		// This check is very useful but could be removed for efficiency
		double minVal, maxVal;
		minMaxLoc(label_image, &minVal, &maxVal); //, &minLoc, &maxLoc );
		if (maxVal > 1.1f) {
			cout << "Error: unexpected label format\n"; 
			throw std::exception(); 
		}

		for (int y = 0; y < label_image.rows; y++) {
			for (int x = 0; x < label_image.cols; x++) {
				int vali = boost::numeric_cast<int>(label_image.at<float>(y, x));
				if (vali == 1) {
					Rect rect;
					floodFill(label_image, cv::Point(x, y), cv::Scalar((float)label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);
					label_count++;
				}
			}
		}
		label_count = label_count - 2;
		// Take one off for consistency 
		label_image = label_image - 1;

	}
	catch (cv::Exception& e) 	{
		throw std::exception();
	}

	return(label_count);
}
