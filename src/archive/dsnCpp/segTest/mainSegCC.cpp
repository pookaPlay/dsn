#include <string>
#include <vector>
#include <fstream>
using namespace std; 

#include "opencv2/opencv.hpp"
using namespace cv;
#include "boost/program_options.hpp" 
namespace po = boost::program_options;
#include <boost/filesystem.hpp>

#include "SegmentParameter.h"
#include "SegmentEval.h"
#include "SegmentWS.h"
#include "Logger.h"
#include "GdalMat.h"
#include "VizMat.h"

static void printUsage(po::options_description& desc)
{
	cout << "segTest options  \n";
	cout << desc;
}

int main(int argc, char *argv[])
{
	if (!boost::filesystem::exists("log4cplus.properties")) LOG_CONFIGURE_BASIC();
	else LOG_CONFIGURE("log4cplus.properties");
	
	Logger logger(LOG_GET_LOGGER("main"));
	LOG_TRACE(logger, "Application started");

	po::options_description desc("Options");
	desc.add_options()
		("inputFile,i", po::value<string>()->default_value(""), "Image file")
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

		string inputFile = vm["inputFile"].as<string>();

		if (inputFile == "") {
			LOG_INFO(logger, "You need to provide the image file");
			printUsage(desc);
			return(0);
		}


		Mat img;
		GdalMat::Read2DAsFloat(inputFile, img);
		LOG_INFO(logger, "Loaded image");
		//VizMat::DisplayFloat(img, "in", 0, 0.5);

		SegmentParameter param;
		SegmentWS segment;

		segment.Init(img, param);
		//param.scaleFactor = 0.25; 
		param.threshold = 0.5;
		param.absoluteThreshold = 1; 
		segment.UpdateThreshold(param);

		Mat seg, rseg;
		int segCount, segMin, segMax;
		segment.GetLabelsWithStats(seg, segCount, segMin, segMax);
		LOG_INFO(logger, "Seg at " << segCount << " from " << segMin << " -> " << segMax);

		double thresh, energy;
		SegmentEval segEval;
		segEval.ComputeCCThreshold(segment.GetGraph(), thresh, energy);

		param.threshold = thresh;
		param.absoluteThreshold = 1;
		segment.UpdateThreshold(param);
		
		segment.GetLabelsWithStats(seg, segCount, segMin, segMax);
		LOG_INFO(logger, "CC Seg at " << segCount << " from " << segMin << " -> " << segMax);
		VizMat::DisplayEdgeSeg(img, seg, "seg", 0, 1.0);

	}
	catch (...) {
		LOG_ERROR(logger, "Problem in main");
		return(0);
	}
	LOG_INFO(logger, "DONE!");
	return(1);
	
}
