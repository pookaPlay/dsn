#include <string>
#include <vector>
#include <fstream>
using namespace std; 

#include "opencv2/opencv.hpp"
using namespace cv;
#include "boost/program_options.hpp" 
namespace po = boost::program_options;
#include <boost/filesystem.hpp>


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
		("inputSeg,j", po::value<string>()->default_value(""), "Segmentation file")
		("output,o", po::value<string>()->default_value("temp.json"), "Output name")
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
		string segFile = vm["inputSeg"].as<string>();
		string outFileName = vm["output"].as<string>();

		if ((inputFile == "") && (segFile == "")) {
			LOG_INFO(logger, "You need to provide the image and segmentation file");
			printUsage(desc);			
			return(0);
		}
		cv::Mat img, seg;

		GdalMat::Read2DAsFloat(inputFile, img); 	
		GdalMat::Read2DAsFloat(segFile, seg); 	
	
		LOG_INFO(logger, "Input image");
		VizMat::DisplayFloat(img, "in", 0.1, 1.0); 

		VizMat::DisplayEdgeSeg(img, seg, "seg", 0, 1.0); 

	} catch(...) {
		LOG_ERROR(logger, "Problem in main");
		return(0);
	}
    LOG_INFO(logger, "DONE!");
	return(1); 
}
