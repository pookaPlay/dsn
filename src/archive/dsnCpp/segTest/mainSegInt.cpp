
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
using namespace std; 


#include "Logger.h"

#include "opencv2/opencv.hpp"
using namespace cv;

#include "boost/program_options.hpp" 
namespace po = boost::program_options;
#include <boost/filesystem.hpp>


#include "GdalMat.h"
#include "VizMat.h"

#include "SegmentParameter.h"
#include "SegmentWS.h"
#include "ThreshSegment_Int.h"

ThreshSegment_Int myISeg;

static void printUsage(po::options_description& desc)
{
	cout << "segInt options  \n";
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

		LOG_INFO(logger, "Segmenting image");
		myISeg.Init(img);
		myISeg.Run();

	}
	catch (...) {
		LOG_ERROR(logger, "Problem in main");
		return(0);
	}
	LOG_INFO(logger, "DONE!");
	return(1);
}
