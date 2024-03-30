#include <string>
using namespace std;

#include "Opt.h"
#include "Info.h"
static Info info;
#include "ACD.h"
#include "opencv2/opencv.hpp"
using namespace cv;

#include "SynData.h"
#include "FisherDiscriminant.h"
#include "SvmDiscriminant.h"
#include "Whiten.h"
#include "Logger.h"
#include <boost/exception/all.hpp>

static void printUsage()
{
	cout << "i2mlTest command options\n\n"; 
	cout << "       fisher    Run fisher on synthetic\n";
	cout << "       svm		  Run libsvm on synthetic\n";
	cout << "       acd		  Run acd on synthetic\n";
	cout << "       whiten    Run whiten\n\n";
}

int main(int argc, char**argv) 
{ 
	LOG_CONFIGURE("log4test.properties");
	Logger logger(LOG_GET_LOGGER("main"));
	
	try {
		Opt::add("inputFile|i", "", "Input file");
		Opt::add("yFile|y", "", "Predicted file");
		Opt::add("segFile|s", "", "Seg file");
		Opt::add("outputFile|o", "", "Output file");
		Opt::add("verbosity|v", 1000, "Verbosity of stdout 0-5 ..5 is most verbose..");

	   // Parse options 
		try  {
			Opt::parse(argc, argv);
		} catch (...)  {
			cout << "Problem parsing options\n";
			return(0);
		}

		info.setVerbosity( Opt::getInt("verbosity") );

		string inputFile = Opt::getString("inputFile");	
		string outputFile = Opt::getString("outputFile");	
		string segFile = Opt::getString("segFile");	

		vector < string > extras; 
		Opt::getExtras(extras);
	
		if (extras.size() < 1) {
			printUsage();
			string temp = Opt::optionString();
			cout << temp;
			return(0);		
		}
  
		if (extras[0] == "fisher") {        
			vector< vector<float> > mlData;
			vector<float> labels; 
			float sigma;

		
			SynData syn;
			syn.SynTwoClassData(100, 2, mlData, labels, 1.5f); 

			Mat mlabels = Mat(labels);	
			info(1, "Labels is %i rows  %i cols\n", mlabels.rows, mlabels.cols);

			Mat mdata = Mat(); 
			for(int i=0; i< mlData.size(); i++) {
				Mat m = Mat(mlData[i]);
				Mat mt = m.t();
				mdata.push_back(mt);
			}
			info(1, "Data is %i rows  %i cols\n", mdata.rows, mdata.cols);

			FisherDiscriminant fish;

			Mat mweights = Mat::ones(mdata.rows, 1, CV_32F);
			fish.Train(mdata, mlabels, mweights);
		
			Mat result = Mat();
			info(1, "Now applying\n");
			fish.Apply(mdata, result);
			vector<float> resultv;
			result.copyTo(resultv);

			syn.PlotData(mlData, labels, 256, 20); 
		
			fish.TrainThreshold(result, mlabels, mweights);
			fish.ApplyThreshold(result, result);
			
			result.copyTo(resultv);
			syn.PlotData(mlData, resultv, 256, 20); 

			return(1);
		} 
		else if (extras[0] == "svm") {        
			vector< vector<float> > mlData;
			vector<float> labels; 
			float sigma;

		
			SynData syn;
			syn.SynTwoClassData(100, 2, mlData, labels, 10.0f); 

			Mat mlabels = Mat(labels);	
			info(1, "Labels is %i rows  %i cols\n", mlabels.rows, mlabels.cols);

			Mat mdata = Mat(); 
			for(int i=0; i< mlData.size(); i++) {
				Mat m = Mat(mlData[i]);
				Mat mt = m.t();
				mdata.push_back(mt);
			}
			info(1, "Data is %i rows  %i cols\n", mdata.rows, mdata.cols);

			syn.PlotData(mlData, labels, 256, 20); 

			SvmDiscriminant fish;
			Mat result = Mat();
			Mat mweights = Mat::ones(mdata.rows, 1, CV_32F);
			fish.Train(mdata, mlabels, mweights);
			fish.SaveModel("testme");
			fish.Apply(mdata, result);
			
			vector<float> resultv;
			result.copyTo(resultv);			
				
			result.copyTo(resultv);
			syn.PlotData(mlData, resultv, 256, 20); 
		
			return(1);
		} 
		else if (extras[0] == "whiten") {        
			vector< vector<float> > mlData;
			vector<float> labels; 
			float sigma;

		
			SynData syn;
			syn.SynTwoClassData(100, 2, mlData, labels, 10.0f); 

			Mat mlabels = Mat(labels);	
			info(1, "Labels is %i rows  %i cols\n", mlabels.rows, mlabels.cols);

			Mat mdata = Mat(); 
			for(int i=0; i< mlData.size(); i++) {
				Mat m = Mat(mlData[i]);
				Mat mt = m.t();
				mdata.push_back(mt);
			}
			info(1, "Data is %i rows  %i cols\n", mdata.rows, mdata.cols);

			syn.PlotData(mlData, labels, 256, 10); 

			Whiten white, white2;
			white.Estimate(mdata);
			white.Apply(mdata);
			cout << "Mean: " << white.M() << "\n";
			cout << "Covar: " << white.IC() << "\n";
		
			white2.Estimate(mdata);
			cout << "New Mean: " << white2.M() << "\n";
			cout << "New Covar: " << white2.IC() << "\n";

			return(1);
		} 

		else if (extras[0] == "acd") {
			try {
				vector< vector<float> > mlData;
				vector<float> labels;
				int N = 200;
				labels.clear(); labels.resize(N, 0);


				SynData syn;				
				syn.Syn2DGaussian(N, 1.0, 0.8, mlData);
				syn.PlotData(mlData, labels, 256, 20);
				waitKey(0); 
				

				Mat x = Mat::zeros(mlData.size(), 1, CV_64F);
				Mat y = Mat::zeros(mlData.size(), 1, CV_64F);
				cout << "With " << x.rows << " and " << x.cols << "\n";
				for (int i = 0; i < mlData.size(); i++) {
					if (mlData[i].size() != 2) cout << "Something up!\n";
					x.at<double>(i, 0) = static_cast<double>(mlData[i][0]);
					y.at<double>(i, 0) = static_cast<double>(mlData[i][1]);
				}
				
				Mat result;
				ACD myACD;
				myACD.Train(x, y);
				myACD.Apply(x, y, result);

				vector<float> vr; vr.clear();
				for (int i = 0; i < result.rows; i++) vr.push_back(static_cast<float>(result.at<double>(i)));
				syn.PlotData(mlData, vr, 256, 20);
				waitKey(0);

			}
			catch (...) {
				cout << "Got it!\n";
			}
			return(1);
		}
		else {
			printUsage();
			string temp = Opt::optionString();
			cout << temp;
			return(0);		
		}
	} catch(boost::exception &e) {
		std::cerr << boost::diagnostic_information(e);
	}
	
	return( 0 ); 
}

