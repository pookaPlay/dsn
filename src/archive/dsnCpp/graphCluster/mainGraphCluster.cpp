
#include <sstream>
#include <algorithm>

#include <string>
#include <vector>
#include <fstream>


using std::ofstream;
using std::string;
#include "Logger.h"

#include <coin/OsiClpSolverInterface.hpp>
#include <coin/CoinPackedMatrix.hpp>
#include <coin/CoinPackedVector.hpp>

#include "GraphClustering.h"
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include "DadaDef.h"
#include "boost/program_options.hpp"
namespace po = boost::program_options;

static void printUsage(po::options_description &desc)
{
  std::cout << "graphCluster options \nVersion 1.0\n\n";
  std::cout << desc;
}

int main(int argc, char *argv[])
{
	LOG_CONFIGURE("log4cplus.properties");
	Logger logger(LOG_GET_LOGGER("dadaInt"));

	LOG_TRACE(logger, "Application started");

	po::options_description desc("Options");
	desc.add_options()
		("inputFile,i", po::value<string>()->default_value("inputGraph.txt"), "Input file (required)")
		("outputFile,o", po::value<string>()->default_value("outputGraph.txt"), "Output file")
		("command,c", po::value<string>()->default_value("kl"), "Which algoirithm")
		("numNodes,n", po::value<int>()->default_value(9), "num nodes in graph")
		("help,h", "Print help messages");
	
	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc),  vm);

		if (vm.count("help")) {
			printUsage(desc);
			return(0);
		}
		if (vm["inputFile"].as<string>()=="") {
			printUsage(desc);
			return(0);
		}        
		po::notify(vm); // throws on error, so do after help in case

	} catch(po::error& e) {
		std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
		printUsage(desc);
		return 0;
	}

    LOG_TRACE(logger, "Parsed command line");
	try{

	    string inputFile = vm["inputFile"].as<string>();
		string outputFile = vm["outputFile"].as<string>();
		string cmd = vm["command"].as<string>();
		int numNodes = vm["numNodes"].as<int>(); 
		
		gmm::row_matrix<gmm::wsvector<double> > K(numNodes, numNodes);

    	std::ifstream fin;

    	fin.open(inputFile); 

    	std::string line;
    	while (std::getline(fin, line))    
    	{   
        	
        	boost::char_separator<char> sep(" ");
        	StringTokenizer tok(line, sep);

        	StringTokenizer::iterator it = tok.begin();
			int u = boost::lexical_cast<int>(*it);	
			it++;
			int v = boost::lexical_cast<int>(*it);	
			it++;
			double w = boost::lexical_cast<double>(*it);

			K(u,v) = w; 
			K(v,u) = w; 
		}
		fin.close();

		LOG_INFO(logger, "Successfully imported graph");
		
		
	
		Clustering::GraphClustering gc;
		gc.Initialize(K);

		if (cmd == string("kl"))  gc.SolveHeuristicallyKL();
		else gc.Solve();

		std::ofstream fout;

    	fout.open(outputFile); 
		
		// Check solution
		const std::vector<double>& sol = gc.Solution();
		//std::cout << "solution: " << gmm::row_vector(sol) << std::endl;
		const std::vector<std::pair<unsigned int, unsigned int> >& emap = gc.Edgemap();
		
		for (unsigned int ei = 0; ei < sol.size(); ++ei) {
			fout << emap[ei].first << " " << emap[ei].second << " " << sol[ei] << "\n"; 			
		}

		fout.close();
		
	
	} catch (const std::exception& ex) {
			LOG_FATAL_EX(logger, "Unhandled exception in main", ex);
	}
	
	LOG_TRACE(logger, "Application exiting");
	return( 0 ); 
/*
	if (argc < 5) {
		std::cout << "Need to specify command input and output files:\n\n";
		std::cout << "		graphCluster cmd weightmatrix.txt numNodes soln.txt\n\n";
		std::cout << "		cmd = lp, lpco, kl\n\n";
		return(0); 	
	}
	std::string cmd = std::string(argv[1]); 
	std::string inputFile = std::string(argv[2]); 
	int numNodes = std::lexical_cast<int>(argv[3]); 
	std::string outputFile = std::string(argv[4]); 
	
	try {
		std::ifstream fin(inputFile);
		std::string temps; 
		fin >> temps; 
		std::cout << temps << "\n";
		fin.close(); 
	} catch(...) {
		std::cout << "No file?\n";
	}
	


	

	// Normalized similarity matrix
	//
	//    -  0.1  0.1 -0.2     -     -
	//  0.1    - -0.05    -     -     -
	//  0.1 -0.1    -    -     -     -
	// -0.2    -    -    -   0.1   0.1
	//    -    -    -  0.1     - -0.05
	//    -    -    -  0.1 -0.05     -
	//
	// corresponding to the weighted graph:
	//
	//   (2)                          (4)
	//   |  \                        /  |
	//   |  0.1                    0.1  |
	//   |     \                  /     |
	// -0.05   (0)---- -0.2 ----(3)   -0.05
	//   |     /                  \     |
	//   |  0.1                    0.1  |
	//   |  /                        \  |
	//   (1)                          (5)
	//
	// which should have the optimal partitioning: {0,1,2} {3,4,5}
	K(0, 1) = 0.1;
	K(0, 2) = 0.2;
	K(0, 3) = -0.2;
	K(1, 2) = -0.05;
	K(3, 4) = 0.1;
	K(3, 5) = 0.1;
	K(4, 5) = -0.05;

	return EXIT_SUCCESS;
	*/
}
