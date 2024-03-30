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

#include "DadaBatch.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "DadaWSUtil.h"
#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include <fstream>
#include "DadaSLIC.h"
#include "ISegHighGUI.h"

void DadaBatch::InitParam(int expi)
{	
	m_param = std::make_shared<DadaParam>();
	//int numImages = m_trainImgNames.size();
	m_numImages = 10; 
	m_numTestImages = 10;
	m_numTreesPerImage = 11; 	

	m_param->ensembleType = "forest";
	m_param->ensembleSize = m_numImages * m_numTreesPerImage;
	m_param->ensembleDepth = 21;
	m_param->classifierType = "stump";
	m_param->treeType = "alt";	// error, split, merge, alt

	//m_param->classifierLossType = "weighted";
	//m_param->finalLossType = "weighted"; 
	//m_param->posWeight = 0.1; 
	//m_param->negWeight = 1.0;
	
	m_param->segmentationType = "cc";
	m_param->trainSegmentationType = 1; 
	
	m_param->featureType = "histogram";
	//m_param->featureType = "histogram";
	
	m_param->featureSubsetSize = 25;
	m_param->slicK = 500; 

	m_param->modelName = "okaytd";

	m_dada = std::make_shared<DadaWS>(m_param);

	m_badTrain.clear();
	m_badTest.clear();
	
	/*
	m_badTrain[4] = 1;
	m_badTrain[7] = 1;
	m_badTrain[8] = 1;
	m_badTrain[10] = 1;
	m_badTrain[11] = 1;
	m_badTrain[12] = 1;
	m_badTrain[15] = 1;

	m_badTest[0] = 1;
	m_badTest[1] = 1;
	m_badTest[9] = 1;
	m_badTest[4] = 1;
	m_badTest[5] = 1;
	m_badTest[6] = 1;
	m_badTest[14] = 1;
	m_badTest[10] = 1;
	m_badTest[17] = 1;
	*/
}


void DadaBatch::RunBaseline(int trainMode)
{
	try {
		DadaWSUtil::SetRandomSeed(1);

		m_param = std::make_shared<DadaParam>();
		//int numImages = m_trainImgNames.size();
		m_numImages = 50;
		m_numTreesPerImage = 1;

		m_param->ensembleType = "";
		m_param->ensembleSize = 1; 
		m_param->ensembleDepth = 1;
		m_param->classifierType = "stump";
		//param->classifierLossType = "weighted";
		m_param->classifierLossType = "std";
		m_param->segmentationType = "cc";
		m_param->trainSegmentationType = 0;

		m_param->featureType = "edge";	

		m_param->featureSubsetSize = -1;
		m_param->slicK = 500;

		m_param->modelName = "std";

		m_dada = std::make_shared<DadaWS>(m_param);

		LoadImage(0, 1);
		m_dada->InitSegmenter();

		if (trainMode) {
			m_dada->GetClassParam()->I() = 0; 
			m_dada->GetClassParam()->T() = 0.0;
			this->TrainThresholds(1);
			m_dada->Save();
		}
		else {
			m_dada->Load(); 
		}
	}
	catch (MamaException &e) {
		std::cerr << boost::diagnostic_information(e);
		std::cerr << e.what();
	}
}

void DadaBatch::Eval(int trainMode, int numImages)
{
	try {
		int firstone = 1;

		if (trainMode) {
			m_finalTrainErrors.clear();
			if (numImages > m_trainImgNames.size()) numImages = m_trainImgNames.size();
		}
		else {
			m_finalTestErrors.clear();
			if (numImages > m_imgNames.size()) numImages = m_imgNames.size();
		}

		double avgError = 0.0; 
		double minError = LARGEST_DOUBLE;
		double maxError = SMALLEST_DOUBLE;
		for (int imgi = 0; imgi < numImages; imgi++) {
			
			this->LoadImage(imgi, trainMode); 

			if (firstone) {
				firstone = 0;
				m_dada->InitSegmenter();
				m_dada->Load();
				LOG_INFO(m_logger, "Load complete");
			}
			
			m_dada->ApplySegmenter(); 			

			DadaError &err = m_dada->EvalSegmenterMST();
			if (trainMode) m_finalTrainErrors.push_back(err.GetError()); 
			else		   m_finalTestErrors.push_back(err.GetError());
			
			avgError += err.GetError(); 
			minError = std::min(minError, err.GetError()); 
			maxError = std::max(maxError, err.GetError());
			Mat myResult;
			m_dada->GetImgLabels(myResult);
			VizMat::DisplayEdgeSeg(m_inputGray, myResult, "r", 1, 1.0);			
			
			LOG_INFO(m_logger, "  Eval " << imgi << ": " << err.GetError()); 			
		}
		avgError = avgError / numImages; 
		LOG_INFO(m_logger, "FINAL: " << avgError << " ( " << minError << " -> " << maxError << " )"); 
	}
	catch (MamaException &e) {
		std::cerr << boost::diagnostic_information(e);
		std::cerr << e.what();
	}
}


void DadaBatch::Train(int trainMode)
{
	try {		
		int firstone = 1; 
				
		int imgCount = 0; 

		DadaWSUtil::ChooseRandomFeatures(m_trainIds, m_trainImgNames.size(), m_numImages);

		for (int imgii = 0; imgii < m_trainIds.size(); imgii++) {

			//if (!m_badTrain.count(imgi)) {

				//LOG_INFO(m_logger, "####### Image " << imgi); 
				LOG_INFO(m_logger, "####### Image " << m_trainIds[imgii]);

				this->LoadImage(m_trainIds[imgii], 1);

				for (int ti = 0; ti < m_numTreesPerImage; ti++) {

					if (firstone) {
						firstone = 0;
						m_dada->InitSegmenter();
						m_dada->TrainInit();
					}

					m_dada->TrainStep();
				}

				//imgCount++;
				//if (imgCount == m_numImages) break;
			//}
		}
		
		m_dada->TrainFinish();

		m_dada->Save();

		if (!trainMode) {
			//double trainError = this->TrainThresholds(1);
			//m_exp1Train.push_back(trainError);
			m_exp1Train.push_back(0.0);
			m_dada->Save();
			double testError = this->TrainThresholds(0);
			m_exp1Test.push_back(testError); 			
		}
		else {
			this->TrainThresholds(1);
			m_dada->Save();
		}
		
	}
	catch (MamaException &e) {
		std::cerr << boost::diagnostic_information(e);
		std::cerr << e.what();
	}
}

double DadaBatch::TrainThresholds(int trainMode)
{
	LOG_INFO(m_logger, "########## Training Threshold");
	m_trainThresholds.clear();
	m_trainErrors.clear();
	double avgThreshold = 0.0; 
	double avgError = 0.0;
	double minError = LARGEST_DOUBLE;
	double maxError = SMALLEST_DOUBLE;

	int imgCount = 0;
	bool goodOne; 
	for (int imgi = 0; imgi < m_trainImgNames.size();  imgi++) {

		if (trainMode) {
			if (!m_badTrain.count(imgi)) goodOne = true;
			else goodOne = false;
		}
		else {
			if (!m_badTest.count(imgi)) goodOne = true;
			else goodOne = false;
		}

		if (goodOne) {
			//LOG_INFO(m_logger, "####### Image " << imgi);
			LoadImage(imgi, trainMode);

			m_dada->SetOutputThreshold(0.0);
			m_dada->ApplySegmenter();
			//LOG_INFO(m_logger, "  Has " << m_dada->GetNumLabels() << " labels");

			m_dada->TrainThreshold();

			double outputMin, outputMax, thresh;
			m_dada->GetOutputThresh(outputMin, outputMax, thresh);

			DadaError &trainErr = m_dada->GetError();
			LOG_INFO(m_logger, "    Img : " << imgi << " threshold " << thresh << " had err " << trainErr.GetError());
			m_trainThresholds.push_back(thresh);
			m_trainErrors.push_back(trainErr.GetError());
			avgThreshold += thresh;

			m_dada->UpdateOutputThreshold(thresh);
			//LOG_INFO(m_logger, "  Has " << m_dada->GetNumLabels() << " labels"); 

			Mat myResult;
			m_dada->GetImgLabels(myResult);
			VizMat::DisplayEdgeSeg(m_inputGray, myResult, "r", 1, 1.0);

			avgError += trainErr.GetError();
			minError = std::min(minError, trainErr.GetError());
			maxError = std::max(maxError, trainErr.GetError());

			imgCount++;
			if (trainMode) {
				if (imgCount == m_numImages) break;
			}
			else {
				if (imgCount == m_numTestImages) break;
			}

		}
	}

	avgThreshold = avgThreshold / static_cast<double>(imgCount); 
	m_dada->SetOutputThreshold(avgThreshold);
	LOG_INFO(m_logger, "Average threshold: " << avgThreshold);
	avgError = avgError / static_cast<double>(imgCount);
	LOG_INFO(m_logger, "FINAL: " << avgError << " ( " << minError << " -> " << maxError << " )");
	return(avgError); 

}

void DadaBatch::LoadImage(int imgi, int trainMode)
{
	
	this->Load(imgi, trainMode, 1);
	this->GenerateBasinsWithSLIC(1);

	m_dada->InitFromBasins(m_basins);
	m_dada->InitFeatures(m_inputStack, 1);
	m_dada->InitGroundTruth(m_inputSeg);
	
}

void DadaBatch::Load(int index, int trainMode, int viz)
{
	string imgs, gts;

	if (trainMode) {
		imgs = m_trainImgNames[index];
		gts = m_trainSegNames[index];
	}
	else {
		imgs = m_imgNames[index];
		gts = m_segNames[index];
	}

	m_inputColor = cv::imread(imgs); //Load as color if possible

	Mat imgb = cv::imread(imgs, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale	
	imgb.convertTo(m_inputGray, CV_32F);		// then convert to float	

	// splits etc
	m_inputStack.clear(); 
	if (m_inputColor.channels() == 1) {
		//LOG_INFO(m_logger, "Single chanel input");
		m_inputStack.resize(1); 
		m_inputColor.convertTo(m_inputStack[0], CV_32F);		// then convert to float	
	}
	else if (m_inputColor.channels() == 3) {
		//LOG_INFO(m_logger, "Color 3 chanel input");
		m_inputStack.resize(3);
		vector<Mat> channels(3);
		split(m_inputColor, channels);
		channels[0].convertTo(m_inputStack[0], CV_32F);		// then convert to float	
		channels[1].convertTo(m_inputStack[1], CV_32F);		// then convert to float	
		channels[2].convertTo(m_inputStack[2], CV_32F);		// then convert to float	
	}
	else BOOST_THROW_EXCEPTION(Unexpected()); 

	FitsMat::Read2DFitsAsFloat(gts, m_inputSeg);

	//img *= 1. / 255;
	//cvtColor(img, img, CV_BGR2Luv);

	if (viz) {		
		Mat imgc = imread(imgs);
		namedWindow("input");
		imshow("input", imgc);
				
		VizMat::DisplayEdgeSeg(m_inputGray, m_inputSeg, "seg", 1, 1.0);
	}
}

void DadaBatch::GenerateBasinsFromGray()
{
	SegmentWS aSegmenter;
	SegmentParameter aSegParam(1.0, 0, 1, 1, 1);
	aSegmenter.Init(m_inputGray, aSegParam);
	aSegmenter.GetBasinLabels(m_basins);
}

void DadaBatch::GenerateBasinsWithSLIC(int viz)
{
	DadaSLIC::Run(m_inputColor, m_basins, m_param->slicK);
	if (viz) VizMat::DisplayEdgeSeg(m_inputGray, m_basins, "slic", 1, 1.0);
}

DadaBatch::DadaBatch()
	:  m_logger(LOG_GET_LOGGER("DadaBatch"))
{		
	m_imgNames.clear();
	m_segNames.clear();
}

DadaBatch::~DadaBatch()
{	
}



/*

void DadaBatch::InitEval()
{
m_segmenter = std::make_unique<DadaWS>(m_param);

m_segmenter->InitFromBasins(m_basins);
m_segmenter->InitFeatures(m_inputStack, 0);
m_segmenter->InitGroundTruth(m_inputSeg);
m_segmenter->InitSegmenter();
try {
m_segmenter->Load();
}
catch (...) {
LOG_INFO(m_logger, "##############################################");
LOG_INFO(m_logger, "No saved classifier so processing with default");
LOG_INFO(m_logger, "##############################################");
}

}

DadaError & DadaBatch::Eval()
{
m_segmenter->ApplySegmenter();
return(m_segmenter->EvalSegmenterMST());
}


void DadaBatch::InitTestList(string fname)
{
m_imgNames.clear();
m_segNames.clear();

ifstream fin(fname);
char buf[4096];
while (!fin.eof()) {
fin.getline(buf, 4096);
string temps(buf);
if (temps == string("")) break;
m_imgNames.push_back(temps);
fin.getline(buf, 4096);
temps = string(buf);
m_segNames.push_back(temps);
}

}


void DadaBatch::InitTrainList(string fname)
{
BOOST_THROW_EXCEPTION(NotImplemented());
}


void DadaBatch::Eval(std::shared_ptr<DadaParam> &param)
{
try {
m_param = param;

if (param->randomSeed >= 0) {
DadaWSUtil::SetRandomSeed(param->randomSeed);
}

m_errors.clear();

for (int i = 0; i < m_imgNames.size(); i++) {

Load(i, 1);
//GenerateBasinsFromGray();
GenerateBasinsWithSLIC();

InitEval();
DadaError err = Eval();
m_errors.push_back(err);

// viz
Mat myResult;
m_segmenter->GetImgLabels(myResult);
VizMat::DisplayEdgeSeg(m_inputGray, myResult, "out", 0, 1.0);
LOG_INFO(m_logger, "Error: " << err.GetError() << " ( " << err.GetPosError() << ", " << err.GetNegError() << ")");

}

}
catch (MamaException &e) {
std::cerr << boost::diagnostic_information(e);
std::cerr << e.what();
}
}

*/