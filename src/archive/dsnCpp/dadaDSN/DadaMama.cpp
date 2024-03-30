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

#include "DadaMama.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"
#include "DadaParam.h"

void DadaMama::Train(cv::Mat &img, cv::Mat &bseg, cv::Mat &oseg, cv::Mat &mseg)
{
	LOG_TRACE_METHOD(m_logger, "Train");

	vector<Mat> tempVec; 
	tempVec.clear();
	tempVec.resize(1);
	tempVec[0] = img;
	m_trainData.Init(tempVec, oseg, mseg, bseg);

	DadaWS dada(m_param);

	dada.InitFromBasins(m_trainData.GetBasins());
	dada.InitFeatures(m_trainData.GetInput(), 1);
	dada.InitGroundTruth(m_trainData.GetGroundTruth());

	dada.InitSegmenter();
	dada.TrainSegmenter();
	dada.Save();

}

void DadaMama::Apply(cv::Mat &img, cv::Mat &bseg)
{
	LOG_TRACE_METHOD(m_logger, "Apply");
	
	m_dada = make_unique<DadaWS>(m_param);

	m_dada->InitFromBasins(bseg);
	vector<Mat> tempVec;
	tempVec.clear();
	tempVec.resize(1);
	tempVec[0] = img;

	m_dada->InitFeatures(tempVec, 0);	
	m_dada->InitSegmenter();

	m_dada->Load();
	m_dada->ApplySegmenter();	
	m_dada->GetImgLabels(m_outputSeg);	
}

void DadaMama::Update(double threshold)
{
	LOG_TRACE_METHOD(m_logger, "Update");
	m_dada->UpdateOutputThreshold(threshold); 
	m_dada->GetImgLabels(m_outputSeg);	
}

DadaMama::DadaMama()
	: m_logger(LOG_GET_LOGGER("DadaMama")),
	m_param(std::make_shared<DadaParam>())
{	
	m_param->ensembleType = "forest";
	m_param->ensembleSize = 10;
	m_param->ensembleDepth = 20;
	m_param->classifierType = "stump";
	m_param->segmentationType = "ws";
	m_param->classifierLossType = "weighted";
	//m_param->classifierLossType = "std";
	m_param->posWeight = 0.1;
	m_param->negWeight = 1.0;

	m_param->featureType = "edges";
	m_param->featureMergeType = "max";
	m_param->treeType = "merge";
	m_param->featureSubsetSize = 10;
	m_param->modelName = "tempDadaMama"; 
}

DadaMama::~DadaMama()
{	
}

