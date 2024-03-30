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

#include "DadaFeatures.h"
#include "KMeanFeatures.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"


void DadaFeatures::MergeFeatures(std::shared_ptr<DadaFeatures> &fp, MamaGraph &gp, MamaGraph &gc,
	std::map<int, MamaVId> &labelChild, std::map<MamaEId, vector<MamaEId> > &childParentEdge)
{
	m_pool->MergeFeatures(*(fp->GetPooler()), gp, gc, labelChild, childParentEdge); 
}

void DadaFeatures::SplitFeatures(std::shared_ptr<DadaFeatures> &fp, MamaGraph &gp, MamaGraph &gc, std::map<MamaVId, MamaVId> &parentChildVertex)
{
	m_pool->SplitFeatures(*(fp->GetPooler()), gp, gc, parentChildVertex); 
}

void DadaFeatures::PoolFeatures(MamaGraph &myGraph)
{
	m_pool->PoolFeatures(m_imgs, m_basins, myGraph);
}


void DadaFeatures::GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode)
{
	m_imgs.clear();
	m_imgs = imgs; 
}

void DadaFeatures::VizFeatureImages(int stepLast, double mag)
{
	int kb; 
	for (int i = 0; i < m_imgs.size(); i++) {
		LOG_INFO(m_logger, "Feature " << i);
		if (stepLast) {
			if (i < m_imgs.size()-1) kb = 1;
			else kb = 0;
		}
		else kb = 0;
		VizMat::DisplayFloat(m_imgs[i], "img", kb, static_cast<float>(mag));
	}
}

DadaFeatures::DadaFeatures(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaFeatures")),
	m_basins(basins),
	m_param(param),
	m_pool(make_unique<DadaPooler>(basins, myVId, param))
{	
	m_imgs.clear();
	m_type = "";
}

DadaFeatures::DadaFeatures(std::shared_ptr<DadaParam> &param)
	: m_logger(LOG_GET_LOGGER("DadaFeatures")),	
	m_param(param),
	m_pool(make_unique<DadaPooler>(param))
{	
	m_type = ""; 
}

DadaFeatures::~DadaFeatures()
{	
}

