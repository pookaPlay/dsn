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

#include "DadaFeaturesVecEdges.h"
#include "DadaPoolerVecEdges.h"
#include "MamaException.h"
#include "VizMat.h"
using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

void DadaFeaturesVecEdges::GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode)
{
	LOG_TRACE_METHOD(m_logger, "GenerateFeatures");
	
	// work on basins	
}


DadaFeaturesVecEdges::DadaFeaturesVecEdges(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaFeatures(basins, myVId, param)
{
	m_type = "vecEdges";
	m_pool = make_unique<DadaPoolerVecEdges>(basins, myVId, param);
}

DadaFeaturesVecEdges::DadaFeaturesVecEdges(std::shared_ptr<DadaParam> &param)
	: DadaFeatures(param)
{
	m_type = "vecEdges";
	m_pool = make_unique<DadaPoolerVecEdges>(param);
}

DadaFeaturesVecEdges::~DadaFeaturesVecEdges()
{
}

