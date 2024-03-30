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

#include "DadaFeaturesEdge.h"
#include "DadaPoolerEdges.h"
#include "SegmentPreprocess.h"
#include "MamaException.h"
#include "VizMat.h"
using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

void DadaFeaturesEdge::GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode)
{
	LOG_TRACE_METHOD(m_logger, "GenerateFeatures");
	if (imgs.size() < 1) BOOST_THROW_EXCEPTION(Unexpected());
	
	m_imgs.resize(1);
	
	SegmentParameter aSegParam(1.0, 0, 1, 1, 1);		
	SegmentPreprocess::Calculate(imgs[0], m_imgs[0], aSegParam); 
}


DadaFeaturesEdge::DadaFeaturesEdge(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaFeatures(basins, myVId, param)
{
	m_type = "edge";
	m_pool = make_unique<DadaPoolerEdges>(basins, myVId, param);
}

DadaFeaturesEdge::DadaFeaturesEdge(std::shared_ptr<DadaParam> &param)
	: DadaFeatures(param)
{
	m_type = "edge";
	m_pool = make_unique<DadaPoolerEdges>(param);
}

DadaFeaturesEdge::~DadaFeaturesEdge()
{
}

