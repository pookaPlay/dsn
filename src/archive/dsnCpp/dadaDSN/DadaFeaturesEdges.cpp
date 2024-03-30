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

#include "DadaFeaturesEdges.h"
#include "DadaPoolerEdges.h"
#include "SegmentPreprocess.h"
#include "MamaException.h"
#include "VizMat.h"
using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

void DadaFeaturesEdges::GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode)
{
	LOG_TRACE_METHOD(m_logger, "GenerateFeatures");
	if (imgs.size() < 1) BOOST_THROW_EXCEPTION(Unexpected());

	SegmentParameter segParam; 
	
	m_imgs.clear();

	int preTypeUpto = 1;
	int preSizeUpto = 3;	
	int gradTypeUpto = 2;
	int postTypeUpto = 1;
	int postSizeUpto = 3;
	int scaleFactorUpto = 3; 
	int gradSize = 1;
	double scaleFactor = 1.0f;

	int localPostSizeUpto, localPreSizeUpto; 

	for (int preType = 0; preType <= preTypeUpto; preType++) {
		if (preType > 0) localPreSizeUpto = preSizeUpto;
		else localPreSizeUpto = 1;
		for (int preSize = 1; preSize <= localPreSizeUpto; preSize += 2) {
			
			for (int gradType = 1; gradType <= gradTypeUpto; gradType++) {
				
				for (int postType = 0; postType <= postTypeUpto; postType++) {
					if (postType > 0) localPostSizeUpto = postSizeUpto;
					else localPostSizeUpto = 1;
					for (int postSize = 1; postSize <= localPostSizeUpto; postSize += 2) {

						for (int scaleType = 0; scaleType < 4; scaleType++) {
							if (scaleType == 0) {
								scaleFactor = 1.0;
							}
							else if (scaleType == 1) {
								scaleFactor = 0.75;
							}
							else if (scaleType == 2) {
								scaleFactor = 0.66;
							}
							else if (scaleType == 3) {
								scaleFactor = 0.5;
							}
							
							segParam.scaleFactor = scaleFactor;
							segParam.preType = preType;
							segParam.preSize = preSize;
							segParam.postType = postType;
							segParam.postSize = postSize;
							segParam.gradType = gradType;
							segParam.gradSize = gradSize;	

							m_imgs.resize(m_imgs.size()+1);
							SegmentPreprocess::Calculate(imgs[0], m_imgs[m_imgs.size() - 1], segParam); 
						}
					}
				}
			}
		}
	}

	
}


DadaFeaturesEdges::DadaFeaturesEdges(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaFeatures(basins, myVId, param)
{
	m_type = "edges";
	m_pool = make_unique<DadaPoolerEdges>(basins, myVId, param);
}

DadaFeaturesEdges::DadaFeaturesEdges(std::shared_ptr<DadaParam> &param)
	: DadaFeatures(param)
{
	m_type = "edges";
	m_pool = make_unique<DadaPoolerEdges>(param);
}

DadaFeaturesEdges::~DadaFeaturesEdges()
{
}

