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

#include "DadaFeaturesKMeans.h"
#include "KMeanFeatures.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "VizMat.h"
#include "MamaDef.h"
#include "FitsMat.h"

void DadaFeaturesKMeans::GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode)
{
	string modelName = "knnstuff";
	// Do some preprocessing to populate edge "features"	
	m_imgs.clear();
	Mat img = imgs[0]; 
	KMeanFeatures knn;
	if (trainMode) {
		knn.LearnFeatures(img, m_param->numFeatures, m_param->featureWinRadius, m_param->featureWhiten);
		knn.SaveModel(m_param->modelName);
	}
	else {
		//cout << "Loading dictionary\n";
		knn.LoadModel(m_param->modelName);
	}

	knn.GenerateFeatures(img, m_imgs, "triMap", m_param->featureWinRadius);
}


DadaFeaturesKMeans::DadaFeaturesKMeans(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaFeatures(basins, myVId, param)
{
	m_type = "kmeans";
	//m_pool = make_unique<DadaPooler>(basins, myVId, param);
}

DadaFeaturesKMeans::DadaFeaturesKMeans(std::shared_ptr<DadaParam> &param)
	: DadaFeatures(param)
{
	m_type = "kmeans";
	//m_pool = make_unique<DadaPooler>(basins, myVId, param);
}

DadaFeaturesKMeans::~DadaFeaturesKMeans()
{
}

