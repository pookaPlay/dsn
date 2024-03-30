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

#include "DadaFeaturesMoments.h"
#include "DadaPoolerMoments.h"
#include "SegmentPreprocess.h"
#include "MamaException.h"
#include "VizMat.h"
using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

void DadaFeaturesMoments::GenerateFeatures(std::vector<cv::Mat> &imgs, int trainMode)
{
	LOG_TRACE_METHOD(m_logger, "GenerateFeatures");
	if (imgs.size() < 1) BOOST_THROW_EXCEPTION(Unexpected());
	
	m_imgs.clear(); 

	if (imgs.size() == 1) {
		m_imgs.resize(1);
		m_imgs[0] = imgs[0].clone(); 
	}
	else if (imgs.size() == 3) {
		m_imgs.resize(6);
		for (int i = 0; i < 3; i++) {
			m_imgs[i] = imgs[i].clone();
			m_imgs[i+3] = imgs[i].clone();
		}

		LOG_INFO(m_logger, "Color image space conversion");
		for (int j = 0; j < imgs[0].rows; j++) {
			for (int i = 0; i < imgs[0].cols; i++) {
				double r, g, b;
				RGB2LAB(imgs[2].at<float>(j, i), imgs[1].at<float>(j, i), imgs[0].at<float>(j, i), r, g, b); 
				m_imgs[3].at<float>(j, i) = static_cast<float>(r); 
				m_imgs[4].at<float>(j, i) = static_cast<float>(g);
				m_imgs[5].at<float>(j, i) = static_cast<float>(b);
			}
		}
		
	}	
	
}


DadaFeaturesMoments::DadaFeaturesMoments(cv::Mat &basins, std::shared_ptr< std::map<int, MamaVId> > &myVId, std::shared_ptr<DadaParam> &param)
	: DadaFeatures(basins, myVId, param)
{
	m_type = "moments";
	m_pool = make_unique<DadaPoolerMoments>(basins, myVId, param);
}

DadaFeaturesMoments::DadaFeaturesMoments(std::shared_ptr<DadaParam> &param)
	: DadaFeatures(param)
{
	m_type = "moments";
	m_pool = make_unique<DadaPoolerMoments>(param);
}

DadaFeaturesMoments::~DadaFeaturesMoments()
{
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void DadaFeaturesMoments::RGB2XYZ(
	float &		sR,
	float &		sG,
	float &		sB,
	double&			X,
	double&			Y,
	double&			Z)
{
	double R = static_cast<double>(sR) / 255.0;
	double G = static_cast<double>(sG) / 255.0;
	double B = static_cast<double>(sB) / 255.0;

	double r, g, b;

	if (R <= 0.04045)	r = R / 12.92;
	else				r = pow((R + 0.055) / 1.055, 2.4);
	if (G <= 0.04045)	g = G / 12.92;
	else				g = pow((G + 0.055) / 1.055, 2.4);
	if (B <= 0.04045)	b = B / 12.92;
	else				b = pow((B + 0.055) / 1.055, 2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

void DadaFeaturesMoments::RGB2LAB(float &sR, float &sG, float &sB, double& lval, double& aval, double& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;
	this->RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X / Xr;
	double yr = Y / Yr;
	double zr = Z / Zr;

	double fx, fy, fz;
	if (xr > epsilon)	fx = pow(xr, 1.0 / 3.0);
	else				fx = (kappa*xr + 16.0) / 116.0;
	if (yr > epsilon)	fy = pow(yr, 1.0 / 3.0);
	else				fy = (kappa*yr + 16.0) / 116.0;
	if (zr > epsilon)	fz = pow(zr, 1.0 / 3.0);
	else				fz = (kappa*zr + 16.0) / 116.0;

	lval = 116.0*fy - 16.0;
	aval = 500.0*(fx - fy);
	bval = 200.0*(fy - fz);
}
