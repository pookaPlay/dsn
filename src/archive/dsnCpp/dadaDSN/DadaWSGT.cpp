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

#include "DadaWSGT.h"

using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include "MamaException.h"
#include "MamaDef.h"
#include "DadaWSGT.h"

void DadaWSGT::Print()
{
	for (auto &it : m_totals) {
		LOG_INFO(m_logger, it.first << ": " << it.second);
	}
}

void DadaWSGT::FinalizeCounts()
{	
	m_totals.clear();

	for (auto &it : m_vertexLabels) {
		for (auto &it2 : it.second) {
			if (!m_totals.count(it2.first)) m_totals[it2.first] = 0.0;
			m_totals[it2.first] += it2.second;
		}
	}

	double tsum = 0.0;
	double tssum = 0.0;

	for (auto &it : m_totals) {
		tssum += (it.second * it.second - it.second) / 2.0;
		tsum += it.second;
	}

	double myCount = (tsum*tsum - tsum) / 2.0;
	m_posCount = tssum; 
	m_negCount = myCount - tssum; 

	if ((m_posWeight < 0.0) || (m_negWeight < 0.0)) {
		m_posWeight = (m_negCount) / (m_posCount + m_negCount);
		m_negWeight = (m_posCount) / (m_posCount + m_negCount);
		LOG_INFO(m_logger, " Even weights are " << m_posWeight << ", " << m_negWeight);
	}
}

void DadaWSGT::Clear()
{
	m_vertexLabels.clear();
	m_vertexExtraPos.clear();
	m_vertexExtraNeg.clear();
	m_vertexErrorPos.clear();
	m_vertexErrorNeg.clear();

	m_errorNeg = 0.0;
	m_errorPos = 0.0;
	m_extraNeg = 0.0;
	m_extraPos = 0.0;
}

DadaWSGT::DadaWSGT()
	: m_logger(LOG_GET_LOGGER("DadaWSGT"))	  
{			
	Clear(); 
}

DadaWSGT::~DadaWSGT()
{	
}
