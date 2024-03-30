#if !defined(DadaWSGT_H__)
#define DadaWSGT_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "SegmentWS.h"

class DadaWSGT 
{
	public:		
		map< MamaVId, map<int, double> > & Labels() {
			return(m_vertexLabels);
		};

		map< MamaVId, double> & VErrorPos() {
			return(m_vertexErrorPos);
		};

		map< MamaVId, double> & VErrorNeg() {
			return(m_vertexErrorNeg);
		};

		map< MamaVId, double> & VExtraPos() {
			return(m_vertexExtraPos);
		};

		map< MamaVId, double> & VExtraNeg() {
			return(m_vertexExtraNeg);
		};

		double & ErrorNeg() {
			return(m_errorNeg);
		}
		double & ErrorPos() {
			return(m_errorPos);
		}
		double & ExtraNeg() {
			return(m_extraNeg);
		}
		double & ExtraPos() {
			return(m_extraPos);
		}

		double & PosWeight() {
			return(m_posWeight);
		}
		double & NegWeight() {
			return(m_negWeight);
		}

		double & PosCount() {
			return(m_posCount);
		}
		double & NegCount() {
			return(m_negCount);
		}

		void Print(); 
		void FinalizeCounts(); 

		void Clear(); 

		DadaWSGT();
		virtual ~DadaWSGT();
	private:

		map< MamaVId, map<int, double> > m_vertexLabels;
		map< MamaVId, double > m_vertexWeights;

		map< MamaVId, double > m_vertexExtraPos;
		map< MamaVId, double > m_vertexExtraNeg;
		map< MamaVId, double > m_vertexErrorPos;
		map< MamaVId, double > m_vertexErrorNeg;

		double m_extraNeg;
		double m_extraPos;

		double m_errorNeg; 
		double m_errorPos;

		double m_negWeight;
		double m_posWeight;

		double m_negCount;
		double m_posCount;
		map<int, double> m_totals;

		Logger m_logger;				
}; 

#endif 

