#if !defined(DadaIID_H__)
#define DadaIID_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "SegmentWS.h"
#include "Logger.h"
#include "DadaWSGT.h"
#include "DadaEval.h"

class DadaIID 
{
	public:		
		
				
		static void TrainThreshold(double &thresh, MamaGraph &myM, DadaWSGT &gt, DadaError &err);

		DadaIID();
		virtual ~DadaIID();
	private:				

		Logger m_logger;	
		static Logger s_logger;

}; 

#endif 

