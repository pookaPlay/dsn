#if !defined(DadaWSACD_H__)
#define DadaWSACD_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "SegmentWS.h"
#include "DadaFeatures.h"
#include "DadaParam.h"
#include "DadaSegmenter.h"
#include "DadaWS.h"

class DadaWSACD : public DadaWS
{
	public:		

		void InitACDTrain();
		void InitACDApply();				

		void InitACDFeatures();
		void InitACDEdgeFeatures();
		void PickNewEdges();
		void SetACD()
		{
			m_seg->SetACD(this);
		};

		DadaWSACD(std::shared_ptr<DadaParam> &param);
		virtual ~DadaWSACD();
	protected:					
		vector<MamaVId> m_origVertices;
		vector<MamaVId> m_newVertices;		
		map< MamaVId, MamaVId > m_newOldVertex;
		map< MamaVId, MamaVId > m_oldNewVertex;
		vector< pair<MamaVId, MamaVId> > m_newEdgeOrigVertices;

		Logger m_logger;				
}; 

#endif 

