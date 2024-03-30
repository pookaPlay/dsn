#if !defined(ISegHighGUI_H__)
#define ISegHighGUI_H__

#include <vector>
#include "opencv2/opencv.hpp"
#include "SegmentWS.h"
#include "Logger.h"

static void onMouse(int event, int x, int y, int, void* );
static int onKeyboard( int key );
static void onTrackbar(int, void*);

class ISegHighGUI {
	
	public:
		std::vector<int> colMap;
		
		
		void Init(cv::Mat &img, cv::Mat &bseg, std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr< std::map<int, MamaVId> > &bmap);

		void Run();
		void Render();				
		void RenderGraph(cv::Mat &picy);		
		void GenerateLabels();

		ISegHighGUI();
		virtual ~ISegHighGUI();

		Logger m_logger;
		cv::Mat m_img, m_bseg, m_seg; 
		std::shared_ptr<MamaGraph> m_myGraph;
		std::shared_ptr< std::map<int, MamaVId> > m_bmap;
		double m_tmin, m_tmax; 

		int m_thresholdInt; 
};

#endif
