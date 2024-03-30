#include "ISegHighGUI.h"
#include "MamaDef.h"

#include "opencv2/opencv.hpp"
#include "DadaWSUtil.h"
#include "VizMat.h"

using namespace cv;


#define THRESHOLD_RESOLUTION 10000
int localThreshold;

static ISegHighGUI m_myGUI; 

static void onTrackbar(int value, void*)
{
	//double t = (float) localThreshold / (float) THRESHOLD_RESOLUTION;	
	//myIO.SetThreshold(t);
	//myIO.GetGUI().Render();	
	m_myGUI.Render(); 
}

static void onMouse( int eid, int x, int y, int flags, void* )
{
	//EVENT_FLAG_CTRLKEY
	/*
    if( flags ==  (EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON)) {		// EVENT_MOUSEMOVE
		if ( eid == EVENT_LBUTTONDBLCLK) {
			//info(1, "Double click!\n");
			myIO.lineSegMode = 0;			
			//myIO.Split();
			myIO.GetGUI().Render();
		}
		if (myIO.lineSegMode == 0) {
			myIO.lineSegMode = 1;
			myIO.lineSeg.clear();			
			myIO.lineSeg.push_back( Point(x, y) );
		} else {
			myIO.lineSeg.push_back(Point(x, y));
			//line(myIO.GetImage(), myIO.lineSeg[ myIO.lineSeg.size()-1], myIO.lineSeg[ myIO.lineSeg.size()-2], CV_RGB(255, 255, 0), 2); 						
			//imshow( "Main", myIO.GetImage());				// opengl: updateWindow("Main"); 
		}
		return;
	}
    if( flags ==  (EVENT_FLAG_SHIFTKEY + EVENT_FLAG_LBUTTON)) {		// EVENT_MOUSEMOVE
		if (myIO.lineSegMode == 0) {
			myIO.lineSegMode = 1;
			myIO.lineSeg.clear();			
			myIO.lineSeg.push_back( Point(x, y) );
		} else {
			myIO.lineSegMode = 0;			
			myIO.lineSeg.push_back( Point(x, y) );
			
			Point p1 = myIO.ConvertToWorld( myIO.lineSeg[0] ); 
			Point p2 = myIO.ConvertToWorld( myIO.lineSeg[1] );
			//myIO.ToggleRect(p1, p2);
			myIO.GetGUI().Render(); 
		}
		return;
	}
	
	//if ( eid == EVENT_MOUSEMOVE && flags == EVENT_FLAG_CTRLKEY) {

	//}
	if( eid == EVENT_LBUTTONDOWN) {		// EVENT_MOUSEMOVE
		Point pin = Point(x,y);
		Point pout = myIO.ConvertToWorld( pin );
		myIO.Toggle( pout ); 	
		myIO.GetGUI().Render(); 
		return;
	}
	*/
	//info(1, "Screen point %i, %i -> world point %i, %i\n", ps.x, ps.y, pw.x, pw.y); 
}

static int onKeyboard( int key ) 
{	
    if (key == 'q') {
		destroyWindow("Main"); 
		destroyWindow("Threshold");
		return(0);
	}  
	return(1); 
	/*
	///////////////////////////////////
	// Gui 	
	else if (key == 'e') {
		myIO.showSegEdges = (myIO.showSegEdges + 1) % 2;
		myIO.GetGUI().Render();
	} else if (key == 's') {
		myIO.showSeg = (myIO.showSeg + 1) % 2;
		myIO.GetGUI().Render();
	} else if (key == 'h') {
		myIO.heatSegEdges = (myIO.heatSegEdges + 1) % 2;
		myIO.GetGUI().Render();
	} else if (key == 'i') {
		myIO.showImg = (myIO.showImg + 1) % 2;
		myIO.GetGUI().Render();
	} else if (key == 'l') {
		myIO.showSel = (myIO.showSel + 1) % 2;
		myIO.GetGUI().Render();
	}// else if (key == 'c') {
	//	myIO.GetSelections().clear();
	//	myIO.GetGUI().Render();
	//} 
	///////////////////////////////////
	// Commands	
	else if (key == 'c') {
		myIO.SetSegType(2);
		myIO.GetGUI().Render();
	} 
	else if (key == 'x') {
		myIO.SetSegType(0);
		myIO.GetGUI().Render();
	} 
	else if (key == 'v') {
		myIO.SetSegType(1);
		myIO.GetGUI().Render();
	} 
	else if (key == 'f') {
		myIO.NextFeature();
		myIO.GetGUI().Render();
	} 
	else if (key == 'g') {
		myIO.PrevFeature();
		myIO.GetGUI().Render();
	} 
	else if (key == '[') {
		myIO.PrevLayer();
		myIO.GetGUI().Render();
	} 
	else if (key == ']') {
		myIO.NextLayer();
		myIO.GetGUI().Render();
	} 
	else if (key == 'w') {
		myIO.ExportSegmentation();		
	} 
	else if (key == 'r') {
		myIO.ReportError();		
	} 
	else if (key == 'a') {
		myIO.AddTree();
		myIO.GetGUI().Render();
	} 

	///////////////////////////////////
	// Threshold
	else if (key == 'z') {
		//myIO.SetThresholdE();
		myIO.TrainEnsemble();
		float t = myIO.GetThreshold() * (float) THRESHOLD_RESOLUTION;
		localThreshold = (int) t;
		namedWindow( "Threshold", 1 );
		createTrackbar( "Slider", "Threshold", &localThreshold, THRESHOLD_RESOLUTION, onTrackbar );
		resizeWindow("Threshold", 500, 100);			

		myIO.GetGUI().Render();
	} 

	else if (key == '.') {		
		if (localThreshold < (THRESHOLD_RESOLUTION-1)) localThreshold += 1;
		float t = (float) localThreshold / (float) THRESHOLD_RESOLUTION;	
		myIO.SetThreshold(t);
		namedWindow( "Threshold", 1 );
		createTrackbar( "Slider", "Threshold", &localThreshold, THRESHOLD_RESOLUTION, onTrackbar );
		resizeWindow("Threshold", 500, 100);			

		myIO.GetGUI().Render();

	} else if (key == ',') {		
		if (localThreshold > 0) localThreshold -= 1;
		float t = (float) localThreshold / (float) THRESHOLD_RESOLUTION;	
		myIO.SetThreshold(t);
		namedWindow( "Threshold", 1 );
		createTrackbar( "Slider", "Threshold", &localThreshold, THRESHOLD_RESOLUTION, onTrackbar );
		resizeWindow("Threshold", 500, 100);			

		myIO.GetGUI().Render();

	} else if (key == 't') {
		if (!myIO.showThresh) {
			myIO.showThresh = 1;	
			float t = myIO.GetThreshold() * (float) THRESHOLD_RESOLUTION;
			localThreshold = (int) t;
			namedWindow( "Threshold", 1 );
			createTrackbar( "Slider", "Threshold", &localThreshold, THRESHOLD_RESOLUTION, onTrackbar );
			resizeWindow("Threshold", 500, 100);			
		} else {
			myIO.showThresh = 0;
			destroyWindow( "Threshold" ); 			
			myIO.GetGUI().Render();
		}
	} 
	///////////////////////////////////
	// Help	
	else if (key == '?') {
		info(1, "\nKeyboard Shortcuts\n");
		info(1, "------------------\n");
		info(1, "  q : quit\n");
		info(1, "GUI/VIEW\n");
		info(1, "  i : toggle image\n");
		info(1, "  e : toggle edges\n");
		info(1, "  s : toggle segmentation\n");		
		info(1, "  l : toggle labels/selection\n");
		info(1, "  h : how about some ground truth!?\n");
		info(1, "  w : write segmentation\n");
		info(1, "  r : report error\n");
		info(1, "COMMANDS\n");		
		info(1, "  a : add the tree\n");
		info(1, "  [ : move down layers\n");
		info(1, "  ] : move up layers (add as required)\n");
		info(1, "  f : next feature\n");
		info(1, "  g : last feature\n");
		info(1, "  z : WS-MST\n");
		info(1, "  x : WS-Cut\n");
		info(1, "  c : CC\n");
		info(1, "  t : adjust threshold\n");
		info(1, "  . : adjust threshold up\n");
		info(1, "  , : adjust threshold down\n");
		info(1, "------------------\n");
	}
	*/
	return(1);
}


void ISegHighGUI::Init(cv::Mat &img, cv::Mat &bseg, std::shared_ptr<MamaGraph> &myGraph, std::shared_ptr< std::map<int, MamaVId> > &bmap)
{

	m_myGUI.m_img = img;
	m_myGUI.m_bseg = bseg;
	m_myGUI.m_bmap = bmap;
	m_myGUI.m_myGraph = myGraph;
	m_myGUI.m_tmin = LARGEST_DOUBLE; 
	m_myGUI.m_tmax = SMALLEST_DOUBLE;

	// Now go through edges
	MamaGraph &gp = *(myGraph.get());
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gp); 
	
	for (eit = estart; eit != eend; eit++) {
		gp[*eit].wasWeight = gp[*eit].weight;
		if (gp[*eit].weight < m_myGUI.m_tmin) m_myGUI.m_tmin = gp[*eit].weight; 
		if (gp[*eit].weight > m_myGUI.m_tmax) m_myGUI.m_tmax = gp[*eit].weight;
	}
	namedWindow("Edges", 1);
	setMouseCallback("Edges", onMouse, 0);
	namedWindow("Segs", 1);
	setMouseCallback("Segs", onMouse, 0);

	namedWindow("Threshold", 1);
	createTrackbar("Slider", "Threshold", &localThreshold, THRESHOLD_RESOLUTION, onTrackbar);
	resizeWindow("Threshold", 500, 100);

	m_myGUI.Run();
	

}

void ISegHighGUI::Run()
{	
	this->Render(); 
	int kb = 1;
	while (onKeyboard(kb)) {		
		kb = waitKey(0);
	}
}

void ISegHighGUI::Render()
{
	// Run last threshold
	GenerateLabels(); 

	Mat picy, picy2; 
	VizMat::RenderEdgeSeg(m_img, m_seg, picy);

	VizMat::RenderColorSeg(m_seg, picy2);
	//this->RenderGraph(picy); 

	imshow("Edges", picy);
	imshow("Segs", picy2);
	
}

void ISegHighGUI::GenerateLabels()
{
	// should be 0 to 1
	double myThresh = static_cast<double>(localThreshold) / static_cast<double>(THRESHOLD_RESOLUTION);

	myThresh = myThresh * (m_tmax - m_tmin) + m_tmin; 
	LOG_INFO(m_logger, "Apply threshold " << myThresh << " in [" << m_tmin << "->" << m_tmax << "] from " << localThreshold);
	
	MamaGraph &gp = *(m_myGraph.get());
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(gp);

	for (eit = estart; eit != eend; eit++) {
		gp[*eit].weight = gp[*eit].wasWeight - myThresh;
	}

	DadaWSUtil::ThresholdLabelEdgeWeight(gp);

	// Now render from graphg
	m_seg = Mat::zeros(m_bseg.rows, m_bseg.cols, CV_32F);
	for (int j = 0; j < m_bseg.rows; j++) {
		for (int i = 0; i < m_bseg.cols; i++) {
			int myl = static_cast<int>(m_bseg.at<float>(j, i));
			MamaVId nid = m_bmap->operator[](myl);			
			m_seg.at<float>(j, i) = static_cast<float>(gp[nid].label);
		}
	}

}

void ISegHighGUI::RenderGraph(Mat &picy)
{
	LOG_INFO(m_logger, "Threshold: " << localThreshold); 
	//DadaWSUtil::ThresholdLabelEdgeWeight(gp);
	VizMat::RenderEdgeSeg(m_img, m_seg, picy); 
	/*
	for (int j = 0; j< picy.rows; j++) {
		for (int i = 0; i< picy.cols; i++) {
			picy.at<Vec3b>(j, i)[0] = static_cast<unsigned char>(m_img.at<float>(j, i));
			picy.at<Vec3b>(j, i)[1] = static_cast<unsigned char>(m_img.at<float>(j, i));
			picy.at<Vec3b>(j, i)[2] = static_cast<unsigned char>(m_img.at<float>(j, i));
		}
	}

	int w = picy.cols; 
	int h = picy.rows;	

	float imageContrast = 0.75f;
	float segColorContrast = 0.5f;
		
	//info(1, "Rendering %i pixels from %i nodes\n", numNodes, img->getNumNodes());
	if (myIO.showImg) {	// we have image pixel data	
		for(uint j=0; j< picy.rows; j++) {
		for(uint i=0; i< picy.cols; i++) {
			picy.at<Vec3b>(j, i)[0] = img.at<float>(j,i) * imageContrast;
			picy.at<Vec3b>(j, i)[1] = img.at<float>(j,i) * imageContrast;
			picy.at<Vec3b>(j, i)[2] = img.at<float>(j,i) * imageContrast;		
		}
		}
	}
		
	if (myIO.showSeg) {
		for(int j=0; j< seg.rows; j++) {
		for(int i=0; i< seg.cols; i++) {	
			int segid = (int) seg.at<float>(j, i);
			int cii = this->colMap[ segid % 64 ];
			int col1 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 0] * 255.0);
			int col2 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 1] * 255.0);
			int col3 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 2] * 255.0);
			picy.at<Vec3b>(j, i)[0] = col1 * segColorContrast; 
			picy.at<Vec3b>(j, i)[1] = col2 * segColorContrast; 
			picy.at<Vec3b>(j, i)[2] = col3 * segColorContrast; 
			if (myIO.showSel) {
				if (FindIn(myIO.GetSelections(), segid)) picy.at<Vec3b>(j, i)[2] = 255; 
			}
		}
		}
	}
	
	// segmentation edges	
	if (myIO.showSegEdges) {		
		//info(1, "Render seg\n");
		int sel = 0;
		for(int j=0; j< seg.rows; j++) {
		for(int i=0; i< seg.cols; i++) {	
			int vali = (int) seg.at<float>(j,i); 
			if (myIO.showSel) {
				if (FindIn(myIO.GetSelections(), vali)) sel = 1; 
				else sel = 0;
			}
			int boundary = 0; 
			if (j > 0) {					
				int vali2 = (int) seg.at<float>(j-1,i);
				if (vali != vali2) boundary = 1;
			}
			if (j < h-1) {	
				int vali2 = (int) seg.at<float>(j+1,i);
				if (vali != vali2) boundary = 1;
			}
			if (i > 0) {	
				int vali2 = (int) seg.at<float>(j,i-1);
				if (vali != vali2) boundary = 1;
			}		
			if (i < w-1) {	
				int vali2 = (int) seg.at<float>(j,i+1);
				if (vali != vali2) boundary = 1;
			}		
			if (boundary) {
				picy.at<Vec3b>(j, i)[1] = 255; 
			} else if (vali < 0.0f) {
				picy.at<Vec3b>(j, i)[0] = 25; 
				picy.at<Vec3b>(j, i)[1] = 0; 				
				picy.at<Vec3b>(j, i)[2] = 0; 
			} else {
				if (sel) picy.at<Vec3b>(j, i)[2] = 255; 
			}
		}
		}
	}	
	// segmentation edges	
	
	if (myIO.heatSegEdges) {		
		//info(1, "Render all seg\n");
		vector<Mat> &allSegs = myIO.GetAllSegs(); 
		Mat segCount = Mat::zeros(seg.rows, seg.cols, CV_32S);
		int maxCount = 0;
		for(int si=0; si< allSegs.size(); si++) {
		
			for(int j=0; j< seg.rows; j++) {
			for(int i=0; i< seg.cols; i++) {	
				int vali = (int) allSegs[si].at<float>(j,i); 
				int boundary = 0; 
				if (j > 0) {					
					int vali2 = (int) allSegs[si].at<float>(j-1,i);
					if (vali != vali2) boundary = 1;
				}
				if (j < h-1) {	
					int vali2 = (int) allSegs[si].at<float>(j+1,i);
					if (vali != vali2) boundary = 1;
				}
				if (i > 0) {	
					int vali2 = (int) allSegs[si].at<float>(j,i-1);
					if (vali != vali2) boundary = 1;
				}		
				if (i < w-1) {	
					int vali2 = (int) allSegs[si].at<float>(j,i+1);
					if (vali != vali2) boundary = 1;
				}		
				if (boundary) {
					segCount.at<int>(j, i) = segCount.at<int>(j, i)+1; 
					if (segCount.at<int>(j, i) > maxCount) maxCount = segCount.at<int>(j, i);
				} 
			}
			}
		}
		for(int j=0; j< seg.rows; j++) {
		for(int i=0; i< seg.cols; i++) {	
			if (segCount.at<int>(j, i) > 0) {
				float segfrac = floor(64.0f * (float) segCount.at<int>(j, i) / (float) maxCount);
				//int segi = max(63 - (int) segfrac, 0);
				int segi = (int) segfrac; 				
				int cii = this->colMap[ segi ];
				int col1 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 0] * 255.0);
				int col2 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 1] * 255.0);
				int col3 = (int) NEAREST_INT( jetcolormap[ cii * 3 + 2] * 255.0);
				picy.at<Vec3b>(j, i)[0] = col1; 
				picy.at<Vec3b>(j, i)[1] = col2; 
				picy.at<Vec3b>(j, i)[2] = col3; 			
			}
		}
		}
	}  		
	*/
}


/***********************************************************
***** Constructor/Destructor
***********************************************************/
ISegHighGUI::ISegHighGUI()
	: m_logger(LOG_GET_LOGGER("ISegHighGUI"))
{	
	RandPerm(64, this->colMap);
}

ISegHighGUI::~ISegHighGUI()
{
}

