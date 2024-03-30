#include "SegmentWS.h"
#include "DadaDef.h"
#include "DadaException.h"
#include "Morphology.h"
#include "VizMat.h"

#include "opencv2/opencv.hpp"
using namespace cv;


void SegmentWS::Init(cv::Mat &img, SegmentParameter &param)
{
	LOG_TRACE(m_logger, "Init");

	//LOG_DEBUG(m_logger, "Segment with " << param.scaleType << ", " << param.preType << ", " << param.postType);

	cv::Mat segIn;
	// edge detection
	this->Preprocess(img, segIn, param);
	// create graph
	this->InitGraph(segIn);	
	// prep data structures with two layer segmentation 
	this->ProcessMST(param);

	LOG_TRACE(m_logger, "Init Done");
}

void SegmentWS::InitNoPre(cv::Mat &img, SegmentParameter &param)
{
	LOG_TRACE(m_logger, "InitNoPre");
	// create graph
	this->InitGraph(img);
	// prep data structures with two layer segmentation 
	this->ProcessMST(param);

	LOG_TRACE(m_logger, "InitNoPre Done");
}


void SegmentWS::GetBasinLabels(cv::Mat &segf)
{
	segf = Mat::zeros(this->myH, this->myW, CV_32F);

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);

	for (nit = nstart; nit != nend; nit++) {
		int li = this->myM[*nit].label;		
		segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float)(li + 1);
	}
}

void SegmentWS::GetLabels(cv::Mat &segf)
{
	LOG_TRACE(m_logger, "GetLabels");
	segf = Mat::zeros(this->myH, this->myW, CV_32F);
	
	if (this->myThresholdIndex < 0) BOOST_THROW_EXCEPTION(Unexpected("You must call UpdateThreshold first"));
	
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);

	for (nit = nstart; nit != nend; nit++) {
		int li = this->myM[*nit].label;

		int li2 = this->myM2[li].label;
		segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float)(li2 + 1);
	}	
	LOG_TRACE(m_logger, "GetLabels Done");
}

void SegmentWS::GetLabelsWithStats(cv::Mat &segf, int &segCount, int &segMin, int &segMax)
{
	LOG_TRACE(m_logger, "GetLabelsWithStats");
	segf = Mat::zeros(this->myH, this->myW, CV_32F);
	
	if (this->myThresholdIndex < 0) BOOST_THROW_EXCEPTION(Unexpected("You must call UpdateThreshold first")); 

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	map< int, int> labelCount;
	labelCount.clear();
	segMin = LARGEST_INT;
	segMax = SMALLEST_INT;
	for (nit = nstart; nit != nend; nit++) {
		int li = this->myM[*nit].label;
		int li2 = this->myM2[li].label;
		int ilabel = li2 + 1;
		if (ilabel > segMax) segMax = ilabel;
		if (ilabel < segMin) segMin = ilabel;
		if (labelCount.count(ilabel) == 0) labelCount[ilabel] = 1;
		else labelCount[ilabel] = labelCount[ilabel] + 1;
		segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float) ilabel;
	}	
	segCount = labelCount.size();	
	LOG_TRACE(m_logger, "GetLabelsWithStats Done");
}

void SegmentWS::UpdateThreshold(SegmentParameter &param)
{
	LOG_TRACE(m_logger, "UpdateThreshold");
	
	if (this->mySortedMergeDepth.size() < 1) BOOST_THROW_EXCEPTION(UnexpectedSize("There is no merge graph to threshold"));
	
	double localThresh;

	if (!param.absoluteThreshold) {
		// Threshold gets converted from an absolute [0..1] range to an index into the sorted list	
		if (param.threshold <= 0.0) {
			this->myThresholdIndex = 0;
		}
		else if (param.threshold >= 1.0) {
			this->myThresholdIndex = this->mySortedMergeDepth.size() - 1;
		}
		else {
			this->myThresholdIndex = (int)floor(param.threshold * (double)(this->mySortedMergeDepth.size()));
			// and check bounds.. just to be safe
			if (this->myThresholdIndex > this->mySortedMergeDepth.size() - 1) this->myThresholdIndex = this->mySortedMergeDepth.size() - 1;
			if (this->myThresholdIndex < 0) this->myThresholdIndex = 0;
		}

		localThresh = this->mySortedMergeDepth[this->myThresholdIndex].first + FEATURE_TOLERANCE;
		LOG_TRACE(m_logger, "Local thresh is " << localThresh << " from " << this->mySortedMergeDepth[0].first << " -> " << this->mySortedMergeDepth[this->mySortedMergeDepth.size() - 1].first);
	}
	else {
		localThresh = param.threshold;
		this->myThresholdIndex = 0;
		LOG_TRACE(m_logger, "Local thresh is absolute " << localThresh); 
	}

	int segCount = 0;
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM2);

	for (nit = nstart; nit != nend; nit++) {
		this->myM2[*nit].label = -1;
	}

	for (nit = nstart; nit != nend; nit++) {
		if (this->myM2[*nit].label < 0) {
			int compSize = this->LabelVertices(*nit, segCount, localThresh);			
			segCount++;
		}
	}	
	LOG_TRACE(m_logger, "UpdateThreshold Done");
}

int SegmentWS::LabelVertices(MamaVId id, int label, double localThresh)
{
	vector<MamaVId> newguy; 
	newguy.clear();
	newguy.push_back(id);
	
	int ci = 0; 
	
	while (newguy.size() > 0) {
		MamaVId nextId = newguy[newguy.size() - 1];
		newguy.pop_back();
		this->myM2[nextId].label = label;
		ci++;
		// get neighbors
		MamaNeighborIt nit, nstart, nend;
		std::tie(nstart, nend) = adjacent_vertices(nextId, this->myM2);
		for (nit = nstart; nit != nend; nit++) {
			if (this->myM2[*nit].label < 0) {
				std::pair < MamaEId, bool > peid = edge(nextId, *nit, this->myM2);
				if (!(peid.second)) BOOST_THROW_EXCEPTION(Unexpected());							
				if (this->myM2[peid.first].weight < localThresh) {
					newguy.push_back(*nit);					
				}
			}
		}
	}

	return(ci);
}

void SegmentWS::ProcessMST(SegmentParameter &param)
{
	LOG_TRACE(m_logger, "WatershedMST");
	
	// clear second layer graph
	this->myM2.clear();
	
	// sort the edges
	vector< pair< double, MamaEId > > sortedEdges;
	sortedEdges.clear();
	sortedEdges.resize(num_edges(this->myM));
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(this->myM);
	int ei = 0;
	for (eit = estart; eit != eend; eit++) {
		pair< double, MamaEId > eid(this->myM[*eit].weight, *eit);
		sortedEdges[ei] = eid;
		ei++;
	}
	// flood fill from the bottom 
	sort(sortedEdges.begin(), sortedEdges.end(), std::less< pair<double, MamaEId> >());
	LOG_INFO(m_logger, "Sorted WS: " << sortedEdges[0].first << " -> " << sortedEdges[sortedEdges.size() - 1].first);
	// make sure we are starting with an empty forrest
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	for (nit = nstart; nit != nend; nit++) {
		this->myM[*nit].label = -1;
	}

	this->mySortedMergeDepth.clear();
	int labelCount = 0;

	// run kruskalish algorithm
	for (int i = 0; i < sortedEdges.size(); i++) {

		MamaEId eid = sortedEdges[i].second;
		MamaVId id1 = source(eid, this->myM);
		MamaVId id2 = target(eid, this->myM);

		if ((this->myM[id1].label < 0) && (this->myM[id2].label < 0)) {		// new basin
			this->myM[id1].label = labelCount;
			this->myM[id2].label = labelCount;
			MamaVId nid = add_vertex(this->myM2);
			if (labelCount != (int)nid) BOOST_THROW_EXCEPTION(Unexpected());
			// track the minimums to calculate depth
			this->myM2[nid].weight = this->myM[eid].weight;			
			this->myM2[nid].label = labelCount;
			labelCount++;
		}
		else if (this->myM[id1].label < 0) {		// extend basin
			this->myM[id1].label = this->myM[id2].label;
		}
		else if (this->myM[id2].label < 0) {		// extend basin
			this->myM[id2].label = this->myM[id1].label;
		}
		else {			
			if (this->myM[id1].label != this->myM[id2].label) {
				// see if we have seen this pair before
				MamaEId mid; bool newEdge;
				std::tie(mid, newEdge) = add_edge(this->myM[id1].label, this->myM[id2].label, this->myM2);
				if (newEdge) {
					MamaVId mid1 = source(mid, this->myM2);
					MamaVId mid2 = target(mid, this->myM2);
					if (param.waterfall != 1) {
						// calculate depth 
						double nMin = (this->myM2[mid1].weight < this->myM2[mid2].weight) ? this->myM2[mid1].weight : this->myM2[mid2].weight;
						double depth = this->myM[eid].weight - nMin;
						this->myM2[mid].weight = depth;
						// presort for threshold
						pair<double, MamaEId> sp(depth, mid);
						this->mySortedMergeDepth.push_back(sp);
					}
					else {
						this->myM2[mid].weight = this->myM[eid].weight;
					}
				}
			}
		}
	}
	// sort for threshold
	if (param.waterfall != 1) {
		sort(this->mySortedMergeDepth.begin(), this->mySortedMergeDepth.end(), std::less< pair<double, MamaEId> >());		
		LOG_INFO(m_logger, "Sorted CC: " << mySortedMergeDepth[0].first << " -> " << mySortedMergeDepth[mySortedMergeDepth.size() - 1].first);
	}
	else {
		this->WSCInitEdge();
	}
	this->myThresholdIndex = -1;

	LOG_TRACE(m_logger, "  Watershed found " << num_vertices(this->myM2) << " basins and " << num_edges(this->myM2) << " potential merges");
	LOG_TRACE(m_logger, "WatershedMST Done");	
}


void SegmentWS::WSCInitEdge()
{
	LOG_TRACE(m_logger, "WatershedCuts");
	LOG_INFO(m_logger, "I am running waterfall");
	this->mySortedMergeDepth.clear();

	MamaEdgeIt eit, estart, eend;
	MamaNeighborEdgeIt neit, nestart, neend;
	int ii = 0;
	std::tie(estart, eend) = edges(this->myM2);
	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, this->myM2);
		MamaVId id2 = target(*eit, this->myM2);

		double allReal1 = LARGEST_DOUBLE;
		double allReal2 = LARGEST_DOUBLE;
		MamaEId wIndex1, wIndex2;

		if (out_degree(id1, this->myM2) > 1) {
			std::tie(nestart, neend) = out_edges(id1, this->myM2);
			for (neit = nestart; neit != neend; neit++) {
				if (*neit != *eit) {
					if (this->myM2[*neit].weight < allReal1) {
						allReal1 = this->myM2[*neit].weight;
						wIndex1 = (MamaEId)(*neit);
					}
				}
			}
		}
		else allReal1 = SMALLEST_DOUBLE;

		if (out_degree(id2, this->myM2) > 1) {
			std::tie(nestart, neend) = out_edges(id2, this->myM2);
			for (neit = nestart; neit != neend; neit++) {
				if (*neit != *eit) {
					if (this->myM2[*neit].weight < allReal2) {
						allReal2 = this->myM2[*neit].weight;
						wIndex2 = (MamaEId)(*neit);
					}
				}
			}
		}
		else allReal2 = SMALLEST_DOUBLE;

		double allReal;
		MamaEId wIndex;
		if (allReal1 > allReal2) {
			allReal = allReal1;
			wIndex = wIndex1;
		}
		else {
			allReal = allReal2;
			wIndex = wIndex2;
		}


		double threshVal = this->myM2[*eit].weight - allReal;

		//this->myM2[*eit].wIndex = wIndex;
		this->myM2[*eit].wasWeight = threshVal;
		pair<double, MamaEId> sp(threshVal, *eit);
		this->mySortedMergeDepth.push_back(sp);
	}
	// sort for threshold
	sort(this->mySortedMergeDepth.begin(), this->mySortedMergeDepth.end(), std::less< pair<double, MamaEId> >());

	for (eit = estart; eit != eend; eit++) {
		double newWeight = this->myM2[*eit].wasWeight;
		this->myM2[*eit].wasWeight = this->myM2[*eit].weight;
		this->myM2[*eit].weight = newWeight;
	}

	LOG_TRACE(m_logger, "WatershedCuts Done");
}


void SegmentWS::Preprocess(cv::Mat &img, cv::Mat &out, SegmentParameter &param)
{
	LOG_TRACE(m_logger, "Preprocess");
	try	{
		Mat gradIn, magOut, scaleOut, smoothOut, postOut;	

		////////////////////////////////////
		// SCALE 
		
		// Rescale as required
		if (param.scaleFactor < 1.0) {			
			Size mysz = Size((int)((double)img.cols * param.scaleFactor), (int)((double)img.rows * param.scaleFactor));
			resize(img, scaleOut, mysz, 0.0, 0.0, INTER_CUBIC);	  //1 is linear INTER_CUBIC, INTER_LANCZOS4
			gradIn = scaleOut;
		} else gradIn = img.clone();

		////////////////////////////////////
		// PRE-FILTER 
		if (param.preType == 1) {
			int winSize = param.preSize * 2 + 1;
			bilateralFilter(gradIn, smoothOut, winSize, (25.0*(double)param.preSize), 1);
			gradIn = smoothOut;			
		}
		else if (param.preType == 2) {			
			Morphology::Reconstruction(gradIn, smoothOut, RECON_OPEN, param.preSize);
			gradIn = smoothOut;
		}
		else if (param.preType == 3) {			
			Morphology::Reconstruction(gradIn, smoothOut, RECON_CLOSE, param.preSize);
			gradIn = smoothOut;
		}

		////////////////////////////////////
		// GRADIENT
		if (param.gradType == 1) {
			Mat grad_x, grad_y, tempOut;
			Scharr(gradIn, grad_x, CV_32F, 1, 0); //, scale, delta, BORDER_DEFAULT );
			Scharr(gradIn, grad_y, CV_32F, 0, 1); //, scale, delta, BORDER_DEFAULT );				
			cartToPolar(grad_x, grad_y, magOut, tempOut); //Mat()); //temp2); 
		}
		else if (param.gradType == 2) {			
			Morphology::Gradient(gradIn, magOut, param.gradSize);
		}
		else if (param.gradType == 3) {
			Laplacian(gradIn, magOut, CV_32F);						
		}
		else {
			magOut = gradIn;
		}
		

		////////////////////////////////////
		// POST-FILTER
		if (param.postType == 1) {
			int winSize = param.postSize * 2 + 1;
			GaussianBlur(magOut, postOut, Size(winSize, winSize), 0.0);			
		}
		else if (param.postType == 2) {			
			Morphology::Open(magOut, postOut, param.postSize);
		}
		else if (param.postType == 3) {			
			Morphology::Reconstruction(magOut, postOut, RECON_OPEN, param.postSize);
		}
		else {
			postOut = magOut;
		}
		
		////////////////////////////////////
		// UNSCALE
		if (param.scaleFactor < 1.0) {
			Size origsz = Size(img.cols, img.rows);
			resize(postOut, out, origsz, 0.0, 0.0, INTER_CUBIC);	  //1 is linear INTER_CUBIC, INTER_LANCZOS4 	
		} else out = postOut;

	} catch (cv::Exception& e) 
	{
		BOOST_THROW_EXCEPTION(Unexpected()); 
	}

	LOG_TRACE(m_logger, "Preprocess Done");
}

void SegmentWS::InitGraph(cv::Mat &img)
{
	LOG_TRACE(m_logger, "InitGraph");
	//	LOG_INFO(m_logger, "Creating graph structure\n");

	this->myM.clear();

	this->myW = img.cols;
	this->myH = img.rows;
	int N = this->myW * this->myH;

	//Mat edgePicy = Mat::zeros( this->H, this->W, CV_32F);
	// Add vertex for each pixel
	int ti = 0;
	vector<int> nodeIds; nodeIds.resize(N);
	for (int j = 0; j< this->myH; j++) {
		for (int i = 0; i< this->myW; i++) {
			MamaVId nid = add_vertex(this->myM);
			if (ti != (int)nid) BOOST_THROW_EXCEPTION(Unexpected("Problem building boost graph"));
			this->myM[nid].x = i;
			this->myM[nid].y = j;
			float val1 = img.at<float>(this->myM[nid].y, this->myM[nid].x);
			this->myM[nid].weight = static_cast<double>(val1); 
			ti++;
		}
	}
	// Add edges of pixel graph
	int ii = 0;
	for (int j = 0; j< this->myH; j++) {
		for (int i = 0; i< this->myW; i++) {
			if (j > 0) add_edge(ii, ii - this->myW, this->myM); // N		
			if (j < (this->myH - 1)) add_edge(ii, ii + this->myW, this->myM); // S		
			if (i > 0) add_edge(ii, ii - 1, this->myM); // W		
			if (i < (this->myW - 1)) add_edge(ii, ii + 1, this->myM); // E
			// increment
			ii = ii + 1;
		}
	}

	// Now put weights on edges
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(this->myM);
	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, this->myM);
		MamaVId id2 = target(*eit, this->myM);
		// get pixel values 
		float val1 = img.at<float>(this->myM[id1].y, this->myM[id1].x);
		float val2 = img.at<float>(this->myM[id2].y, this->myM[id2].x);
		// average for edge weight
		float vala = (val1 + val2) / 2.0f;
		this->myM[*eit].weight = (double)vala;
	}

	LOG_TRACE(m_logger, "  Image graph has " << num_vertices(this->myM) << " vertices and " << num_edges(this->myM) << " edges");	
	LOG_TRACE(m_logger, "InitGraph Doone");
}


void SegmentWS::MarkerSegmentation(cv::Mat &img, cv::Mat &mark, SegmentParameter &param)
{
	LOG_TRACE(m_logger, "MarkerSegmentation");

	cv::Mat segIn;
	// edge detection
	this->Preprocess(img, segIn, param);
	// create graph
	this->InitMarkerGraph(segIn, mark);
	// prep data structures with two layer segmentation 
	this->ProcessMarkerGraph(param);
	// prep data structures with two layer segmentation 
	this->PropagateMarkers(param);

	LOG_TRACE(m_logger, "MarkerSegmentation Done");
}

void SegmentWS::GetMarkerLabels(cv::Mat &segf)
{
	LOG_TRACE(m_logger, "GetMarkerLabels");
	segf = Mat::zeros(this->myH, this->myW, CV_32F);

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);

	for (nit = nstart; nit != nend; nit++) {
		int li = this->myM[*nit].label;

		int li2 = this->myM2[li].ownedBy;
		segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float)(li2 + 1);
	}
	LOG_TRACE(m_logger, "GetMarkerLabels Done");
}


void SegmentWS::InitMarkerGraph(cv::Mat &img, cv::Mat &mark)
{
	LOG_TRACE(m_logger, "InitGraph");
	//	LOG_INFO(m_logger, "Creating graph structure\n");

	this->myM.clear();

	this->myW = img.cols;
	this->myH = img.rows;
	int N = this->myW * this->myH;
	this->myMarkers.clear();
	this->myMaxMarker = -1;
	//Mat edgePicy = Mat::zeros( this->H, this->W, CV_32F);
	// Add vertex for each pixel
	int ti = 0;
	vector<int> nodeIds; nodeIds.resize(N);
	for (int j = 0; j< this->myH; j++) {
		for (int i = 0; i< this->myW; i++) {
			MamaVId nid = add_vertex(this->myM);
			if (ti != (int)nid) BOOST_THROW_EXCEPTION(Unexpected("Problem building boost graph"));
			this->myM[nid].x = i;
			this->myM[nid].y = j;
			this->myM[nid].label = -1;
			// set marker
			int marki = static_cast<int>(mark.at<float>(j, i)); 
			this->myM[nid].ownedBy = marki;
			this->myMarkers[marki] = 1;
			if (marki > this->myMaxMarker) this->myMaxMarker = marki;

			float val1 = img.at<float>(this->myM[nid].y, this->myM[nid].x);
			this->myM[nid].weight = static_cast<double>(val1);
			ti++;
		}
	}
	// Add edges of pixel graph
	int ii = 0;
	for (int j = 0; j< this->myH; j++) {
		for (int i = 0; i< this->myW; i++) {
			if (j > 0) add_edge(ii, ii - this->myW, this->myM); // N		
			if (j < (this->myH - 1)) add_edge(ii, ii + this->myW, this->myM); // S		
			if (i > 0) add_edge(ii, ii - 1, this->myM); // W		
			if (i < (this->myW - 1)) add_edge(ii, ii + 1, this->myM); // E
			// increment
			ii = ii + 1;
		}
	}

	// Now put weights on edges
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(this->myM);
	for (eit = estart; eit != eend; eit++) {
		MamaVId id1 = source(*eit, this->myM);
		MamaVId id2 = target(*eit, this->myM);
		// get pixel values 
		float val1 = img.at<float>(this->myM[id1].y, this->myM[id1].x);
		float val2 = img.at<float>(this->myM[id2].y, this->myM[id2].x);
		// average for edge weight
		float vala = (val1 + val2) / 2.0f;
		this->myM[*eit].weight = (double)vala;
	}

	LOG_TRACE(m_logger, "  Image graph has " << num_vertices(this->myM) << " vertices and " << num_edges(this->myM) << " edges, and " << this->myMarkers.size() << " markers");
	LOG_TRACE(m_logger, "InitGraph Doone");
}


void SegmentWS::ProcessMarkerGraph(SegmentParameter &param)
{
	LOG_TRACE(m_logger, "WatershedMST");

	// clear second layer graph
	this->myM2.clear();

	// sort the edges
	vector< pair< double, MamaEId > > sortedEdges;
	sortedEdges.clear();
	sortedEdges.resize(num_edges(this->myM));
	MamaEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(this->myM);
	int ei = 0;
	for (eit = estart; eit != eend; eit++) {
		pair< double, MamaEId > eid(this->myM[*eit].weight, *eit);
		sortedEdges[ei] = eid;
		ei++;
	}
	// flood fill from the bottom 
	sort(sortedEdges.begin(), sortedEdges.end(), std::less< pair<double, MamaEId> >());

	// make sure we are starting with an empty forrest
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	for (nit = nstart; nit != nend; nit++) {
		this->myM[*nit].label = -1;
	}

	this->mySortedMergeDepth.clear();
	int labelCount = 0;

	// run kruskalish algorithm
	for (int i = 0; i < sortedEdges.size(); i++) {

		MamaEId eid = sortedEdges[i].second;
		MamaVId id1 = source(eid, this->myM);
		MamaVId id2 = target(eid, this->myM);

		if ((this->myM[id1].label < 0) && (this->myM[id2].label < 0)) {		// new basin
			this->myM[id1].label = labelCount;
			this->myM[id2].label = labelCount;
			MamaVId nid = add_vertex(this->myM2);
			if (labelCount != (int)nid) BOOST_THROW_EXCEPTION(Unexpected());
			// track the minimums to calculate depth
			this->myM2[nid].weight = this->myM[eid].weight;
			this->myM2[nid].label = labelCount;
			this->myM2[nid].ownedBy = 0;
			labelCount++;
		}
		else if (this->myM[id1].label < 0) {		// extend basin
			this->myM[id1].label = this->myM[id2].label;
		}
		else if (this->myM[id2].label < 0) {		// extend basin
			this->myM[id2].label = this->myM[id1].label;
		}
		else {
			if (this->myM[id1].label != this->myM[id2].label) {
				// see if we have seen this pair before
				MamaEId mid; bool newEdge;
				std::tie(mid, newEdge) = add_edge(this->myM[id1].label, this->myM[id2].label, this->myM2);
				if (newEdge) {
					MamaVId mid1 = source(mid, this->myM2);
					MamaVId mid2 = target(mid, this->myM2);
					double depth;
					if (param.waterfall != 1) {
						double nMin = (this->myM2[mid1].weight < this->myM2[mid2].weight) ? this->myM2[mid1].weight : this->myM2[mid2].weight;
						depth = this->myM[eid].weight - nMin;
					}
					else {
						depth = this->myM[eid].weight;
					}
					// presort for threshold
					this->myM2[mid].weight = depth;										
					pair<double, MamaEId> sp(depth, mid);
					this->mySortedMergeDepth.push_back(sp);
				}
			}
		}
	}
	// sort for threshold
	sort(this->mySortedMergeDepth.begin(), this->mySortedMergeDepth.end(), std::less< pair<double, MamaEId> >());

	this->myThresholdIndex = -1;

	LOG_TRACE(m_logger, "  Watershed found " << num_vertices(this->myM2) << " basins and " << num_edges(this->myM2) << " potential merges");
	LOG_TRACE(m_logger, "WatershedMST Done");
}


void SegmentWS::PropagateMarkers(SegmentParameter &param)
{
	LOG_TRACE(m_logger, "WatershedMST");

	// initilize basins with markers
	map<int, int> mcount, lcount;
	mcount.clear(); lcount.clear();

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	for (nit = nstart; nit != nend; nit++) {
		MamaVId nid = this->myM[*nit].label;
		if (this->myM[*nit].ownedBy > 0) {
			this->myM2[nid].ownedBy = this->myM[*nit].ownedBy;
			mcount[nid] = 1;
			lcount[this->myM[*nit].ownedBy] = 1;
		}
	}
	LOG_INFO(m_logger, "There are " << lcount.size() << " covering " << mcount.size() << " segments"); 
	// run kruskalish algorithm on second layer
	LOG_INFO(m_logger, "Second layer has edges weights from " << this->mySortedMergeDepth[0].first << " -> " << this->mySortedMergeDepth[this->mySortedMergeDepth.size() - 1].first); 
	
	for (int i = 0; i < this->mySortedMergeDepth.size(); i++) {
		double thresh = this->mySortedMergeDepth[i].first;
		MamaEId eid = this->mySortedMergeDepth[i].second;
		MamaVId id1 = source(eid, this->myM);
		MamaVId id2 = target(eid, this->myM);
		
		if ((this->myM2[id1].ownedBy > 0) && (this->myM2[id2].ownedBy <= 0)) {						
				RecurseOwnership(id2, this->myM2[id1].ownedBy, thresh);
		}
		else if ((this->myM2[id1].ownedBy <= 0) && (this->myM2[id2].ownedBy > 0)) {
				RecurseOwnership(id1, this->myM2[id2].ownedBy, thresh);
		}
	}

	LOG_TRACE(m_logger, "Watershed Marker Done");
}

void SegmentWS::RecurseOwnership(MamaVId starti, int owned, double thresh)
{
	vector<MamaVId> newguy;
	newguy.clear();
	newguy.push_back(starti);

	int ci = 0;

	while (newguy.size() > 0) {
		MamaVId nextId = newguy[newguy.size() - 1];
		newguy.pop_back();
		this->myM2[nextId].ownedBy = owned;
		ci++;
		// get neighbors
		MamaNeighborIt nit, nstart, nend;
		std::tie(nstart, nend) = adjacent_vertices(nextId, this->myM2);
		for (nit = nstart; nit != nend; nit++) {
			std::pair < MamaEId, bool > peid = edge(nextId, *nit, this->myM2);
			if (!(peid.second)) BOOST_THROW_EXCEPTION(Unexpected());
			if (this->myM2[peid.first].weight < (thresh + 1.0e-6)) {
				if (this->myM2[*nit].ownedBy <= 0) {
					newguy.push_back(*nit);
				}
			}
		}
	}
}

void SegmentWS::Clear()
{
	this->myM.clear();
	this->myM2.clear();
	this->mySortedMergeDepth.clear();
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
SegmentWS::SegmentWS() : m_logger(LOG_GET_LOGGER("SegmentWS"))
{
}

SegmentWS::~SegmentWS()
{
}

