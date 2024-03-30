#include "SegmentWSMarker.h"
#include "MamaException.h"
#include "MamaDef.h"
#include "Morphology.h"
#include "VizMat.h"

#include "opencv2/opencv.hpp"
using namespace cv;


void SegmentWSMarker::MarkerSegmentation(cv::Mat &img, cv::Mat &mark, SegmentParameter &param)
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
	// set this tpo indicate complete
	this->myThresholdIndex = 1;
	LOG_TRACE(m_logger, "MarkerSegmentation Done");
}

void SegmentWSMarker::InitMarkerGraph(cv::Mat &img, cv::Mat &mark)
{
	LOG_TRACE(m_logger, "InitGraph");
	//	LOG_INFO(m_logger, "Creating graph structure\n");

	this->myM.clear();

	this->myW = img.cols;
	this->myH = img.rows;
	int N = this->myW * this->myH;
	this->m_pixelMarkers.clear(); 
	this->m_markers.clear();
	this->m_maxMarkers = -1;
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
			this->m_pixelMarkers[nid] = marki;			
			this->m_markers[marki] = 1;
			if (marki > this->m_maxMarkers) this->m_maxMarkers = marki;

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

	LOG_TRACE(m_logger, "  Image graph has " << num_vertices(this->myM) << " vertices and " << num_edges(this->myM) << " edges, and " << this->m_markers.size() << " markers");
	LOG_TRACE(m_logger, "InitGraph Doone");
}


void SegmentWSMarker::ProcessMarkerGraph(SegmentParameter &param)
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
			this->myM2[nid].label = 0; 			
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


void SegmentWSMarker::PropagateMarkers(SegmentParameter &param)
{
	LOG_TRACE(m_logger, "WatershedMST");

	// initilize basins with markers
	//map<int, int> mcount, lcount;
	//mcount.clear(); lcount.clear();

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	for (nit = nstart; nit != nend; nit++) {
		MamaVId nid = this->myM[*nit].label;

		if (this->m_pixelMarkers[*nit] > 0) {
			this->myM2[nid].label = this->m_pixelMarkers[*nit];
			//mcount[nid] = 1;
			//lcount[this->myM[*nit].ownedBy] = 1;
		}
	}
	//LOG_INFO(m_logger, "There are " << lcount.size() << " covering " << mcount.size() << " segments");
	// run kruskalish algorithm on second layer
	LOG_INFO(m_logger, "Second layer has edges weights from " << this->mySortedMergeDepth[0].first << " -> " << this->mySortedMergeDepth[this->mySortedMergeDepth.size() - 1].first);

	for (int i = 0; i < this->mySortedMergeDepth.size(); i++) {
		double thresh = this->mySortedMergeDepth[i].first;
		MamaEId eid = this->mySortedMergeDepth[i].second;
		MamaVId id1 = source(eid, this->myM);
		MamaVId id2 = target(eid, this->myM);

		if ((this->myM2[id1].label > 0) && (this->myM2[id2].label <= 0)) {
			RecurseOwnership(id2, this->myM2[id1].label, thresh);
		}
		else if ((this->myM2[id1].label <= 0) && (this->myM2[id2].label > 0)) {
			RecurseOwnership(id1, this->myM2[id2].label, thresh);
		}
	}

	LOG_TRACE(m_logger, "Watershed Marker Done");
}

void SegmentWSMarker::RecurseOwnership(MamaVId starti, int owned, double thresh)
{
	vector<MamaVId> newguy;
	newguy.clear();
	newguy.push_back(starti);

	int ci = 0;

	while (newguy.size() > 0) {
		MamaVId nextId = newguy[newguy.size() - 1];
		newguy.pop_back();
		this->myM2[nextId].label = owned;
		ci++;
		// get neighbors
		MamaNeighborIt nit, nstart, nend;
		std::tie(nstart, nend) = adjacent_vertices(nextId, this->myM2);
		for (nit = nstart; nit != nend; nit++) {
			std::pair < MamaEId, bool > peid = edge(nextId, *nit, this->myM2);
			if (!(peid.second)) BOOST_THROW_EXCEPTION(Unexpected());
			if (this->myM2[peid.first].weight < (thresh + 1.0e-6)) {
				if (this->myM2[*nit].label <= 0) {
					newguy.push_back(*nit);
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
SegmentWSMarker::SegmentWSMarker() 
{
}

SegmentWSMarker::~SegmentWSMarker()
{
}

