#include "SegmentWS.h"
#include "MamaException.h"
#include "MamaDef.h"

#include "opencv2/opencv.hpp"
using namespace cv;

int SegmentWS::LabelVertices(MamaVId id, int label, double localThresh)
{
	// probably a more efficient way to do this 
	map<MamaVId, int> component; component.clear();
	vector<MamaVId> newguy; newguy.clear();
	newguy.push_back(id);
	component[id] = 1;

	while (newguy.size() > 0) {
		MamaVId nextId = newguy[newguy.size() - 1];
		newguy.pop_back();
		// get neighbor edges
		MamaNeighborEdgeIt neit, nestart, neend;
		std::tie(nestart, neend) = out_edges(nextId, this->myM2);
		for (neit = nestart; neit != neend; neit++) {
			if (this->myM2[*neit].weight < localThresh) {
				MamaVId myNext;
				if (source(*neit, this->myM2) != nextId) myNext = source(*neit, this->myM2);
				else myNext = target(*neit, this->myM2);

				if (component.count(myNext) == 0) {
					newguy.push_back(myNext);
					component[myNext] = 1;
				}
			}
		}
	}

	for (map<MamaVId, int>::iterator it = component.begin(); it != component.end(); ++it) {
		this->myM2[it->first].label = label;
	}
	return(component.size());
}


void SegmentWS::Init(cv::Mat &img, MamaSegParam &param)
{
	LOG_TRACE(m_logger, "Init");

	cv::Mat segIn;
	// edge detection
	this->Preprocess(img, segIn, param);
	// create graph
	this->InitGraph(segIn);	
	// prep data structures with two layer segmentation 
	this->ProcessMST(param);

	LOG_TRACE(m_logger, "Init Done");
}

void SegmentWS::GetLabels(cv::Mat &segf)
{
	segf = Mat::zeros(this->myH, this->myW, CV_32F);

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);

	if (this->myThresholdIndex < 0) {	// this happens when UpdateThreshold is not called. 
		for (nit = nstart; nit != nend; nit++) {
			segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float)(this->myM[*nit].basin + 1);
		}
	}
	else {
		for (nit = nstart; nit != nend; nit++) {
			segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float)(this->myM[*nit].label + 1);
		}
	}
}

void SegmentWS::GetLabelsWithStats(cv::Mat &segf, int &segCount, int &segMin, int &segMax)
{
	segf = Mat::zeros(this->myH, this->myW, CV_32F);

	if (this->myThresholdIndex < 0) BOOST_THROW_EXCEPTION(Unexpected("You must call UpdateThreshold first")); 

	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	map< int, int> labelCount;
	labelCount.clear();
	segMin = LARGEST_INT;
	segMax = SMALLEST_INT;
	for (nit = nstart; nit != nend; nit++) {
		int ilabel = this->myM[*nit].label + 1;
		if (ilabel > segMax) segMax = ilabel;
		if (ilabel < segMin) segMin = ilabel;
		if (labelCount.count(ilabel) == 0) labelCount[ilabel] = 1;
		else labelCount[ilabel] = labelCount[ilabel] + 1;
		segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float) ilabel;
	}	
	segCount = labelCount.size();
}

void SegmentWS::UpdateThreshold(MamaSegParam &param)
{
	LOG_TRACE(m_logger, "UpdateThreshold");
	
	if (this->mySortedMergeDepth.size() < 1) BOOST_THROW_EXCEPTION(UnexpectedSize("There is no merge graph to threshold"));
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
		if (this->myThresholdIndex < 0) this->myThresholdIndex  = 0;
	}
	
	LOG_TRACE(m_logger, "  Threshold at " << this->myThresholdIndex << " from " << this->mySortedMergeDepth.size());
	// reset basin labels
	for (int i = 0; i < this->myBasinMins.size(); i++) {
		this->myLabelTree[i] = i;
	}
	cout << "Add merge\n";
	// add merges
	int mergeLabel = this->myBasinMins.size(); 
	for (int i = 0; i < this->myThresholdIndex; i++) {
		// get the pair index
		int pairi = this->mySortedMergeDepth[i].second;
		// get the current labels
		int l1 = this->GetFinalLabel( this->myMergePair[pairi].first );
		int l2 = this->GetFinalLabel( this->myMergePair[pairi].second );
		// update label tree
		this->myLabelTree[mergeLabel] = mergeLabel;
		this->myLabelTree[l1] = mergeLabel;
		this->myLabelTree[l2] = mergeLabel;
		mergeLabel++;
	}
	cout << "Relabel\n";
	// now relabel
	MamaNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	for (nit = nstart; nit != nend; nit++) {
		this->myM[*nit].label = this->GetFinalLabel(this->myM[*nit].basin); 
	}	
	cout << "Yah\n";
	LOG_TRACE(m_logger, "UpdateThreshold Done");
}

int SegmentWS::GetFinalLabel(int lin)
{
	while (this->myLabelTree[lin] != lin) {
		lin = this->myLabelTree[lin];
	}
	return(lin);
}

void SegmentWS::ProcessMST(MamaSegParam &param)
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
		this->myM[*nit].basin = -1;
	}

	// run kruskal 
	this->myBasinMins.clear();	
	this->myMergePair.clear();
	this->mySeenMergeBefore.clear();
	this->mySortedMergeDepth.clear();
	this->myLabelTree.clear();

	int labelCount = 0;
	
	for (int i = 0; i < sortedEdges.size(); i++) {

		MamaEId eid = sortedEdges[i].second;
		MamaVId id1 = source(eid, this->myM);
		MamaVId id2 = target(eid, this->myM);

		if ((this->myM[id1].basin < 0) && (this->myM[id2].basin < 0)) {		// new basin
			this->myM[id1].basin = labelCount;
			this->myM[id2].basin = labelCount;
			MamaVId nid = add_vertex(this->myM2);
			if (labelCount != (int)nid) BOOST_THROW_EXCEPTION(Unexpected());
			// track the minimums to calculate depth
			this->myM2[nid].weight = this->myM[eid].weight;
			//this->myBasinMins.push_back(this->myM[eid].weight);
			labelCount++;
		}
		else if (this->myM[id1].basin < 0) {		// extend basin
			this->myM[id1].basin = this->myM[id2].basin;
		}
		else if (this->myM[id2].basin < 0) {		// extend basin
			this->myM[id2].basin = this->myM[id1].basin;
		}
		else {
			// check to see if we have seen this pair before
			if (this->myM[id1].basin != this->myM[id2].basin) {
				MamaEId eid; bool newEdge;
				std::tie(eid, newEdge) = add_edge(this->myM[id1].basin, this->myM[id2].basin, this->myM2);
				if (newEdge) {}
				pair<int, int> mp = OrderedKey(this->myM[id1].basin, this->myM[id2].basin);
				if (this->mySeenMergeBefore.count(mp) == 0) {
					// this edge is the saddle
					this->mySeenMergeBefore[mp] = this->myMergePair.size();
					// calculate depth 
					double nMin = (this->myBasinMins[this->myM[id1].basin] < this->myBasinMins[this->myM[id2].basin]) ?
						this->myBasinMins[this->myM[id1].basin] : this->myBasinMins[this->myM[id2].basin];
					double depth = this->myM[eid].weight - nMin;
					// presort for threshold
					pair<double, int> sp(depth, this->myMergePair.size());
					this->mySortedMergeDepth.push_back(sp);
					this->myMergePair.push_back(mp);
				}
			}
		}
	}
	// sort for threshold
	sort(this->mySortedMergeDepth.begin(), this->mySortedMergeDepth.end(), std::less< pair<double, int> >());	
	this->myThresholdIndex = -1;
	// allocate label tree 
	this->myLabelTree.resize( this->myBasinMins.size() + this->myMergePair.size() );
	

	LOG_TRACE(m_logger, "  Watershed found " << this->myBasinMins.size() << " basins and " << this->myMergePair.size() << " potential merges"); 
	LOG_TRACE(m_logger, "WatershedMST Done");	
}


void SegmentWS::Preprocess(cv::Mat &img, cv::Mat &out, MamaSegParam &param)
{
	LOG_TRACE(m_logger, "Preprocess");

	Mat temp, tempTemp;
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Size origsz = Size(img.cols, img.rows);
	// Gradient X, Y
	Scharr(img, grad_x, CV_32F, 1, 0); //, scale, delta, BORDER_DEFAULT );
	Scharr(img, grad_y, CV_32F, 0, 1); //, scale, delta, BORDER_DEFAULT );			
	// Magnitude
	cartToPolar(grad_x, grad_y, temp, tempTemp); //Mat()); //temp2); 
	// Smooth		
	GaussianBlur(temp, out, Size(3, 3), 0.0);

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

void SegmentWS::Clear()
{
	this->myM.clear();
	this->myBasinMins.clear();	
	this->mySortedMergeDepth.clear();
	this->myMergePair.clear();	
	this->mySeenMergeBefore.clear();
	this->myLabelTree.clear();
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

