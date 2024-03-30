#include "SegmentACD.h"
#include "MamaException.h"
#include "MamaDef.h"
#include "Morphology.h"
#include "VizMat.h"
#include "KMeanFeatures.h"

#include "opencv2/opencv.hpp"
using namespace cv;


void SegmentACD::Init(cv::Mat &img, SegmentParameter &param)
{
	LOG_TRACE(m_logger, "Init");
	
	// create features
	vector<Mat> imgs;
	imgs.clear();
	imgs.push_back(img);


	// Do some preprocessing to populate edge "features"	
	int winSize = 1;
	int numFeatures = 8;
	int useWhite = 0;
	string modelName = "knnstuff";
	
	//KMeanFeatures knn;
	//knn.LearnFeatures(img, numFeatures, winSize, useWhite);
	//knn.SaveModel(modelName);
	//knn.LoadModel(modelName);
	//knn.GenerateFeatures(img, imgs, "triMap", winSize);


	//Mat temp1, temp2, temp3; 
	//GaussianBlur(img, temp1, Size(3, 3), 0.0);
	//GaussianBlur(img, temp2, Size(5, 5), 0.0);
	//GaussianBlur(img, temp3, Size(7, 7), 0.0);
	//imgs.push_back(temp1);
	//imgs.push_back(temp2);
	//imgs.push_back(temp3);

	this->InitGraph(imgs);	
	
	this->ACDInitEdge();
	this->WSCInitEdge(); 

	// prep data structures with two layer segmentation 
	//this->ProcessMST();

	LOG_TRACE(m_logger, "Init Done");
}


void SegmentACD::ACDInitEdge()
{
	LOG_INFO(m_logger, "ACD Edge");	

	ACDNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);

	int D = this->myM[*nstart].val.size(); 
	int N = num_edges(this->myM);

	Mat x = Mat::zeros(N, D, CV_64F);
	Mat y = Mat::zeros(N, D, CV_64F);
	Mat result; 

	ACDEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(this->myM);
	int ei = 0;
	for (eit = estart; eit != eend; eit++) {
		ACDVId n1 = source(*eit, this->myM);
		ACDVId n2 = target(*eit, this->myM);
		
		for (int di = 0; di < this->myM[n1].val.size(); di++) {
			x.at<double>(ei, di) = this->myM[n1].val[di];
		}
		for (int di = 0; di < this->myM[n2].val.size(); di++) {
			y.at<double>(ei, di) = this->myM[n2].val[di];
		}

		ei++;
	}

	mySortedBaseEdge.clear();
	this->myACD.Train(x, y); 
	this->myACD.Apply(x, y, result);
	ei = 0;
	for (eit = estart; eit != eend; eit++) {
		this->myM[*eit].weight = result.at<double>(ei);
		pair<double, ACDEId> sp(result.at<double>(ei), *eit);
		this->mySortedBaseEdge.push_back(sp);
		ei++;
	}
	// sort for threshold
	sort(this->mySortedBaseEdge.begin(), this->mySortedBaseEdge.end(), std::less< pair<double, ACDEId> >());

	LOG_INFO(m_logger, "ACD Edge Done");
}


void SegmentACD::UpdateBaseThreshold(SegmentParameter &param)
{
	LOG_TRACE(m_logger, "UpdateBaseThreshold");

	if (this->mySortedBaseEdge.size() < 1) BOOST_THROW_EXCEPTION(UnexpectedSize("There is no graph to threshold"));

	double localThresh;

	if (!param.absoluteThreshold) {
		// Threshold gets converted from an absolute [0..1] range to an index into the sorted list	
		if (param.threshold <= 0.0) {
			this->myThresholdIndex = 0;
		}
		else if (param.threshold >= 1.0) {
			this->myThresholdIndex = this->mySortedBaseEdge.size() - 1;
		}
		else {
			this->myThresholdIndex = (int)floor(param.threshold * (double)(this->mySortedBaseEdge.size()));
			// and check bounds.. just to be safe
			if (this->myThresholdIndex > this->mySortedBaseEdge.size() - 1) this->myThresholdIndex = this->mySortedBaseEdge.size() - 1;
			if (this->myThresholdIndex < 0) this->myThresholdIndex = 0;
		}

		localThresh = this->mySortedBaseEdge[this->myThresholdIndex].first + FEATURE_TOLERANCE;
		LOG_INFO(m_logger, "Local thresh is " << localThresh << " from " << this->mySortedBaseEdge[0].first << " -> " << this->mySortedBaseEdge[this->mySortedBaseEdge.size() - 1].first << "\n");
	}
	else {
		localThresh = param.threshold;
		this->myThresholdIndex = 0;
		LOG_TRACE(m_logger, "Local thresh is absolute " << localThresh);
	}

	int segCount = 0;
	ACDNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);

	for (nit = nstart; nit != nend; nit++) {
		this->myM[*nit].label = -1;
	}

	for (nit = nstart; nit != nend; nit++) {
		if (this->myM[*nit].label < 0) {
			int compSize = this->LabelBaseVertices(*nit, segCount, localThresh);
			segCount++;
		}
	}
	LOG_TRACE(m_logger, "UpdateThreshold Done");
}

int SegmentACD::LabelBaseVertices(ACDVId id, int label, double localThresh)
{
	vector<ACDVId> newguy;
	newguy.clear();
	newguy.push_back(id);

	int ci = 0;

	while (newguy.size() > 0) {
		ACDVId nextId = newguy[newguy.size() - 1];
		newguy.pop_back();
		this->myM[nextId].label = label;
		ci++;
		// get neighbors
		ACDNeighborIt nit, nstart, nend;
		std::tie(nstart, nend) = adjacent_vertices(nextId, this->myM);
		for (nit = nstart; nit != nend; nit++) {
			if (this->myM[*nit].label < 0) {
				std::pair < ACDEId, bool > peid = edge(nextId, *nit, this->myM);
				if (!(peid.second)) BOOST_THROW_EXCEPTION(Unexpected());
				if (this->myM[peid.first].weight < localThresh) {
					newguy.push_back(*nit);
				}
			}
		}
	}

	return(ci);
}


void SegmentACD::WSCInitEdge()
{
	LOG_TRACE(m_logger, "WatershedCuts");
	
	mySortedBaseEdge.clear();

	ACDEdgeIt eit, estart, eend;
	ACDNeighborEdgeIt neit, nestart, neend;
	int ii = 0;
	std::tie(estart, eend) = edges(this->myM);
	for (eit = estart; eit != eend; eit++) {
		ACDVId id1 = source(*eit, this->myM);
		ACDVId id2 = target(*eit, this->myM);

		double allReal1 = LARGEST_DOUBLE;
		double allReal2 = LARGEST_DOUBLE;
		ACDEId wIndex1, wIndex2;

		if (out_degree(id1, this->myM) > 1) {
			std::tie(nestart, neend) = out_edges(id1, this->myM);
			for (neit = nestart; neit != neend; neit++) {
				if (*neit != *eit) {
					if (this->myM[*neit].weight < allReal1) {
						allReal1 = this->myM[*neit].weight;
						wIndex1 = (ACDEId)(*neit);
					}
				}
			}
		}
		else allReal1 = SMALLEST_DOUBLE;

		if (out_degree(id2, this->myM) > 1) {
			std::tie(nestart, neend) = out_edges(id2, this->myM);
			for (neit = nestart; neit != neend; neit++) {
				if (*neit != *eit) {
					if (this->myM[*neit].weight < allReal2) {
						allReal2 = this->myM[*neit].weight;
						wIndex2 = (ACDEId)(*neit);
					}
				}
			}
		}
		else allReal2 = SMALLEST_DOUBLE;

		double allReal;
		ACDEId wIndex;
		if (allReal1 > allReal2) {
			allReal = allReal1;
			wIndex = wIndex1;
		}
		else {
			allReal = allReal2;
			wIndex = wIndex2;
		}


		double threshVal = this->myM[*eit].weight - allReal;

		//this->myM[*eit].wIndex = wIndex;
		this->myM[*eit].wasWeight = threshVal;
		pair<double, ACDEId> sp(threshVal, *eit);
		this->mySortedBaseEdge.push_back(sp);		
	}
	// sort for threshold
	sort(this->mySortedBaseEdge.begin(), this->mySortedBaseEdge.end(), std::less< pair<double, ACDEId> >());

	for (eit = estart; eit != eend; eit++) {
		double newWeight = this->myM[*eit].wasWeight; 
		this->myM[*eit].wasWeight = this->myM[*eit].weight;
		this->myM[*eit].weight = newWeight;
	}

	LOG_TRACE(m_logger, "WatershedCuts Done");
}



void SegmentACD::GetBasinLabels(cv::Mat &segf)
{
	segf = Mat::zeros(this->myH, this->myW, CV_32F);

	ACDNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	map< int, int> labelCount;
	labelCount.clear();
	for (nit = nstart; nit != nend; nit++) {
		int li = this->myM[*nit].label;		
		if (labelCount.count(li) == 0) labelCount[li] = 1;
		else labelCount[li] = labelCount[li] + 1;
		segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float)(li + 1);
	}
	LOG_INFO(m_logger, "I have " << labelCount.size() << " segments\n");
}

void SegmentACD::GetLabels(cv::Mat &segf)
{
	LOG_TRACE(m_logger, "GetLabels");
	segf = Mat::zeros(this->myH, this->myW, CV_32F);
	
	if (this->myThresholdIndex < 0) BOOST_THROW_EXCEPTION(Unexpected("You must call UpdateThreshold first"));
	
	ACDNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);

	for (nit = nstart; nit != nend; nit++) {
		int li = this->myM[*nit].label;

		int li2 = this->myM2[li].label;
		segf.at<float>(this->myM[*nit].y, this->myM[*nit].x) = (float)(li2 + 1);
	}	
	LOG_TRACE(m_logger, "GetLabels Done");
}

void SegmentACD::GetLabelsWithStats(cv::Mat &segf, int &segCount, int &segMin, int &segMax)
{
	LOG_TRACE(m_logger, "GetLabelsWithStats");
	segf = Mat::zeros(this->myH, this->myW, CV_32F);
	
	if (this->myThresholdIndex < 0) BOOST_THROW_EXCEPTION(Unexpected("You must call UpdateThreshold first")); 

	ACDNodeIt nit, nstart, nend;
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

void SegmentACD::UpdateThreshold(SegmentParameter &param)
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
	ACDNodeIt nit, nstart, nend;
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

int SegmentACD::LabelVertices(ACDVId id, int label, double localThresh)
{
	vector<ACDVId> newguy; 
	newguy.clear();
	newguy.push_back(id);
	
	int ci = 0; 
	
	while (newguy.size() > 0) {
		ACDVId nextId = newguy[newguy.size() - 1];
		newguy.pop_back();
		this->myM2[nextId].label = label;
		ci++;
		// get neighbors
		ACDNeighborIt nit, nstart, nend;
		std::tie(nstart, nend) = adjacent_vertices(nextId, this->myM2);
		for (nit = nstart; nit != nend; nit++) {
			if (this->myM2[*nit].label < 0) {
				std::pair < ACDEId, bool > peid = edge(nextId, *nit, this->myM2);
				if (!(peid.second)) BOOST_THROW_EXCEPTION(Unexpected());							
				if (this->myM2[peid.first].weight < localThresh) {
					newguy.push_back(*nit);					
				}
			}
		}
	}

	return(ci);
}

void SegmentACD::ProcessMST()
{
	LOG_TRACE(m_logger, "WatershedMST");
	
	// clear second layer graph
	this->myM2.clear();
	
	// sort the edges
	vector< pair< double, ACDEId > > sortedEdges;
	sortedEdges.clear();
	sortedEdges.resize(num_edges(this->myM));
	ACDEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(this->myM);
	int ei = 0;
	for (eit = estart; eit != eend; eit++) {
		pair< double, ACDEId > eid(this->myM[*eit].weight, *eit);
		sortedEdges[ei] = eid;
		ei++;
	}
	// flood fill from the bottom 
	sort(sortedEdges.begin(), sortedEdges.end(), std::less< pair<double, ACDEId> >());

	// make sure we are starting with an empty forrest
	ACDNodeIt nit, nstart, nend;
	std::tie(nstart, nend) = vertices(this->myM);
	for (nit = nstart; nit != nend; nit++) {
		this->myM[*nit].label = -1;
	}

	this->mySortedMergeDepth.clear();
	int labelCount = 0;

	// run kruskalish algorithm
	for (int i = 0; i < sortedEdges.size(); i++) {

		ACDEId eid = sortedEdges[i].second;
		ACDVId id1 = source(eid, this->myM);
		ACDVId id2 = target(eid, this->myM);

		if ((this->myM[id1].label < 0) && (this->myM[id2].label < 0)) {		// new basin
			this->myM[id1].label = labelCount;
			this->myM[id2].label = labelCount;
			ACDVId nid = add_vertex(this->myM2);
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
				ACDEId mid; bool newEdge;
				std::tie(mid, newEdge) = add_edge(this->myM[id1].label, this->myM[id2].label, this->myM2);
				if (newEdge) {
					ACDVId mid1 = source(mid, this->myM2);
					ACDVId mid2 = target(mid, this->myM2);
					// calculate depth 
					double nMin = (this->myM2[mid1].weight < this->myM2[mid2].weight ) ? this->myM2[mid1].weight : this->myM2[mid2].weight;
					double depth = this->myM[eid].weight - nMin;
					this->myM2[mid].weight = depth;
					// presort for threshold
					pair<double, ACDEId> sp(depth, mid); 
					this->mySortedMergeDepth.push_back(sp);					
				}
			}
		}
	}
	// sort for threshold
	sort(this->mySortedMergeDepth.begin(), this->mySortedMergeDepth.end(), std::less< pair<double, ACDEId> >());	
	this->myThresholdIndex = -1;
	

	LOG_TRACE(m_logger, "  Watershed found " << num_vertices(this->myM2) << " basins and " << num_edges(this->myM2) << " potential merges");
	LOG_TRACE(m_logger, "WatershedMST Done");	
}

void SegmentACD::Preprocess(cv::Mat &img, cv::Mat &out, SegmentParameter &param)
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

void SegmentACD::InitGraph(cv::Mat &img)
{
	vector<cv::Mat> singleImage;
	singleImage.clear();
	singleImage.push_back(img); 
	this->InitGraph(singleImage);
}

void SegmentACD::InitGraph(vector<cv::Mat> &imgs)
{
	LOG_TRACE(m_logger, "InitGraph");
	//	LOG_INFO(m_logger, "Creating graph structure\n");
	if (imgs.size() < 1) BOOST_THROW_EXCEPTION(Unexpected()); 
	
	this->myM.clear();

	this->myW = imgs[0].cols;
	this->myH = imgs[0].rows;
	int N = this->myW * this->myH;
	int D = imgs.size(); 

	//Mat edgePicy = Mat::zeros( this->H, this->W, CV_32F);
	// Add vertex for each pixel
	int ti = 0;
	vector<int> nodeIds; nodeIds.resize(N);
	for (int j = 0; j< this->myH; j++) {
		for (int i = 0; i< this->myW; i++) {
			ACDVId nid = add_vertex(this->myM);
			if (ti != (int)nid) BOOST_THROW_EXCEPTION(Unexpected("Problem building boost graph"));
			this->myM[nid].x = i;
			this->myM[nid].y = j;
			this->myM[nid].val.clear();
			for (int di = 0; di < D; di++) {
				float val1 = imgs[di].at<float>(this->myM[nid].y, this->myM[nid].x);
				this->myM[nid].val.push_back(static_cast<double>(val1));
			}
						
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
	ACDEdgeIt eit, estart, eend;
	std::tie(estart, eend) = edges(this->myM);
	for (eit = estart; eit != eend; eit++) {
		ACDVId id1 = source(*eit, this->myM);
		ACDVId id2 = target(*eit, this->myM);
		// get pixel values 
		//float val1 = img.at<float>(this->myM[id1].y, this->myM[id1].x);
		//float val2 = img.at<float>(this->myM[id2].y, this->myM[id2].x);
		// average for edge weight
		//float vala = (val1 + val2) / 2.0f;
		//this->myM[*eit].weight = (double)vala;
	}

	LOG_TRACE(m_logger, "  Image graph has " << num_vertices(this->myM) << " vertices and " << num_edges(this->myM) << " edges");	
	LOG_TRACE(m_logger, "InitGraph Doone");
}

void SegmentACD::Clear()
{
	this->myM.clear();
	this->myM2.clear();
	this->mySortedMergeDepth.clear();
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
SegmentACD::SegmentACD() : m_logger(LOG_GET_LOGGER("SegmentACD"))
{
}

SegmentACD::~SegmentACD()
{
}


/*

void SegmentACD::ACDInitVertex()
{
ACDNodeIt nit, nstart, nend;
std::tie(nstart, nend) = vertices(this->myM);
for (nit = nstart; nit != nend; nit++) {
this->myM[*nit].val.clear();
double li = this->myM[*nit].weight;
this->myM[*nit].val.push_back(li);
}
}
*/