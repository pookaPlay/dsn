// OhmHungarian.h: Implementation of the OhmHungarian class.
//
//////////////////////////////////////////////////////////////////////
#include "OhmHungarian.h"
#include "Hungarian.h"

#include "opencv2/opencv.hpp"
using namespace cv;

// We need this in any file we want to use libBasil::Info
static Info info;

#define MAX_DISTANCE	100.0f
#define MAX_COAST_TIME	1
#define MIN_STARTUP_TIME 2

void OhmHungarian::Execute(Mat &targets, Mat &unassigned) 
{		
	/*
	doNotAssignCost = 1.0f;
	maxCoastTime = MAX_COAST_TIME;
	int minStartupTime = MIN_STARTUP_TIME;
	maxDistance = MAX_DISTANCE;
	float initThreshold = 0.005f;

	unsigned int numObs = targets.rows;
	unsigned int ntf = tracks.NumActiveTracks();
	unsigned int ntt = numObs+ntf;		

	if (ntf == 0) {	
		unassigned = targets.clone();
		return;	// leave in first observation type. 
	}
	//info(10, "Associate:  Associating %i tracks with %i observations\n", ntf, ntt); 
		
	Mat costMatrix;
	costMatrix = Mat::zeros(ntt, ntf, CV_64F); 
	PopulateCostMatrix(costMatrix, targets, tracks);

//*****************************************************************
//***************** FIND ASSIGNMENT 
//******************************************************************

	int *assignment;
	try {
		assignment = new int[ntf];
	} catch(...) {
		BOOST_THROW_EXCEPTION(MemoryProblem()); 
	}
	double cost;	
	Hungarian::OptimalHungarian(assignment, &cost, (double *) costMatrix.ptr(), ntf, ntt);

//*****************************************************************
//***************** COPY OBSERVATIONS 
//******************************************************************
	unsigned int assignedCount = 0;
	vector<bool> assigned;
	assigned.clear();
	assigned.resize(numObs, false);

	tracks.ResetActive();
	for(unsigned int from=0; from<ntf; from++) {
		Track *fromTrack = tracks.GetActiveTrack();
		if (assignment[from] >= (int) numObs) {
			fromTrack->assigned[0] = false;				
		} else {
			fromTrack->assigned[0] = true;
			fromTrack->coastTime = 0;
			fromTrack->xobs[0] = targets.at<float>(assignment[from], 0); 
			fromTrack->yobs[0] = targets.at<float>(assignment[from], 1); 
			assigned[ assignment[from] ] = true;
			assignedCount++;
		}
	}
	
	delete [] assignment;
	
//*****************************************************************
//***************** OUTPUT UNASSIGNED OBSERVATIONS 
//******************************************************************
	// Output unassigned observations
	unsigned int unassignedCount = numObs-assignedCount;
	//info(10, "Associate: %i observations unassigned\n", unassignedCount);
	unassigned = Mat::zeros(unassignedCount, 2, CV_32F);

	unsigned int upto =0;
	for(unsigned int j=0; j< numObs; j++) {
		if (!assigned[j]) {
			// here we only initiate on strong movers
			targets.row(j).copyTo( unassigned.row(upto) ); 
			upto++;
		}
	}

//*****************************************************************
//***************** REMOVE UNASSIGNED 
//******************************************************************
	tracks.ResetActive();
	while(tracks.MoreActiveTracks()) {
		Track *fromTrack = tracks.GetActiveTrack();
		if (!(fromTrack->assigned[0])) {
			if (fromTrack->startupTime < minStartupTime) tracks.DeleteActiveTrack(fromTrack->id);			
			else {
				fromTrack->coastTime++;
				if (fromTrack->coastTime > maxCoastTime) tracks.MoveFromActiveToInactive(fromTrack);
			}
		}
		fromTrack->startupTime++;
	}
	*/
	return;
}

/*****************************************************************
*********  TARGET / OBSERVATION ASSOCIATION WEIGHT 
******************************************************************/

void OhmHungarian::PopulateCostMatrix(Mat &costMatrix, Mat &targets)
{
	/*
	unsigned int ntf, ntt, numObs;
	float maxAppearance, minAppearance, appearanceCost, locationCost;

	ntf = costMatrix.cols;
	ntt = costMatrix.rows; 
	numObs = ntt - ntf;
	maxAppearance = SMALLEST_FLOAT;
	minAppearance = LARGEST_FLOAT;

	
	// Now calculate location cost and combine
	tracks.ResetActive();
	for(unsigned int from=0; from<ntf; from++) {
		Track *fromTrack = tracks.GetActiveTrack();
		for(unsigned int to=0; to<numObs; to++) {		
			float xloc = targets.at<float>(to, 0); 
			float yloc = targets.at<float>(to, 1); 
			locationCost = fromTrack->ComparePredictedPosition(xloc, yloc);	
			if (locationCost > maxDistance) locationCost = LARGEST_FLOAT;
			else locationCost = locationCost / maxDistance;
			costMatrix.at<double>(to, from) = locationCost;
		}
	}
	// The "do not assign" cost
	for(unsigned int to=numObs; to<ntt; to++) {
		for(unsigned int from=0; from<ntf; from++) {
			costMatrix.at<double>(to, from) = (double) doNotAssignCost;
		}
	}
	*/
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

OhmHungarian::OhmHungarian() 
{
}
OhmHungarian::~OhmHungarian()
{
}
