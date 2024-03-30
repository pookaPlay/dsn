///////////////////////////////////////////////////////

#if !defined(OhmHungarian_H__)
#define OhmHungarian_H__

#include "Info.h"
#include "MamaException.h"

#include "opencv2/opencv.hpp"

using namespace std;
/**
 * OhmHungarian should be copied and extended
 */

class OhmHungarian
{
  public:	
	bool haveBoundaries;
	bool useAppearance;
	float synAppearance; 	
	float obsID;
	unsigned int numObsTypes, maxCoastTime;
	float maxDistance;
	float doNotAssignCost;		

	void Execute(cv::Mat &targets, cv::Mat &unassigned);	
	void PopulateCostMatrix(cv::Mat &costMatrix, cv::Mat &targets );	

	OhmHungarian();
    virtual ~OhmHungarian();
};

#endif 
