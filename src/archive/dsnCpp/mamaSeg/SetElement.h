#if !defined(SetElement_H__)
#define SetElement_H__

#include <vector>
using namespace std;

#include "Logger.h"

/** 
* Container used in disjoint set datastructure
* for efficiently computing optimal thresholds in 
* the segmentation hierachy 
*/
class SetElement 
{
public:
    int nodeID;
    map<int, double> labelCount;
    	
    SetElement(int nodeID, int label);
    SetElement(int myNodeID, map<int, double> &myLabel);
	
    double GetNumberOfItems();
    void AddLabelCounts(map<int, double> a, map<int, double> b);
    static double DotProductLabels(map<int, double> a, map<int, double>b);
	
};


#endif 

