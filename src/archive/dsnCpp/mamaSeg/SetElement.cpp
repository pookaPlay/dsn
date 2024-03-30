#include "SetElement.h"


#include "opencv2/opencv.hpp"
using namespace cv;
#include <boost/pending/disjoint_sets.hpp>
#include <boost/pending/property.hpp>

void SetElement::AddLabelCounts(map<int, double> a, map<int, double> b) {
    // Copy map a, then add in elements from b
    labelCount = a;
    map<int, double>::iterator it;
    for (it=b.begin(); it!=b.end(); ++it) {
        if (labelCount.count(it->first)) {
            labelCount[it->first] += it->second;
        } else {
            labelCount[it->first] = it->second;
        }
    }
}

double SetElement::GetNumberOfItems() {
    double sum = 0.0;
    map<int, double>::iterator it;
    for (it = labelCount.begin(); it!=labelCount.end(); ++it) {
        sum += it->second;
    }
    return sum;
}

double SetElement::DotProductLabels(map<int, double> a, map<int, double> b) {
    double sum = 0.0;
    // multiply matched entries in a and b and add them
    map<int, double>::iterator it_a;
    for (it_a = a.begin(); it_a!=a.end(); ++it_a) {
        if (b.count(it_a->first)) {
            sum += it_a->second * b[it_a->first];
        }
    }
    return sum;
}

SetElement::SetElement(int myNodeID, int myLabel) {
	labelCount.clear();
	nodeID = myNodeID;
	labelCount[myLabel] = 1.0;
}

SetElement::SetElement(int myNodeID, map<int, double> &myLabel) {
	labelCount.clear();
	nodeID = myNodeID;
	labelCount = myLabel;
}
