#if !defined(Hungarian_H__)
#define Hungarian_H__

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
using namespace std;
#include "Info.h"
#include "MamaException.h"
#include "MamaDef.h"

class Hungarian 
{
  public:
	static void SubOptimalHungarian(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns);
	static void OptimalHungarian(int *assignment, double *cost, double *distMatrix, int nOfRows, int nOfColumns);
	static void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns);
	static void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows);
	static void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	static void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	static void step3 (int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	static void step4 (int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
	static void step5 (int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);

    Hungarian();
    virtual ~Hungarian();
};

#endif 

