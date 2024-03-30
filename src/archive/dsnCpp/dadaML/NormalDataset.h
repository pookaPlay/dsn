#if !defined(normal_dataset_H__)
#define normal_dataset_H__

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;


class NormalDataset
{
	public:			
	static void GenerateSamples(int N, int D, double *a, double *mu, int seed, vector< vector<float> > &samples);

	static double *multinormal_sample ( int m, int n, double a[], double mu[], int *seed );

	int i4_max ( int i1, int i2 );
	int i4_min ( int i1, int i2 );	
	double r8_uniform_01 ( int *seed );
	void r8mat_print ( int m, int n, double a[], string title );
	void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi, 
	  int jhi, string title );
	void r8mat_write ( string output_filename, int m, int n, double table[] );
	static double *r8po_fa ( int n, double a[] );
	static double *r8vec_normal_01_new ( int n, int *seed );
	void r8vec_print ( int n, double a[], string title );
	static double *r8vec_uniform_01_new ( int n, int *seed );
	void timestamp ( );

	NormalDataset();
	virtual ~NormalDataset();
};

#endif
