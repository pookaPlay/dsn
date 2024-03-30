#include "ErrorEval.h"
#include "Info.h"
#include "MamaException.h"
#include "MamaDef.h"

#include "opencv2/opencv.hpp"
//#include "cv.h"
using namespace cv;
//using namespace std;
static Info info;

double ErrorEval::RandIndexMatSampled(cv::Mat &seg1, cv::Mat &seg2, int numIter, int verbose)
{
	double err, dr, fa, pc, nc, tpc, fnc;
	tpc = 0.0; fnc = 0.0; pc = 0.0; nc = 0.0;
	if (seg1.rows != seg2.rows) BOOST_THROW_EXCEPTION( UnexpectedSize() ); 
	if (seg1.cols != seg2.cols) BOOST_THROW_EXCEPTION( UnexpectedSize() ); 
	
	if (verbose) cout << "Rand estimate...\n";

	int N = seg1.rows*seg1.cols;
	int n1i, n2i; 
	float *op1 = (float *) seg1.ptr();  	
	float *op2 = (float *) seg2.ptr();  	

	for(int i=0; i < numIter; i++) {
		n1i = rand() % N;
		n2i = rand() % N;

		float *s1p1 = op1 + n1i;
		float *s1p2 = op1 + n2i;
		float *s2p1 = op2 + n1i;
		float *s2p2 = op2 + n2i;

		if ( CLOSE_ENOUGH( *s1p1, *s1p2 ) ) {
			pc += 1.0;
			if ( CLOSE_ENOUGH( *s2p1, *s2p2 ) ) tpc += 1.0;					
		} else {
			nc += 1.0;
			if ( CLOSE_ENOUGH( *s2p1, *s2p2 ) ) fnc += 1.0;
		}
	}

	double totalPair = pc + nc;

	err = (pc - tpc + fnc) / (pc + nc);
	dr = tpc / pc;
	fa = fnc / nc;	
	float posErrs = pc - tpc;

	if (verbose) {
		cout << "ER: " << err << "\n";
		cout << "DR: " << dr << "\n";
		cout << "FA: " << fa << "\n";
		cout << "PE: " << posErrs << "\n";	
		cout << "NE: " << fnc << "\n";	
		cout << "PC: " << pc << "\n";	
		cout << "NC: " << nc << "\n";	
	} 
	return(err);
	//info(1, "%i/%i pos errors and %i/%i neg errors\n", errs, pc, fnc, nc); 
}

double ErrorEval::RandIndexMat(cv::Mat &seg1, cv::Mat &seg2, int verbose)
{
	double err, dr, fa, pc, nc, tpc, fnc;
	tpc = 0.0; fnc = 0.0; pc = 0.0; nc = 0.0;
	if (seg1.rows != seg2.rows) BOOST_THROW_EXCEPTION( UnexpectedSize() ); 
	if (seg1.cols != seg2.cols) BOOST_THROW_EXCEPTION( UnexpectedSize() ); 
	
	if (verbose) cout << "Rand estimate...\n";
	
	for(int n1i=0; n1i < (seg1.rows*seg1.cols); n1i++) {
	for(int n2i=0; n2i < (seg1.rows*seg1.cols); n2i++) {
		if (n1i < n2i) {
			float *s1p1 = (float *) seg1.ptr();  s1p1 += n1i;
			float *s1p2 = (float *) seg1.ptr();  s1p2 += n2i;
			float *s2p1 = (float *) seg2.ptr();  s2p1 += n1i;
			float *s2p2 = (float *) seg2.ptr();  s2p2 += n2i;


			if ( CLOSE_ENOUGH( *s1p1, *s1p2 ) ) {
				pc += 1.0;
				if ( CLOSE_ENOUGH( *s2p1, *s2p2 ) ) tpc += 1.0;					
			} else {
				nc += 1.0;
				if ( CLOSE_ENOUGH( *s2p1, *s2p2 ) ) fnc += 1.0;
			}
		}
	}
	}

	double totalPair = pc + nc;

	err = (pc - tpc + fnc) / (pc + nc);
	dr = tpc / pc;
	fa = fnc / nc;	
	float posErrs = pc - tpc;

	if (verbose) {
		cout << "ER: " << err << "\n";
		cout << "DR: " << dr << "\n";
		cout << "FA: " << fa << "\n";
		cout << "PE: " << posErrs << "\n";	
		cout << "NE: " << fnc << "\n";	
		cout << "PC: " << pc << "\n";	
		cout << "NC: " << nc << "\n";	
	}
	return(err);
	//info(1, "%i/%i pos errors and %i/%i neg errors\n", errs, pc, fnc, nc); 
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
ErrorEval::ErrorEval() 
{	
}

ErrorEval::~ErrorEval()
{	
}


