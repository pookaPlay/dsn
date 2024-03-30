#if !defined(SynData_H__)
#define SynData_H__

#include <vector>
using namespace std;

class SynData 
{
	public:					
		void SynTwoClassData(int N, int D, vector< vector<float> > &mlData, vector<float> &labels, float sigma = 1.0f);
		void SynTwoClassThreshold(int N, int D, vector< vector<float> > &mlData, vector<float> &labels, float sigma = 1.0f);
		void SynGaussianData(int N, int D, vector< vector<float> > &mlData, vector<float> &labels);
		void PlotData(vector< vector<float> > &mlData, vector<float> &labels, int imgSize, int pointScale = 1);		
		void PopCode(vector< vector<float> > &mlData, vector< vector<float> > &mlDataOut, float minVal, float maxVal, int numCount);
		void Syn2DGaussian(int N, double myTerm, double crossTerm, vector< vector<float> > &mlData);

		SynData();
		virtual ~SynData();
}; 


#endif 