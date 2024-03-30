#if !defined(DadaWSUtil_H__)
#define DadaWSUtil_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "DadaWSUtil.h"
#include "Logger.h"
#include "SegmentWS.h"

#include <random>
#include <algorithm>


class DadaWSUtil 
{
	public:				
		static int ThresholdLabelEdgeWeight(MamaGraph &myGraph, double threshold = 0.0);
		static int LabelVertices(MamaVId id, int label, double threshold, MamaGraph &myGraph);

		static void PrintLabelCount(std::map<MamaVId, int> &lmap); 

		/**
		* This tracks the minimax edge for each primary edge
		**/		

		static void ApplyWatershed(MamaGraph &myGraph, map<MamaEId, MamaEId> &watershedNeighbors);

		static void ExportSegmentWS(SegmentWS &ws, string fname, int saveBase = 0);
		static void ExportSegmentWS(SegmentWS &ws, std::ofstream &fout);
		static void ImportSegmentWS(SegmentWS &ws, string fname, int saveBase = 0);
		static void ImportSegmentWS(SegmentWS &ws, std::ifstream &fin);

		static void ExportMamaGraph(MamaGraph &m, string fname);
		static void ExportMamaGraph(MamaGraph &m, std::ofstream &fout);
		static void ImportMamaGraph(MamaGraph &m, string fname);
		static void ImportMamaGraph(MamaGraph &m, std::ifstream &fout);

		static void ChooseRandomFeatures(vector<int> &fv, int outOf, int subsetSize = -1);
		static void ChooseRandomEdges(vector<MamaVId> &origv, MamaGraph &gp, int N, vector< pair<MamaVId, MamaVId> > &newe);
		static void ChooseDegreeEdges(vector<MamaVId> &origv, MamaGraph &gp, int DN, vector< pair<MamaVId, MamaVId> > &newe);

		static void SetRandomSeed(int seed); 
		
		DadaWSUtil();
		virtual ~DadaWSUtil();
	private:				
		static std::mt19937 m_rgen; 
		static std::random_device m_rd;
		static Logger m_logger;				
}; 

#endif 

