#if !defined(DadaClassifierParam_H__)
#define DadaClassifierParam_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "SegmentWS.h"

class DadaClassifierParam 
{
	public:						
		cv::Mat & W() {
			return(m_weights);
		}; 

		double & T() {
			return(m_threshold);
		};

		int & I() {
			return(m_index);
		};

		int & D() {
			return(m_D);
		};

		std::string & SegType() {
			return(m_segType);
		};

		int & ProblemType() {
			return(m_probType);
		};

		void SetValidFeatures(vector<int> &fv) {
			m_validFeatures = fv; 
		};
		vector<int> & GetValidFeatures() {
			return(m_validFeatures); 
		};

		void Init(int D); 
		void Print();		

		void Save();
		void Load();
		void Save(string fname);
		void Load(string fname);
		void Save(cv::FileStorage &fs) const;
		void Load(cv::FileStorage &fs);
		void Load(cv::FileNodeIterator &fs);

		void CopyTo(std::shared_ptr<DadaClassifierParam> &dest);

		DadaClassifierParam(std::shared_ptr<DadaParam> &param);
		virtual ~DadaClassifierParam();

	protected:				
		std::shared_ptr< DadaParam > m_param;
		cv::Mat m_weights; 
		vector<int> m_validFeatures;
		int    m_index;
		std::string m_segType; 
		double m_threshold; 
		int m_D; 
		int m_probType;

		Logger m_logger;				
}; 

#endif 

