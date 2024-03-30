#if !defined(DadaSegmenterForestParam_H__)
#define DadaSegmenterForestParam_H__

#include <vector>
using namespace std;

#include "opencv2/opencv.hpp"
#include "Logger.h"
#include "DadaParam.h"
#include "DadaClassifierParam.h"
#include "DadaSegmenterTreeParam.h"
#include "MamaException.h"

class DadaSegmenterForestParam
{
	public:						
		void Init(); 

		int NumTrees() { return(m_treeParams.size()); };

		std::shared_ptr<DadaSegmenterTreeParam> GetTree(int i) {
			if (i >= m_treeParams.size()) BOOST_THROW_EXCEPTION(UnexpectedSize("don't have the tree param"));
			return(m_treeParams[i]);
		};

		std::shared_ptr<DadaClassifierParam> GetVote() {
			return(m_voteParam);
		};


		void Print();
		void CopyTo(std::shared_ptr<DadaSegmenterForestParam> &dest);

		void Save();
		void Load();
		void Save(string fname);
		void Load(string fname);
		void Save(cv::FileStorage &fs) const;
		void Load(cv::FileStorage &fs);		

		DadaSegmenterForestParam(std::shared_ptr<DadaParam> &param);
		virtual ~DadaSegmenterForestParam();
	
	protected:						
		
		std::shared_ptr<DadaClassifierParam > m_voteParam;
		std::vector< std::shared_ptr<DadaSegmenterTreeParam> > m_treeParams;
		std::shared_ptr< DadaParam > m_param;
		Logger m_logger;
}; 

#endif 

