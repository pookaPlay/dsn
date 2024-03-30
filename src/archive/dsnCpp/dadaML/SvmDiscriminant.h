#if !defined(SvmDiscriminant_H__)
#define SvmDiscriminant_H__

#include <vector>
using namespace std;

#include "Discriminant.h"
#include "opencv2/opencv.hpp"
#include "linear.h"

typedef struct problem SvmProblem;
typedef struct parameter SvmParameters;
typedef struct model* SvmModel;
#define SvmMalloc(type,n) (type *)malloc((n)*sizeof(type))
//#define INF HUGE_VAL

void no_print_out(const char *s);

class SvmDiscriminant : public Discriminant
{
	public:					
		SvmModel myModel;	
		SvmProblem myProb;
		SvmParameters myParam;

		struct feature_node *x_space;	
		float mySign;
		
		void Train(cv::Mat &mlData, cv::Mat &labels, cv::Mat &weights, float regularize = -1.0f);
		//void Apply(cv::Mat &mlData, cv::Mat &result); 
		
		void SaveModel(string fname);
		void LoadModel(string fname);

		void Mat2SvmProblem(cv::Mat &mlData, cv::Mat &labels, cv::Mat &weights, SvmProblem &prob);
		void DefaultSvmParameters(SvmParameters &param);
		
		void CleanUp();
		SvmDiscriminant();
		virtual ~SvmDiscriminant();
}; 


#endif 


/*	SVM PARAMETERS
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- multi-class support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"	11 -- L2-regularized L2-loss epsilon support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss epsilon support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss epsilon support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" 
	"	-s 1, 3, 4, and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"-W weight_file: set weight file\n"

	*/