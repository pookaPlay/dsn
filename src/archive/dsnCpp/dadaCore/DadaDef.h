#if !defined(KookaDef_H__)
#define KookaDef_H__

#include <string>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include "opencv2/opencv.hpp"
#include <memory>

typedef boost::tokenizer< boost::char_separator<char> > StringTokenizer;

#define SMALLEST_FLOAT		-std::numeric_limits<float>::max()
#define LARGEST_FLOAT		std::numeric_limits<float>::max()
#define SMALLEST_DOUBLE		-std::numeric_limits<double>::max()
#define LARGEST_DOUBLE		std::numeric_limits<double>::max()
#define SMALLEST_INT		-std::numeric_limits<int>::max()
#define LARGEST_INT			std::numeric_limits<int>::max()

#define FEATURE_TOLERANCE		1.0e-12
#define FLOAT_TOLERANCE		    1.0e-16

#if !defined(NEAREST_INT)
#define NEAREST_INT(X) (X > (floor(X)+0.5)) ? ceil(X) : floor(X)
#endif 

#if !defined(IJ)
#define IJ(matrix, ncol, i, j) (matrix[(j)*(ncol) + (i)])
#endif 

#if !defined(CLOSE_ENOUGH)
#define CLOSE_ENOUGH(f1, f2) (fabs(f1-f2) < FLOAT_TOLERANCE) 
#endif 

#if !defined(mxIsFinite)
#define mxIsFinite(x) x < LARGEST_DOUBLE
#endif

std::string MatTypeAsString(int number);
std::string FrameFileName(std::string fileName, unsigned int frameNum, int spaceNum);
std::string v2s(std::vector<int> &tempi);
int CheckMask(cv::Mat &mask, int i, int j, int winSize);

template<typename T, typename... Args> std::unique_ptr<T> make_unique(Args&&... args); 

static double jetcolormap[] = {	
	     0         ,0    ,0.5625,
         0         ,0    ,0.6250,
         0         ,0    ,0.6875,
         0         ,0    ,0.7500,
         0         ,0    ,0.8125,
         0         ,0    ,0.8750,
         0         ,0    ,0.9375,
         0         ,0    ,1.0000,
         0    ,0.0625    ,1.0000,
         0    ,0.1250    ,1.0000,
         0    ,0.1875    ,1.0000,
         0    ,0.2500    ,1.0000,
         0    ,0.3125    ,1.0000,
         0    ,0.3750    ,1.0000,
         0    ,0.4375    ,1.0000,
         0    ,0.5000    ,1.0000,
         0    ,0.5625    ,1.0000,
         0    ,0.6250    ,1.0000,
         0    ,0.6875    ,1.0000,
         0    ,0.7500    ,1.0000,
         0    ,0.8125    ,1.0000,
         0    ,0.8750    ,1.0000,
         0    ,0.9375    ,1.0000,
         0    ,1.0000    ,1.0000,
    0.0625    ,1.0000    ,0.9375,
    0.1250    ,1.0000    ,0.8750,
    0.1875    ,1.0000    ,0.8125,
    0.2500    ,1.0000    ,0.7500,
    0.3125    ,1.0000    ,0.6875,
    0.3750    ,1.0000    ,0.6250,
    0.4375    ,1.0000    ,0.5625,
    0.5000    ,1.0000    ,0.5000,
    0.5625    ,1.0000    ,0.4375,
    0.6250    ,1.0000    ,0.3750,
    0.6875    ,1.0000    ,0.3125,
    0.7500    ,1.0000    ,0.2500,
    0.8125    ,1.0000    ,0.1875,
    0.8750    ,1.0000    ,0.1250,
    0.9375    ,1.0000    ,0.0625,
    1.0000    ,1.0000         ,0,
    1.0000    ,0.9375         ,0,
    1.0000    ,0.8750         ,0,
    1.0000    ,0.8125         ,0,
    1.0000    ,0.7500         ,0,
    1.0000    ,0.6875         ,0,
    1.0000    ,0.6250         ,0,
    1.0000    ,0.5625         ,0,
    1.0000    ,0.5000         ,0,
    1.0000    ,0.4375         ,0,
    1.0000    ,0.3750         ,0,
    1.0000    ,0.3125         ,0,
    1.0000    ,0.2500         ,0,
    1.0000    ,0.1875         ,0,
    1.0000    ,0.1250         ,0,
    1.0000    ,0.0625         ,0,
    1.0000         ,0         ,0,
    0.9375         ,0,         0,
    0.8750         ,0,         0,
    0.8125         ,0,         0,
    0.7500         ,0,         0,
    0.6875         ,0,         0,
    0.6250         ,0,         0,
    0.5625         ,0,         0,
	0.5000         ,0,         0 };

static double hsvcolormap[] = {	
    1.0000         ,0         ,0,
    1.0000    ,0.0938         ,0,
    1.0000    ,0.1875         ,0,
    1.0000    ,0.2813         ,0,
    1.0000    ,0.3750         ,0,
    1.0000    ,0.4688         ,0,
    1.0000    ,0.5625         ,0,
    1.0000    ,0.6563         ,0,
    1.0000    ,0.7500         ,0,
    1.0000    ,0.8438         ,0,
    1.0000    ,0.9375         ,0,
    0.9688    ,1.0000         ,0,
    0.8750    ,1.0000         ,0,
    0.7813    ,1.0000         ,0,
    0.6875    ,1.0000         ,0,
    0.5938    ,1.0000         ,0,
    0.5000    ,1.0000         ,0,
    0.4063    ,1.0000         ,0,
    0.3125    ,1.0000         ,0,
    0.2188    ,1.0000         ,0,
    0.1250    ,1.0000         ,0,
    0.0313    ,1.0000         ,0,
         0    ,1.0000    ,0.0625,
         0    ,1.0000    ,0.1563,
         0    ,1.0000    ,0.2500,
         0    ,1.0000    ,0.3438,
         0    ,1.0000    ,0.4375,
         0    ,1.0000    ,0.5313,
         0    ,1.0000    ,0.6250,
         0    ,1.0000    ,0.7188,
         0    ,1.0000    ,0.8125,
         0    ,1.0000    ,0.9063,
         0    ,1.0000    ,1.0000,
         0    ,0.9063    ,1.0000,
         0    ,0.8125    ,1.0000,
         0    ,0.7188    ,1.0000,
         0    ,0.6250    ,1.0000,
         0    ,0.5313    ,1.0000,
         0    ,0.4375    ,1.0000,
         0    ,0.3438    ,1.0000,
         0    ,0.2500    ,1.0000,
         0    ,0.1563    ,1.0000,
         0    ,0.0625    ,1.0000,
    0.0313         ,0    ,1.0000,
    0.1250         ,0    ,1.0000,
    0.2188         ,0    ,1.0000,
    0.3125         ,0    ,1.0000,
    0.4063         ,0    ,1.0000,
    0.5000         ,0    ,1.0000,
    0.5938         ,0    ,1.0000,
    0.6875         ,0    ,1.0000,
    0.7813         ,0    ,1.0000,
    0.8750         ,0    ,1.0000,
    0.9688         ,0    ,1.0000,
    1.0000         ,0    ,0.9375,
    1.0000         ,0    ,0.8438,
    1.0000         ,0    ,0.7500,
    1.0000         ,0    ,0.6563,
    1.0000         ,0    ,0.5625,
    1.0000         ,0    ,0.4688,
    1.0000         ,0    ,0.3750,
    1.0000         ,0    ,0.2813,
    1.0000         ,0    ,0.1875,
	1.0000         ,0    ,0.0938 };

#endif 

