#include "DadaDef.h"

#include <string>
using namespace std; 


template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::string MatTypeAsString(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

string FrameFileName(string fileName, unsigned int frameNum, int spaceNum)
{
		string subString, resultString;
		char buf[256];
		unsigned int i, extra = 0;
		subString = string("");
		
		if (spaceNum > 0) {
			if (frameNum < 10) extra = spaceNum - 1;
			else if (frameNum < 100) extra = spaceNum - 2;
			else if (frameNum < 1000) extra = spaceNum - 3;
			else if (frameNum < 10000) extra = spaceNum - 4;
			if (extra < 0) extra = 0;
			for(i=0; i<extra; i++) subString = subString + "0";
		}		
		if (spaceNum > 0) 
			sprintf(buf, "%s%s%i", fileName.c_str(), subString.c_str(), frameNum);
		else 
			sprintf(buf, "%s.%s%i", fileName.c_str(), subString.c_str(), frameNum);
		resultString = string(buf);		
		return(resultString);
}

std::string v2s(std::vector<int> &tempi)
{
	string temps = "";
	for (int i = 0; i < tempi.size(); i++) {
		temps += boost::lexical_cast<string>(tempi[i]); 
		temps += " ";
	}
	return(temps); 
}


int CheckMask(cv::Mat &mask, int i, int j, int winSize)
{
  int tl = mask.at<unsigned char>(j - winSize, i - winSize);
  int tr = mask.at<unsigned char>(j - winSize, i + winSize);
  int br = mask.at<unsigned char>(j + winSize, i + winSize);
  int bl = mask.at<unsigned char>(j + winSize, i - winSize);
  if (tl && tr && br && bl) return(1);
  else return(0);
}


