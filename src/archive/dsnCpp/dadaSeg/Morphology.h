// Morphology.h
// $Id: Morphology.h$
// Authors: Neal R. Harvey, Reid Porter
// Purpose: wrapper for all kinds of great mathematical morphology
//////////////////////////////////////////////////////////////////

#if !defined(Morphology_H__)
#define Morphology_H__

#include "opencv2/opencv.hpp"

#include <algorithm>
#include <numeric>
using namespace std;


// Queue Structure Used for Fast/Efficient Implementation of Reconstruction Operators
typedef struct myqueue {
  long     *buffer;
  long     head;
  long     tail;
  long     size;
  long     in_queue;
  long     num_slots;
  int      check_sum;
} Queue;

/**
 * Morphology is a class for implementing morphological reconstruction processing layers.
 */

enum ReconOp_TYPE { RECON_OPEN, RECON_CLOSE };

class Morphology 
{
public:	

	/** 
	 * Top-level call for reconstruction. 
	 * @param in input image
	 * @param out output image
	 * @param in opType image
	 * @param winSize Size of window radius ie. 1 is a 3x3 window
	 **/

	static void Reconstruction(cv::Mat &in, cv::Mat &out, ReconOp_TYPE opType, int radius);

	static void Gradient(cv::Mat &in, cv::Mat &out, int radius);
	
	static void Open(cv::Mat &in, cv::Mat &out, int radius);

	static void Positive_Reconstruction(cv::Mat &in, cv::Mat &out); 
	
	static void Negative_Reconstruction(cv::Mat &in, cv::Mat &out);

	Morphology();

	virtual ~Morphology();

private:

	static void *queue_make(long size);

	static int valid_queue(Queue *queue);

	static int queue_kill(void *Q);

	static int queue_add(void *Q, long item);

	static int queue_get(void *Q);

	static int queue_num(void *Q);

	static int is_full(void *Q);

	static int queue_free(void *Q);

	static void Positive_Raster_Scan(float *mask_array, float *marker_array, int rows, int cols);

	static void Positive_AntiRaster_Scan(float *mask_array, float *marker_array, int rows, int cols, void *Q);

	static void Positive_Propagation(float *mask_array, float *marker_array, int rows, int cols, void *Q);

	static void Negative_Raster_Scan(float *mask_array, float *marker_array, int rows, int cols);

	static void Negative_AntiRaster_Scan(float *mask_array, float *marker_array, int rows, int cols, void *Q);

	static void Negative_Propagation(float *mask_array, float *marker_array, int rows, int cols, void *Q);
	
};

#endif 
