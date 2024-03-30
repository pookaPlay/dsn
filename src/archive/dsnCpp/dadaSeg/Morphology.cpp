// Morphology.cpp: Implementation of the Morphology class.
//
//////////////////////////////////////////////////////////////////////

#include "Morphology.h"
#include "DadaDef.h"
#include "DadaException.h"

#include "opencv2/opencv.hpp"
using namespace cv;

/*****************************************************************************/
/***** queue_make function declaration.                                  *****/
/***** takes a positive integer and makes enough space for it.           *****/
/***** It returns a pointer to the queue struct in the form of a void *. *****/
/***** If there was an error the return value is NULL.                   *****/
/*****************************************************************************/

void *Morphology::queue_make(long     size) 

{

  Queue     *cq;

  if (size < 0) {
      return NULL;
  }

  cq = (Queue *)malloc(sizeof(Queue));

  if (cq == NULL) {
      return NULL;
  }

  cq->buffer = (long *)malloc(((size + 1) * sizeof(long)));
  cq->head = 0;
  cq->tail = 0;
  cq->size = size;
  cq->num_slots = size;
  cq->in_queue = 0;
  cq->check_sum = 666;
   
  return cq;
}

/*****************************************************************************/
/***** End of queue_make function declaration. *******************************/
/*****************************************************************************/

/*****************************************************************************/
/***** valid_queue function declaration.                                 *****/
/***** This function checks for the existence of a valid queue,          *****/
/***** returning a 1 if one exists, and a 0 if not.                      *****/
/*****************************************************************************/

int Morphology::valid_queue(Queue     *queue)

{  
  if (queue == NULL) {
       return 0;
  } else {
    if (queue->check_sum != 666) {
      return 0;
    } else {
      return 1;
    }
  }
}

/*****************************************************************************/
/***** End of valid_queue function declaration. ******************************/
/*****************************************************************************/

/*****************************************************************************/
/***** queue_kill function declaration.                                  *****/
/***** This function removes a queue and frees up the memory             *****/
/*****************************************************************************/

int Morphology::queue_kill(void     *Q)

{
 
  Queue *queue = (Queue *)Q;

  if (!valid_queue(queue)) {
      return -1;
  }

  free(queue->buffer); 
  free(queue);

  return 0;
} 
  
/*****************************************************************************/
/***** End of queue_kill function declaration. *******************************/
/*****************************************************************************/

/*****************************************************************************/
/***** queue_add function declaration.                                   *****/
/***** This function adds an item to the queue.                          *****/
/***** Takes two arguments. The first is a pointer to a Queue struct and *****/
/***** the second is a pointer to the item that the user is trying to    *****/
/***** insert. It returns 0 on success, -1 otherwise. Trying to overflow *****/
/***** the buffer of a circular queue is not an error. The library       *****/
/***** simply throws away the extra characters and returns 0 to the user *****/
/***** application.                                                      *****/
/*****************************************************************************/

int Morphology::queue_add(void     *Q,
				    long     item)
{

  Queue     *queue = (Queue *)Q;

  if (!valid_queue(queue)) {
    return 1;
  }

  if (queue->num_slots == 0) {
    return 0;
  }

  queue->buffer[queue->head] = item;
  /* Assign item to head of the queue */

  queue->head=(queue->head+1)%queue->size;
  /* update head position marker */

  queue->num_slots--;
  /* Update number of available slots */

  queue->in_queue++;
  /* Update count of items in the queue */

  return 0;  
}

/*****************************************************************************/
/***** End of queue_add function declaration. ********************************/
/*****************************************************************************/

/*****************************************************************************/
/***** queue_get function declaration.                                   *****/
/***** This function retrieves an item from the queue in the form of a   *****/
/***** int *.                                                            *****/
/*****************************************************************************/

int Morphology::queue_get(void     *Q)

{

  Queue     *queue = (Queue *)Q;

  long      current_item;

  if (!valid_queue(queue)) {
    return 0;
  }

  if (queue->in_queue == 0) {
    return 0;  
  }

  current_item = queue->buffer[queue->tail];
  /* Item to be returned comes from the tail of the queue */

  queue->tail = (queue->tail+1)%queue->size;
  /* Update tail position marker */ 

  queue->num_slots++;
  /* Update number of available slots */

  queue->in_queue--;
  /* Update count of items in the queue */ 

  return current_item;
}

/*****************************************************************************/
/***** End of queue_get function declaration. ********************************/
/*****************************************************************************/

/*****************************************************************************/
/***** queue_num function declaration.                                   *****/
/***** This function takes a queue struct and returns the number of      *****/
/***** items currently in the queue.                                     *****/
/*****************************************************************************/

int Morphology::queue_num(void     *Q)

{

  Queue     *queue = (Queue *)Q;

  if (!valid_queue(queue)) {
    return -1;
  }

  return queue->in_queue;
}

/*****************************************************************************/
/***** End of queue_num function declaration. ********************************/
/*****************************************************************************/

/*****************************************************************************/
/***** is_full function declaration.                                     *****/
/***** This function returns TRUE if the queue is full, FALSE is the     *****/
/***** queue isn't.                                                      *****/
/*****************************************************************************/

int Morphology::is_full(void     *Q)

{
  Queue     *queue = (Queue *)Q;

  if (!valid_queue(queue)) {
    return -1;
  }
  
  if (queue->num_slots == 0) {
    return 1;
  } else {
    return 0;
  }
} 

/*****************************************************************************/
/***** End of is_full function declaration. **********************************/
/*****************************************************************************/

/*****************************************************************************/
/***** queue_free function declaration.                                  *****/
/***** This function returns the number of free slots in the queue       *****/
/***** buffer                                                            *****/
/*****************************************************************************/

int Morphology::queue_free(void     *Q)

{
  
  Queue     *queue = (Queue *)Q;
  
  if  (!valid_queue(queue)) {
    return -1;
  }
  
  return queue->num_slots;

}

/*****************************************************************************/
/***** End of queue_free function declaration. *******************************/
/*****************************************************************************/

/*****************************************************************************/
/***** Positive_Raster_Scan function declaration.                        *****/
/***** This function performs the operations in the raster-scan part of  *****/
/***** Luc Vincent's reconstruction algorithm.                           *****/
/*****  "Morphological Grayscale Reconstruction in Image                 *****/
/*****   Analysis: Applications and Efficient Algorithms"                *****/
/*****                                                                   *****/
/*****************************************************************************/

void Morphology::Positive_Raster_Scan(float *mask_array, 
						float *marker_array, 
						int rows, 
						int cols)

{
   int	count_rows, count_cols, row_pos, col_pos, x, y;

   float     temp_pix, max = -30000.0, I_p, J_p;

   //cout << endl << "      Performing Positive Raster Scan ... ";

   for (count_rows = 0; count_rows < rows; count_rows++) {
     for (count_cols = 0; count_cols < cols; count_cols++) {
       max = (float)-30000.0;
       y = -1;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = -1; x <= 1; x++) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, row_pos);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, row_pos);
	   }
	   if (temp_pix > max) {
	     max = temp_pix;
	   }
	 }
       } else {
	 for (x = -1; x <= 1; x++) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, count_rows);
	   }
	   if (temp_pix > max) {
	     max = temp_pix;
	   }
	 }
       }

       y = 0;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = -1; x <= 0; x++) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, row_pos);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, row_pos);
	   }
	   if (temp_pix > max) {
	     max = temp_pix;
	   }
	 }
       }
       else {
	 for (x = -1; x <= 0; x++) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, count_rows);
	   }
	   if (temp_pix > max) {
	     max = temp_pix;
	   }
	 }
       }


       I_p = IJ(mask_array, cols, count_cols, count_rows);
       if (I_p < max) {
	 J_p = I_p;
       } else {
	 J_p = max;
       }
       IJ(marker_array, cols, count_cols, count_rows) = J_p;
     }
   }

   //cout << " ... Completed" << endl;
}

/*****************************************************************************/
/***** End of Positive_Raster_Scan function declaration. *********************/
/*****************************************************************************/

/*****************************************************************************/
/***** Positive_AntiRaster_scan function declaration.                    *****/
/***** This function performs the operations in the anti-raster-scan     *****/
/***** part of Vincent's reconstruction algorithm.                       *****/
/*****                                                                   *****/
/*****************************************************************************/

void Morphology::Positive_AntiRaster_Scan(float *mask_array, 
						    float *marker_array, 
						    int rows, 
						    int cols, 
						    void *Q)

{
   // This routine modifies marker_array and Q. It does a stencil
   // operator on marker_array and adds onto Q pointers to elements os
   // the now modified marker_array 
   int		flag = 0;

   int count_rows, count_cols, x, y, row_pos, col_pos, array_pos;

   float temp_pix, max = -30000.0, J_p, J_q, I_q;

   Queue *fifo_ptr = (Queue *)Q;

   //cout << endl << "      Performing Positive Anti-Raster Scan ... ";

   for (count_rows = (rows - 1); count_rows >= 0; count_rows--) {
     for (count_cols = (cols - 1); count_cols >= 0; count_cols--) {
       array_pos = ((count_rows * cols) + count_cols); 
       max = -30000.0;

       // This entire first section simply sets marker_array = min(mask_array,max(stencil of marker_array))
       // Stencil is {[i+1][j-1], [i+1][j], [i+1][j+1], [i][j+1], [i][j]}

       y = 1;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = 1; x >= -1; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, row_pos);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, row_pos);
	   }
	   if (temp_pix > max) {
	     max = temp_pix;
	   }
	 }
       } else {
	 for (x = 1; x >= -1; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, count_rows);
	   }
	   if (temp_pix > max) {
	     max = temp_pix;
	   }
	 }
       }

       y = 0;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = 1; x >= 0; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, row_pos);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, row_pos);
	   }
	   if (temp_pix > max) {
	     max = temp_pix;
	   }
	 }
       } else {
	 for (x = 1; x >= 0; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, count_rows);
	   }
	   if (temp_pix > max) {
	     max = temp_pix;
	   }
	 }
       }

       IJ(marker_array, cols, count_cols, count_rows) = std::min(max, IJ(mask_array, cols, count_cols, count_rows));
       
       /*************************************************************/
       
       // The last part adds the current position to Q if any element
       // of the stencil is < the current element and < the
       // corresponding element of the mask array.

       J_p = IJ(marker_array, cols, count_cols, count_rows);

       flag = 0;
       y = 1;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {

	 for (x = 1; x >= -1; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     J_q = IJ(marker_array, cols, col_pos, row_pos);
	     I_q = IJ(mask_array, cols, col_pos, row_pos);
	   } else {
	     J_q = IJ(marker_array, cols, count_cols, row_pos);
	     I_q = IJ(mask_array, cols, count_cols, row_pos);
	   }
	   if ((J_q < J_p) && (J_q < I_q)) {
	     flag = 1;
	   }
	 }
       } else {
	 for (x = 1; x >= -1; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     J_q = IJ(marker_array, cols, col_pos, count_rows);
	     I_q = IJ(mask_array, cols, col_pos, count_rows);
	   } else {
	     J_q = IJ(marker_array, cols, count_cols, count_rows);
	     I_q = IJ(mask_array, cols, count_cols, count_rows);
	   }
	   if ((J_q < J_p) && (J_q < I_q)) {
	     flag = 1;
	   }
	 }
       }

       y = 0;
       row_pos = (count_rows + y);
       col_pos = (count_cols + 1);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 if ((col_pos >= 0) && (col_pos < cols)) {
	   J_q = IJ(marker_array, cols, col_pos, row_pos);
	   I_q = IJ(mask_array, cols, col_pos, row_pos);
	 } else {
	   J_q = IJ(marker_array, cols, count_cols, row_pos);
	   I_q = IJ(mask_array, cols, count_cols, row_pos);
	 }
	 if ((J_q < J_p) && (J_q < I_q)) {
	   flag = 1;
	 } 
       } else {
	 if ((col_pos >= 0) && (col_pos < cols)) {
	   J_q = IJ(marker_array, cols, col_pos, count_rows);
	   I_q = IJ(mask_array, cols, col_pos, count_rows);
	 } else {
	   J_q = IJ(marker_array, cols, count_cols, count_rows);
	   I_q = IJ(mask_array, cols, count_cols, count_rows);
	 }
	 if ((J_q < J_p) && (J_q < I_q)) {
	   flag = 1;
	 } 
       }


       if (flag == 1) {
	 queue_add(fifo_ptr, array_pos);
       }
     }
   }

   //cout << " ... Completed" << endl;
}

/****************************************************************************/
/***** End of Positive_AntiRaster_Scan function declaration. ****************/
/****************************************************************************/

/*****************************************************************************/
/***** Positive_Propagation function declaration                           *****/
/***** This function performs the propagation part of Vincent's          *****/
/***** algorithm.                                                        *****/
/*****                                                                   *****/
/*****************************************************************************/

void Morphology::Positive_Propagation(float *mask_array, 
						float *marker_array, 
						int rows, 
						int cols, 
						void *Q)

{
   int x, y, row_pos, col_pos, array_pos, neighbourhood_pos;

   float J_q, I_q, J_p; 

   Queue *fifo_ptr = (Queue *)Q;

   //cout << endl << "      Performing Positive Propagation ... ";

   while (fifo_ptr->in_queue != 0) {
     array_pos = queue_get(fifo_ptr);
     
     col_pos = (array_pos % cols);
     row_pos = (array_pos / cols);
     
     y = -1;
     if (((row_pos + y) >= 0) && ((row_pos + y) < rows)) {
       for (x = -1; x <= 1; x++) {
	 if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	   neighbourhood_pos = (((row_pos + y) * cols) + (col_pos + x));
	 } else {
	   neighbourhood_pos = (((row_pos + y) * cols) + col_pos);
	 }
	 J_q = marker_array[neighbourhood_pos];
	 I_q = mask_array[neighbourhood_pos];
	 J_p = marker_array[array_pos];
	 if ((J_q < J_p) && (I_q != J_q)) {
	   if (J_p < I_q) {
	     J_q = J_p;
	   } else {
	     J_q = I_q;
	   }
	   marker_array[neighbourhood_pos] = J_q;
	   queue_add(fifo_ptr, neighbourhood_pos);
	 }
       }
     } else {
       for (x = -1; x <= 1; x++) {
	 if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	   neighbourhood_pos = ((row_pos * cols) + (col_pos + x));
	 } else {
	   neighbourhood_pos = ((row_pos * cols) + col_pos);
	 }
	 J_q = marker_array[neighbourhood_pos];
	 I_q = mask_array[neighbourhood_pos];
	 J_p = marker_array[array_pos];
	 if ((J_q < J_p) && (I_q != J_q)) {
	   if (J_p < I_q) {
	     J_q = J_p;
	   } else {
	     J_q = I_q;
	   }
	   marker_array[neighbourhood_pos] = J_q;
	   queue_add(fifo_ptr, neighbourhood_pos);
	 }
       }
     }
     
     y = 0;
     if (((row_pos + y) >= 0) && ((row_pos + y) < rows)) {
       x = -1;
       if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	 neighbourhood_pos = (((row_pos + y) * cols) + (col_pos + x));
       } else {
	 neighbourhood_pos = (((row_pos + y) * cols) + col_pos);
       }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q < J_p) && (I_q != J_q)) {
	 if (J_p < I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
       x = 1;
       if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	 neighbourhood_pos = (((row_pos + y) * cols) + (col_pos + x));
       } else {
	 neighbourhood_pos = (((row_pos + y) * cols) + col_pos);
       }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q < J_p) && (I_q != J_q)) {
	 if (J_p < I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
     } else {
       x = -1;
       if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	 neighbourhood_pos = ((row_pos * cols) + (col_pos + x));
       } else {
	 neighbourhood_pos = ((row_pos * cols) + col_pos);
       }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q < J_p) && (I_q != J_q)) {
	 if (J_p < I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
       x = 1;
       if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	 neighbourhood_pos = ((row_pos * cols) + (col_pos + x));
       } else {
	 neighbourhood_pos = ((row_pos * cols) + col_pos);
       }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q < J_p) && (I_q != J_q)) {
	 if (J_p < I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
     }
     y = 1;
     if (((row_pos + y) >= 0) && ((row_pos + y) < rows)) {
       for (x = -1; x <= 1; x++) {
	 if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	   neighbourhood_pos = (((row_pos + y) * cols) + (col_pos + x));
	 } else {
	   neighbourhood_pos = (((row_pos + y) * cols) + col_pos);
	 }
	 J_q = marker_array[neighbourhood_pos];
	 I_q = mask_array[neighbourhood_pos];
	 J_p = marker_array[array_pos];
	 if ((J_q < J_p) && (I_q != J_q)) {
	   if (J_p < I_q) {
	     J_q = J_p;
	   } else {
	     J_q = I_q;
	   }
	   marker_array[neighbourhood_pos] = J_q;
	   queue_add(fifo_ptr, neighbourhood_pos);
	 }
       }
     } else {
       for (x = -1; x <= 1; x++) {
	 if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	   neighbourhood_pos = ((row_pos * cols) + (col_pos + x));
	 } else {
	   neighbourhood_pos = ((row_pos * cols) + col_pos);
	 }
	 J_q = marker_array[neighbourhood_pos];
	 I_q = mask_array[neighbourhood_pos];
	 J_p = marker_array[array_pos];
	 if ((J_q < J_p) && (I_q != J_q)) {
	   if (J_p < I_q) {
	     J_q = J_p;
	   } else {
	     J_q = I_q;
	   }
	   marker_array[neighbourhood_pos] = J_q;
	   queue_add(fifo_ptr, neighbourhood_pos);
	 }
       }
     }
   }

   //cout << " ... Completed" << endl;
}

/*****************************************************************************/
/***** End of Positive_Propagation function declaration. *********************/
/*****************************************************************************/

/*****************************************************************************/
/***** Negative_Raster_Scan function declaration.                        *****/
/***** This function performs the operations in the raster-scan part of  *****/
/***** Vincent's reconstruction algorithm.                               *****/
/*****                                                                   *****/
/***** Note: There are more efficient/better ways of dealing with the    *****/
/***** problem of edges, but this is a first stab, and pretty darned     *****/
/***** simple.                                                           *****/
/*****************************************************************************/

void Morphology::Negative_Raster_Scan(float *mask_array, 
						float *marker_array, 
						int rows, 
						int cols)

{
   int	count_rows, count_cols, row_pos, col_pos, x, y;

   float temp_pix, min = 30000.0, I_p, J_p;

   //cout << endl << "      Performing Negative Raster Scan ... ";

   for (count_rows = 0; count_rows < rows; count_rows++) {
     for (count_cols = 0; count_cols < cols; count_cols++) {
       min = 30000.0;
       y = -1;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = -1; x <= 1; x++) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, row_pos);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, row_pos);
	   }
	   if (temp_pix < min) {
	     min = temp_pix;
	   }
	 }
       } else {
	 for (x = -1; x <= 1; x++) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, count_rows);
	   }
	   if (temp_pix < min) {
	     min = temp_pix;
	   }
	 }
       }
       y = 0;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = -1; x <= 0; x++) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, row_pos);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, row_pos);
	   }
	   if (temp_pix < min) {
	     min = temp_pix;
	   }
	 }
       } else {
	 for (x = -1; x <= 0; x++) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, count_rows);
	   }
	   if (temp_pix < min) {
	     min = temp_pix;
	   }
	 }
       }
       I_p = *(mask_array + (count_rows * cols) + count_cols);
       if (I_p > min) {
	 J_p = I_p;
       } else {
	 J_p = min;
       }
       IJ(marker_array, cols, count_cols, count_rows) = J_p;
     }
   }

   //cout << " ... Completed" << endl;
}

/****************************************************************************/
/***** End of Negative_Raster_Scan function declaration. ********************/
/****************************************************************************/

/*****************************************************************************/
/***** Negative_AntiRaster_Scan function declaration.                    *****/
/***** This function performs the operations in the anti-raster-scan     *****/
/***** part of Vincent's reconstruction algorithm.                       *****/
/*****                                                                   *****/
/*****************************************************************************/

void Morphology::Negative_AntiRaster_Scan(float *mask_array, 
						    float *marker_array, 
						    int rows, 
						    int cols, 
						    void *Q)

{
   int flag = 0;

   int count_rows, count_cols, x, y, row_pos, col_pos, array_pos;

   float temp_pix, min = 30000.0, I_p, J_p, J_q, I_q;

   Queue *fifo_ptr = (Queue *)Q;

   //cout << endl << "      Performing Negative Anti-Raster Scan ... ";

   for (count_rows = (rows - 1); count_rows >= 0; count_rows--) {
     for (count_cols = (cols - 1); count_cols >= 0; count_cols--) {
       array_pos = ((count_rows * cols) + count_cols); 
       min = 30000.0;
       y = 1;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = 1; x >= -1; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, row_pos);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, row_pos);
	   }
	   if (temp_pix < min) {
	     min = temp_pix;
	   }
	 }
       } else {
	 for (x = 1; x >= -1; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, count_rows);
	   }
	   if (temp_pix < min) {
	     min = temp_pix;
	   }
	 }
       }
       y = 0;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = 1; x >= 0; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, row_pos);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, row_pos);
	   }
	   if (temp_pix < min) {
	     min = temp_pix;
	   }
	 }
       } else {
	 for (x = 1; x >= 0; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     temp_pix = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     temp_pix = IJ(marker_array, cols, count_cols, count_rows);
	   }
	   if (temp_pix < min) {
	     min = temp_pix;
	   }
	 }
       }
       I_p = *(mask_array + (count_rows * cols) + count_cols);
       if (I_p > min) {
	 J_p = I_p;
       } else {
	 J_p = min;
       }
       IJ(marker_array, cols, count_cols, count_rows) = J_p;

       /*************************************************************/

       flag = 0;
       y = 1;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 for (x = 1; x >= -1; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     J_q = IJ(marker_array, cols, col_pos, row_pos);
	     J_p = IJ(marker_array, cols, count_cols, count_rows);
	     I_q = IJ(mask_array, cols, col_pos, row_pos);
	   } else {
	     J_q = IJ(marker_array, cols, count_cols, row_pos);
	     J_p = IJ(marker_array, cols, count_cols, count_rows);
	     I_q = IJ(mask_array, cols, count_cols, row_pos);
	   }
	   if ((J_q > J_p) && (J_q > I_q)) {
	     flag = 1;
	   }
	 }
       } else {
	 for (x = 1; x >= -1; x--) {
	   col_pos = (count_cols + x);
	   if ((col_pos >= 0) && (col_pos < cols)) {
	     J_q = IJ(marker_array, cols, col_pos, count_rows);
	     J_p = IJ(marker_array, cols, count_cols, count_rows);
	     I_q = IJ(marker_array, cols, col_pos, count_rows);
	   } else {
	     J_q = IJ(marker_array, cols, count_cols, count_rows);
	     J_p = IJ(marker_array, cols, count_cols, count_rows);
	     I_q = IJ(mask_array, cols, count_cols, count_rows);
	   }
	   if ((J_q > J_p) && (J_q > I_q)) {
	     flag = 1;
	   }
	 }
       }
       y = 0;
       row_pos = (count_rows + y);
       if ((row_pos >= 0) && (row_pos < rows)) {
	 x = 1;
	 col_pos = (count_cols + x);
	 if ((col_pos >= 0) && (col_pos < cols)) {
	   J_q = IJ(marker_array, cols, col_pos, row_pos);
	   J_p = IJ(marker_array, cols, count_cols, count_rows);
	   I_q = IJ(mask_array, cols, col_pos, row_pos);
	 } else {
	   J_q = IJ(marker_array, cols, count_cols, row_pos);
	   J_p = IJ(marker_array, cols, count_cols, count_rows);
	   I_q = IJ(mask_array, cols, count_cols, row_pos);
	 }
	 if ((J_q > J_p) && (J_q > I_q)) {
	   flag = 1;
	 } 
       } else {
	 x = 1;
	 col_pos = (count_cols + x);
	 if ((col_pos >= 0) && (col_pos < cols)) {
	   J_q = IJ(marker_array, cols, col_pos, count_rows);
	   J_p = IJ(marker_array, cols, count_cols, count_rows);
	   I_q = IJ(mask_array, cols, col_pos, count_rows);
	 } else {
	   J_q = IJ(marker_array, cols, count_cols, count_rows);
	   J_p = IJ(marker_array, cols, count_cols, count_rows);
	   I_q = IJ(mask_array, cols, count_cols, count_rows);
	 }
	 if ((J_q > J_p) && (J_q > I_q)) {
	   flag = 1;
	 } 
       }
       if (flag == 1) {
	 queue_add(fifo_ptr, array_pos);
       }
     }
   }

   //cout << " ... Completed" << endl;
}

/****************************************************************************/
/***** End of Negatibe_AntiRaster_scan function declaration. ****************/
/****************************************************************************/

/*****************************************************************************/
/***** Negative_Propagation function declaration                         *****/
/***** This function performs the propagation part of Vincent's          *****/
/***** algorithm.                                                        *****/
/*****************************************************************************/

void Morphology::Negative_Propagation(float *mask_array, 
						float *marker_array, 
						int rows, 
						int cols, 
						void *Q)
  
{
   int x, y, row_pos, col_pos, array_pos, neighbourhood_pos;

   float J_q, I_q, J_p; 

   Queue *fifo_ptr = (Queue *)Q;

   //cout << endl << "      Performing Negative Propagation ... ";

   while (fifo_ptr->in_queue != 0) {
     array_pos = queue_get(fifo_ptr);

     col_pos = (array_pos % cols);
     row_pos = (array_pos / cols);
     
     y = -1;
     if (((row_pos + y) >= 0) && ((row_pos + y) < rows)) {
       for (x = -1; x <= 1; x++) {
	 if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	   neighbourhood_pos = (((row_pos + y) * cols) + (col_pos + x));
	 } else {
	   neighbourhood_pos = (((row_pos + y) * cols) + col_pos);
	 }
	 J_q = marker_array[neighbourhood_pos];
	 I_q = mask_array[neighbourhood_pos];
	 J_p = marker_array[array_pos];
	 if ((J_q > J_p) && (I_q != J_q)) {
	   if (J_p > I_q) {
	     J_q = J_p;
	   } else {
	     J_q = I_q;
	   }
	   marker_array[neighbourhood_pos] = J_q;
	   queue_add(fifo_ptr, neighbourhood_pos);
	 }
       }
     } else {
       for (x = -1; x <= 1; x++) {
	 if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	   neighbourhood_pos = ((row_pos * cols) + (col_pos + x));
	 } else {
	   neighbourhood_pos = ((row_pos * cols) + col_pos);
	 }
   	 J_q = marker_array[neighbourhood_pos];
	 I_q = mask_array[neighbourhood_pos];
	 J_p = marker_array[array_pos];
	 if ((J_q > J_p) && (I_q != J_q)) {
	   if (J_p > I_q) {
	     J_q = J_p;
	   } else {
	     J_q = I_q;
	   }
	   marker_array[neighbourhood_pos] = J_q;
	   queue_add(fifo_ptr, neighbourhood_pos);
	 }
       }
     }
     
     y = 0;
     if (((row_pos + y) >= 0) && ((row_pos + y) < rows)) {
       x = -1;
       if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	 neighbourhood_pos = (((row_pos + y) * cols) + (col_pos + x));
       } else {
	 neighbourhood_pos = (((row_pos + y) * cols) + col_pos);
       }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q > J_p) && (I_q != J_q)) {
	 if (J_p > I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
       x = 1;
       if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	 neighbourhood_pos = (((row_pos + y) * cols) + (col_pos + x));
       } else {
	 neighbourhood_pos = (((row_pos + y) * cols) + col_pos);
       }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q > J_p) && (I_q != J_q)) {
	 if (J_p > I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
     } else {
       x = -1;
       if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	 neighbourhood_pos = ((row_pos * cols) + (col_pos + x));
       } else {
	 neighbourhood_pos = ((row_pos * cols) + col_pos);
       }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q > J_p) && (I_q != J_q)) {
	 if (J_p > I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
       x = 1;
       if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	 neighbourhood_pos = ((row_pos * cols) + (col_pos + x));
       } else {
	 neighbourhood_pos = ((row_pos * cols) + col_pos);
       }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q > J_p) && (I_q != J_q)) {
	 if (J_p > I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
     }
     y = 1;
     if (((row_pos + y) >= 0) && ((row_pos + y) < rows)) {
       for (x = -1; x <= 1; x++) {
	 if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	   neighbourhood_pos = (((row_pos + y) * cols) + (col_pos + x));
	 } else {
	   neighbourhood_pos = (((row_pos + y) * cols) + col_pos);
	 }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q > J_p) && (I_q != J_q)) {
	 if (J_p > I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
       }
     } else {
       for (x = -1; x <= 1; x++) {
	 if (((col_pos + x) >= 0) && ((col_pos + x) < cols)) {
	   neighbourhood_pos = ((row_pos * cols) + (col_pos + x));
	 } else {
	   neighbourhood_pos = ((row_pos * cols) + col_pos);
	 }
       J_q = marker_array[neighbourhood_pos];
       I_q = mask_array[neighbourhood_pos];
       J_p = marker_array[array_pos];
       if ((J_q > J_p) && (I_q != J_q)) {
	 if (J_p > I_q) {
	   J_q = J_p;
	 } else {
	   J_q = I_q;
	 }
	 marker_array[neighbourhood_pos] = J_q;
	 queue_add(fifo_ptr, neighbourhood_pos);
       }
       }
     }
   }

   //cout << " ... Completed" << endl;
}

/*****************************************************************************/
/***** End of Negative_Propagation function declaration. *********************/
/*****************************************************************************/


void Morphology::Positive_Reconstruction(cv::Mat &in, cv::Mat &out)
{	
	float *inP, *outP; 
	Mat tempM = in.clone();
	inP = (float *) tempM.ptr(); 	
	if (in.size != out.size) BOOST_THROW_EXCEPTION( UnexpectedSize("Mats different sizes") ); 

	outP = (float *) out.ptr(); 
			
	// Allocate fifo_queue memory 
	void *fifo_queue = queue_make( in.rows*in.cols);

	// Perform raster scan part of reconstruction algorithm 
	Positive_Raster_Scan(inP, outP, in.rows, in.cols); 

	// Perform anti-raster scan part of reconstruction algorithm
	Positive_AntiRaster_Scan(inP, outP, in.rows, in.cols, fifo_queue);

	// Perform propgation part of reconstruction algorithm 
	Positive_Propagation(inP, outP, in.rows, in.cols, fifo_queue);

	// Free up fifo memory 
	queue_kill(fifo_queue);
}

void Morphology::Negative_Reconstruction(cv::Mat &in, cv::Mat &out)
{
	float *inP, *outP; 
	Mat tempM = in.clone();
	inP = (float *) tempM.ptr(); 
	if (in.size != out.size) BOOST_THROW_EXCEPTION( UnexpectedSize("Mats different sizes") ); 

	outP = (float *) out.ptr(); 
			
	// Allocate fifo_queue memory 
	void *fifo_queue = queue_make( in.rows*in.cols);

	// Perform raster scan part of reconstruction algorithm 
	Negative_Raster_Scan(inP, outP, in.rows, in.cols); 

	// Perform anti-raster scan part of reconstruction algorithm
	Negative_AntiRaster_Scan(inP, outP, in.rows, in.cols, fifo_queue);

	// Perform propgation part of reconstruction algorithm 
	Negative_Propagation(inP, outP, in.rows, in.cols, fifo_queue);

	// Free up fifo memory 
	queue_kill(fifo_queue);
}

void Morphology::Reconstruction(cv::Mat &in, cv::Mat &out, ReconOp_TYPE opType, int winSize)
{

	if (in.type() != CV_32F) BOOST_THROW_EXCEPTION( UnsupportedType(in.type(), CV_32F) );
	if (winSize < 1) BOOST_THROW_EXCEPTION( UnexpectedSize(winSize, 1) );

	// Now, we use the original input planes as the "mask" array
	// and the eroded result as the "marker" array for the      
	// reconstruction. The reconstructed result gets written to 
	// the marker array.          
	if (opType == RECON_OPEN) {		
		Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*winSize+ 1, 2*winSize+1 ),Point( winSize, winSize) );
		erode( in, out, element );
		Morphology::Positive_Reconstruction(in, out); 
	} else if (opType == RECON_CLOSE) {		
		Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*winSize+ 1, 2*winSize+1 ),Point( winSize, winSize) );
		dilate( in, out, element );
		Morphology::Negative_Reconstruction(in, out);
	} else {		
		BOOST_THROW_EXCEPTION( NotImplemented() ); 
	} 
}

void Morphology::Gradient(cv::Mat &in, cv::Mat &out, int winSize)
{
	if (winSize < 1) BOOST_THROW_EXCEPTION( UnexpectedSize(winSize, 1) );

	Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*winSize+ 1, 2*winSize+1 ), Point( winSize, winSize) );	
	morphologyEx( in, out, MORPH_GRADIENT, element );  
}

void Morphology::Open(cv::Mat &in, cv::Mat &out, int winSize)
{
	if (winSize < 1) BOOST_THROW_EXCEPTION(UnexpectedSize(winSize, 1));

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * winSize + 1, 2 * winSize + 1), Point(winSize, winSize));
	morphologyEx(in, out, MORPH_OPEN, element);
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Morphology::Morphology()
{
}

Morphology::~Morphology()
{	
}

