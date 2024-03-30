#include <vtkVersion.h>
#include <vtkProperty.h>
#include <vtkDataSetMapper.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkTriangle.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkLine.h>
#include <vtkIndent.h>
#include <vtkImageData.h>
#include <vtkProbeFilter.h>
#include <vtkDelaunay2D.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkCharArray.h>
#include <vtkMath.h>
#include <vtkCellLocator.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkFloatArray.h>
#include <vtkWarpScalar.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkXMLPUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellIterator.h>
#include "GdalMat.h"
#include "opencv2/opencv.hpp"
#include "Logger.h"

using namespace cv; 
using namespace std; 

//#define MAX_IMAGE_SIZE  1024.0
#define MAX_IMAGE_SIZE  512.0

#if !defined(CLOSEST_INT)
#define CLOSEST_INT(X) (X > (floor(X)+0.5)) ? ceil(X) : floor(X)
#endif 

int main(  int argc, char *argv[] )
{
  LOG_CONFIGURE("log4cplus.properties");
  Logger m_logger(LOG_GET_LOGGER("InterpolateImage"));	
  
  //parse command line arguments
  if(argc != 3)  
  {
  std::cerr << "Usage: " << argv[0]
            << " frameNumString outputPrefix" << std::endl;
  return EXIT_FAILURE;
  }

  std::string frameNumStr = string(argv[1]);
  std::string filename = string("Sfrc_") + frameNumStr + ".pvtu";
  std::string outputPrefix = string(argv[2]);
  
  //read all the data from the file
  vtkSmartPointer<vtkXMLPUnstructuredGridReader> reader =
  vtkSmartPointer<vtkXMLPUnstructuredGridReader>::New();

  reader->SetFileName(filename.c_str());
  reader->Update();
  
  cout << "On input:\nNumber cells: "; 
  cout << reader->GetOutput()->GetNumberOfCells() << "\nNumber of points: ";
  cout << reader->GetOutput()->GetNumberOfPoints() << "\n";
  
  vtkSmartPointer<vtkCellData> inCellData = reader->GetOutput()->GetCellData();  
  cout << "Cell Data: " << inCellData->GetNumberOfArrays() << "\n";
  for (int i = 0; i < inCellData->GetNumberOfArrays(); i++) {
    std::cout << std::string(inCellData->GetArrayName(i)) << "\n"; 
  }
  vtkSmartPointer<vtkPointData> inPointData = reader->GetOutput()->GetPointData();  
  cout << "Point Data: " << inPointData->GetNumberOfArrays() << "\n";
  for (int i = 0; i < inPointData->GetNumberOfArrays(); i++) {
    std::cout << std::string(inPointData->GetArrayName(i)) << "\n"; 
  }
  
  double bounds[6];
  reader->GetOutput()->GetBounds(bounds);
  
  std::cout  << "xmin: " << bounds[0] << " " 
  << "xmax: " << bounds[1] << std::endl
  << "ymin: " << bounds[2] << " " 
  << "ymax: " << bounds[3] << std::endl
  << "zmin: " << bounds[4] << " " 
  << "zmax: " << bounds[5] << std::endl;
  
  bounds[0] = 0.0; 
  bounds[1] = 2.0; 
  bounds[2] = 0.0; 
  bounds[3] = 3.0; 

  double xoff = bounds[0]; 
  double yoff = bounds[2]; 
  double xdiff = bounds[1] - bounds[0]; 
  double ydiff = bounds[3] - bounds[2]; 
  int wider; 
  double res; 
  int width, height;

  if (xdiff > ydiff) {
    wider = 1; 
    res = xdiff / MAX_IMAGE_SIZE; 
    width = (int) MAX_IMAGE_SIZE; 
    height = (int) ceil(ydiff / res); 
  } 
  else {
      wider = 0; 
      res = ydiff / MAX_IMAGE_SIZE; 
      height = (int) MAX_IMAGE_SIZE; 
      width = (int) ceil(xdiff / res);   
  }

  cout << "Image will be " << width << " by " << height << " and spacing of " << res << "\n";
  Mat output0 = Mat::zeros(height, width, CV_32F);
  Mat output1 = Mat::zeros(height, width, CV_32F);

  //vtkSmartPointer<vtkFloatArray> cellArray = vtkFloatArray::SafeDownCast(inCellData->GetArray(0));

  double pt[3];  
  int xi[2];
  int yi[2];
  double xf[2];
  double yf[2];
  
  for(int ci=0; ci< reader->GetOutput()->GetNumberOfCells(); ci++) {
    
    int nump = reader->GetOutput()->GetCell(ci)->GetNumberOfPoints();
    if (nump != 2) {
      cout << "Not an edge!\n";
      return(0); 
    }
    //vtkIdList *ptids;
    //reader->GetOutput()->GetCellPoints(ci, ptids); 
    for(int ii=0; ii< nump; ii++) {
      
        reader->GetOutput()->GetPoint(reader->GetOutput()->GetCell(ci)->GetPointId(ii), pt);
        
        xf[ii] = ((double) pt[0]) / res; 
        yf[ii] = ((double) pt[1]) / res; 

        //xi[ii] = (int) CLOSEST_INT(xf[ii]); 
        //yi[ii] = (int) CLOSEST_INT(yf[ii]);
        
    }
    if (xf[0] < xf[1]) {
      xi[0] = (int) floor(xf[0]);
      xi[1] = (int) ceil(xf[1]);
    } else {
      xi[0] = (int) ceil(xf[0]);
      xi[1] = (int) floor(xf[1]);
    }
    if (yf[0] < yf[1]) {
      yi[0] = (int) floor(yf[0]);
      yi[1] = (int) ceil(yf[1]);
    } else {
      yi[0] = (int) ceil(yf[0]);
      yi[1] = (int) floor(yf[1]);
    }
    if (xi[0] >= width) {
      xi[0] = width - 1; 
    }
    if (xi[1] >= width) {
      xi[1] = width - 1; 
    }
    if (yi[0] >= height) {
      yi[0] = height - 1; 
    }
    if (yi[1] >= height) {
      yi[1] = height - 1; 
    }
    if (xi[0] < 0) {
      xi[0] = 0; 
    }
    if (xi[1] < 0) {
      xi[1] = 0; 
    }
    if (yi[0] < 0)  {
      yi[0] = 0;
    }
    if (yi[1] < 0) {
      yi[1] = 0; 
    }
    
      //if (xi[0] != xi[1]) cout << "x's different " << xi[0] << " verse " << xi[1] << "\n"; 
      //if (yi[0] != yi[1]) cout << "y's different " << yi[0] << " verse " << yi[1] << "\n"; 
    
      float val0 = (float) inCellData->GetArray(0)->GetTuple1(ci);
      float val1 =(float) inCellData->GetArray(1)->GetTuple1(ci);

      LineIterator it1(output0, cv::Point(xi[0], yi[0]), cv::Point(xi[1], yi[1]));
      
      for(int i = 0; i < it1.count; i++, ++it1)
      {
        float myVal = (output0.at<float>(it1.pos()) > val0) ? output0.at<float>(it1.pos()) : val0;
        output0.at<float>(it1.pos()) = myVal; 
      }

      LineIterator it2(output1, cv::Point(xi[0], yi[0]), cv::Point(xi[1], yi[1]));

      for(int i = 0; i < it2.count; i++, ++it2)
      {
        float myVal = (output1.at<float>(it2.pos()) > val1) ? output1.at<float>(it2.pos()) : val1;
        output1.at<float>(it2.pos()) = myVal; 
      }

        //cout <<  xi[0] << ", " << yi[0] << " -> " << xi[1] << ", " << yi[1] << "\n"; 
        //cv::line(output0, cv::Point(xi[0], yi[0]), cv::Point(xi[1], yi[1]), cv::Scalar(val0));
        //cv::line(output1, cv::Point(xi[0], yi[0]), cv::Point(xi[1], yi[1]), cv::Scalar(val1));
       
  }  
  std::string fname1 = outputPrefix + "/Damage_" + frameNumStr + ".tif";
  GdalMat::Write2DTiffFloat(fname1, output0);
  cout << "Saved: " + fname1 << "\n";    

  std::string fname2 = outputPrefix + "/TensileStrngth_" + frameNumStr + ".tif";  
  GdalMat::Write2DTiffFloat(fname2, output0);
  cout << "Saved: " + fname2 << "\n";    

  return EXIT_SUCCESS;
}
