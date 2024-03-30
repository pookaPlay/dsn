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
  std::string filename = string("Sout_") + frameNumStr + ".pvtu";
  std::string outputPrefix = string(argv[2]);
  
  //CellTag
  //MunjizaStress_V
  //MunjizaStress_1
  //MunjizaStress_2
  //MunjizaStress_12
  //CauchyStress_1
  //CauchyStress_2
  //CauchyStress_12
  //
  //NodalVelocity
  //NodalGlobalID

  std::vector<std::string> cellNames;
  std::vector<std::string> nodeNames;
    
  cellNames.clear(); 
  nodeNames.clear(); 

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
    cellNames.push_back(std::string(inCellData->GetArrayName(i)));
  }
  vtkSmartPointer<vtkPointData> inPointData = reader->GetOutput()->GetPointData();  
  cout << "Point Data: " << inPointData->GetNumberOfArrays() << "\n";
  for (int i = 0; i < inPointData->GetNumberOfArrays(); i++) {
    std::cout << std::string(inPointData->GetArrayName(i)) << "\n"; 
    nodeNames.push_back(std::string(inPointData->GetArrayName(i)));
  }
  
  double bounds[6];
  reader->GetOutput()->GetBounds(bounds);
  
  std::cout  << "xmin: " << bounds[0] << " " 
  << "xmax: " << bounds[1] << std::endl
  << "ymin: " << bounds[2] << " " 
  << "ymax: " << bounds[3] << std::endl
  << "zmin: " << bounds[4] << " " 
  << "zmax: " << bounds[5] << std::endl;
  
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

  
  for(int ai=1; ai<8; ai++) {
    std::vector< std::vector < cv::Point > > contours;
    contours.resize(1); 
    contours[0].resize(3);
  
    double pt[3]; 
    int xi[4];
    int yi[4];
      
    std::string fname = outputPrefix + "/" + cellNames[ai] + "_" + frameNumStr + ".tif";      
    
    cv::Mat output  = Mat::zeros(height, width, CV_32F);  

    for(int ci=0; ci< reader->GetOutput()->GetNumberOfCells(); ci++) {
          
          int nump = reader->GetOutput()->GetCell(ci)->GetNumberOfPoints();
          if (nump != 3) {
            cout << "Not a triangle!\n";
            return(0); 
          }
          //vtkIdList *ptids;
          //reader->GetOutput()->GetCellPoints(ci, ptids); 
          for(int ii=0; ii< nump; ii++) {
            
              reader->GetOutput()->GetPoint(reader->GetOutput()->GetCell(ci)->GetPointId(ii), pt);
              
              xi[ii] = (int) CLOSEST_INT(pt[0] / res); 
              yi[ii] = (int) CLOSEST_INT(pt[1] / res);
              contours[0][ii] = cv::Point(xi[ii], yi[ii]);
          }
            //if (xi[0] != xi[1]) cout << "x's different " << xi[0] << " verse " << xi[1] << "\n"; 
            //if (yi[0] != yi[1]) cout << "y's different " << yi[0] << " verse " << yi[1] << "\n"; 
          for(int ai=1; ai<8; ai++) {  
              float val = (float) inCellData->GetArray(ai)->GetTuple1(ci);                
              cv::drawContours(output, contours, 0, cv::Scalar(val), -1); 
            }
            
    }  
    
    GdalMat::Write2DTiffFloat(fname, output);
    cout << "Saved: " + fname << "\n";        
  }
/*
  cv::Mat output1 = Mat::zeros(height, width, CV_32F);  
  cv::Mat output2 = Mat::zeros(height, width, CV_32F);  

  //vtkSmartPointer<vtkFloatArray> pointArray = vtkFloatArray::SafeDownCast(inPointData->GetArray(0));
  for(int pii=0; pii< reader->GetOutput()->GetNumberOfPoints(); pii++) {
    double pt[3]; 
    int xi[4];
    int yi[4];

    inPointData->GetArray(0)->GetTuple(pii, pt);                
    float val1 = (float) pt[0];
    float val2 = (float) pt[1];                
    reader->GetOutput()->GetPoint(pii, pt);
    xi[0] = (int) CLOSEST_INT(pt[0] / res); 
    yi[0] = (int) CLOSEST_INT(pt[1] / res);
    
    output1.at<float>(yi[0], xi[0]) = val1; 
    output2.at<float>(yi[0], xi[0]) = val2; 
    
  }


  std::string fname1 = outputPrefix + "/NodalVelocityX_" + frameNumStr + ".tif";      
  GdalMat::Write2DTiffFloat(fname1, output1);
  cout << "Saved: " + fname1 << "\n";    

  std::string fname2 = outputPrefix + "/NodalVelocityY_" + frameNumStr + ".tif";        
  GdalMat::Write2DTiffFloat(fname2, output2);
  cout << "Saved: " + fname2 << "\n";    
  */
  return EXIT_SUCCESS;
}
