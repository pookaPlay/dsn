#include "GdalMat.h"
#include "DadaDef.h"
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()
#include "cpl_string.h"

Logger GdalMat::m_logger(LOG_GET_LOGGER("GdalMat"));

void GdalMat::ReadHeader(std::string fname, int &w, int &h, int &d)
{
	GDALDataset  *poDataset;
	GDALAllRegister();
	poDataset = (GDALDataset *)GDALOpen(fname.c_str(), GA_ReadOnly);

	if (poDataset) {
		w = poDataset->GetRasterXSize();
		h = poDataset->GetRasterYSize();
		d = poDataset->GetRasterCount();
	}
}

void GdalMat::Read2DWindowAsFloat(std::string fname, cv::Mat &img, int x0, int y0, int x1, int y1)
{

	GDALDataset  *poDataset;
	GDALAllRegister();
	poDataset = (GDALDataset *)GDALOpen(fname.c_str(), GA_ReadOnly);

	if (poDataset) {
		//double        adfGeoTransform[6];
		//LOG_INFO(m_logger, "Driver: " << poDataset->GetDriver()->GetDescription() << "/" << poDataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME));
		//LOG_INFO(m_logger, "Size: " << poDataset->GetRasterXSize() << " by " << poDataset->GetRasterYSize() << " by " << poDataset->GetRasterCount());

		GDALRasterBand  *poBand;
		poBand = poDataset->GetRasterBand(1);

		int w = x1 - x0; 
		int h = y1 - y0; 
		
		img.create(h, w, CV_32F);
		float* imgp = img.ptr<float>(0);
		poBand->RasterIO(GF_Read,  x0, y0, w, h, imgp, w, h, GDT_Float32, 0, 0);
	}
}

void GdalMat::Read2DAsFloat(std::string fname, cv::Mat &img)
{

	GDALDataset  *poDataset;
	GDALAllRegister();
	poDataset = (GDALDataset *) GDALOpen(fname.c_str(), GA_ReadOnly);

	if (poDataset) {
		double        adfGeoTransform[6];
		//LOG_INFO(m_logger, "Driver: " << poDataset->GetDriver()->GetDescription() << "/" << poDataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME));
		//LOG_INFO(m_logger, "Size: " << poDataset->GetRasterXSize() << " by " << poDataset->GetRasterYSize() << " by " << poDataset->GetRasterCount());

		GDALRasterBand  *poBand;
		int             nBlockXSize, nBlockYSize;
		int             bGotMin, bGotMax;
		double          adfMinMax[2];
		poBand = poDataset->GetRasterBand(1);
		
		//poBand->GetBlockSize(&nBlockXSize, &nBlockYSize);
		//LOG_INFO(m_logger, "Block = " << nBlockXSize << "," << nBlockYSize << " Type = " << GDALGetDataTypeName(poBand->GetRasterDataType())); 
		
		//adfMinMax[0] = poBand->GetMinimum(&bGotMin);
		//adfMinMax[1] = poBand->GetMaximum(&bGotMax);
		//if (!(bGotMin && bGotMax)) GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);		
		//LOG_INFO(m_logger, "Min = " << adfMinMax[0] << " Max = " << adfMinMax[1]);
	
		int   nXSize = poBand->GetXSize();
		int   nYSize = poBand->GetYSize();
		img.create(nYSize, nXSize, CV_32F);
		float* imgp = img.ptr<float>(0);		
		poBand->RasterIO(GF_Read, 0, 0, nXSize, nYSize, imgp, nXSize, nYSize, GDT_Float32, 0, 0);
	}
}

void GdalMat::ReadColorAsFloat(std::string fname, std::vector< cv::Mat > &imgs, int numBands)
{
  imgs.clear();

  GDALDataset  *poDataset;
  GDALAllRegister();
  poDataset = (GDALDataset *)GDALOpen(fname.c_str(), GA_ReadOnly);
  
  if (poDataset) {
    double        adfGeoTransform[6];
    //LOG_INFO(m_logger, "Driver: " << poDataset->GetDriver()->GetDescription() << "/" << poDataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME));
    LOG_INFO(m_logger, "Size: " << poDataset->GetRasterXSize() << " by " << poDataset->GetRasterYSize() << " by " << poDataset->GetRasterCount());

    if (numBands > poDataset->GetRasterCount()) throw std::runtime_error("Not enough bands in image file");

    imgs.resize(numBands);

    GDALRasterBand  *poBand;

    for (int i = 0; i < numBands; i++) {
      poBand = poDataset->GetRasterBand(i + 1);
      int   nXSize = poBand->GetXSize();
      int   nYSize = poBand->GetYSize();
      imgs[i].create(nYSize, nXSize, CV_32F);
      float* imgp = imgs[i].ptr<float>(0);
      poBand->RasterIO(GF_Read, 0, 0, nXSize, nYSize, imgp, nXSize, nYSize, GDT_Float32, 0, 0);
    }
  }
}

void GdalMat::ReadColorAs8UC3(std::string fname, cv::Mat &img)
{
  std::vector< cv::Mat > imgs;   
  GdalMat::ReadColorAsFloat(fname, imgs, 3); 

  std::vector< cv::Mat > imgb(imgs.size());

  for (int b = 0; b < imgs.size(); b++) {
    double minVal, maxVal;
    minMaxLoc(imgs[b], &minVal, &maxVal); //, &minLoc, &maxLoc );    
    float alpha = 255.0f / (maxVal - minVal);
    float beta = -(minVal * alpha);
    imgs[b].convertTo(imgb[b], CV_8U, alpha, beta);
  }

  merge(imgb, img); 
 // LOG_INFO("After merge I have " << img.type() << " and 8UC3 is " << CV_8UC3);

}

void GdalMat::Write2DTiffFloat(std::string fname, cv::Mat &img)
{
  if (img.type() != CV_32F) {
    LOG_ERROR(m_logger, "ERROR: Image is not a float?");
    return; 
  }
  GDALAllRegister();
  //const char *pszFormat = "GTiff";
  const char *pszFormat = "GTiff";
  GDALDriver *poDriver;
  
  poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
  if( poDriver == NULL ) {
	LOG_ERROR(m_logger, "ERROR: No GDAL driver\n"); 
	return; 
  }
  
  char **papszOptions = NULL;
  GDALDataset *poDstDS = poDriver->Create( fname.c_str(), img.cols, img.rows, 1, GDT_Float32, papszOptions );

  GDALRasterBand *poBand = poDstDS->GetRasterBand(1);  
  float* imgp = img.ptr<float>(0);
 
  poBand->RasterIO( GF_Write, 0, 0, img.cols, img.rows,
					imgp, img.cols, img.rows, GDT_Float32, 0, 0 );
  
  GDALClose( (GDALDatasetH) poDstDS );
								
  //papszMetadata = poDriver->GetMetadata();
  //if( CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATE, FALSE ) )
	//	  printf( "Driver %s supports Create() method.\n", pszFormat );
  //if( CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATECOPY, FALSE ) )
	//	  printf( "Driver %s supports CreateCopy() method.\n", pszFormat );

}

GdalMat::GdalMat()
{
}

GdalMat::~GdalMat(){
}

