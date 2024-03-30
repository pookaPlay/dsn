import gdal

TEST_FILE = '../../data/gradSquare.tif'

def LoadSyn():

    gdal.AllRegister()

    ds = gdal.Open(TEST_FILE)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    print(arr.shape)
    return arr
