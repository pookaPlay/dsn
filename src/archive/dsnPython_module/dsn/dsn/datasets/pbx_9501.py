import os
import pdb
import sys
import glob
import skimage.io as skio
from astropy.io import fits
import matplotlib.pyplot as plt

# functions/methods from current module
from ..fd_ops import create_dir


class PBX9501:

    
    def __init__(self, rdir, gt_rdir):
        """
        Initiates an instance having methods to perform various operations 
        on Braniac2-snemi3d dataset.

        Initiates an instance having methods to perform operations on 
        Brian Petterson X-Ray dataset.

        Parameters
        ----------
        rdir: str
            Path to dataset. This path is assumed to contain FITS files 
            (`.fits`).
        gt_rdir: str
            Path to ground truth.
        """
        self._rdir     = rdir
        self._gt_rdir  = gt_rdir

        # An list having path of all the fits files under the root directory
        self._fits_lst = glob.glob(self._gt_rdir + '/*.fits*')

        # An list having image files.
        self._tiff_lst = glob.glob(self._rdir + '/*.tif*')

    def extract_gt_images(self, cut_sz):
        """
        Extracts images into directories having same name as the fits file stack.

        Parameters
        ----------
        cut_sz
            Size of each image extracted from the original image as
            [nrows, ncols], For now only perfect cutting is supported.

        Notes
        -----
            When extracting label stack user might get a Warning, 
            "is a low contrast image". This is okay to ignore.
        """
        for cfits_path in self._fits_lst:
            print("INFO: Extracting ", cfits_path)
            
            # Create a directory to extract frames
            extraction_path = os.path.splitext(cfits_path)[0] + '_' +\
                str(cut_sz[0]) + 'x' + str(cut_sz[1])
            create_dir(extraction_path)

            # Load fits file
            hdu_list = fits.open(cfits_path)
            img      = hdu_list[0].data

            # Cut edges to make image having size multiple of `cut_sz`
            cut_row_sz                 = cut_sz[0]
            cut_col_sz                 = cut_sz[1]

            img_rows, img_cols         = img.shape

            rem_rows                   = img_rows%cut_row_sz
            rem_cols                   = img_cols%cut_col_sz

            row_trim_idx0              = 0        + int(rem_rows/2)
            row_trim_idx1              = img_rows - (rem_rows - int(rem_rows/2))
            col_trim_idx0              = 0        + int(rem_cols/2)
            col_trim_idx1              = img_cols - (rem_cols - int(rem_cols/2))

            img_trimmed                = img[row_trim_idx0:row_trim_idx1,
                                             col_trim_idx0:col_trim_idx1]
            trm_img_rows, trm_img_cols = img_trimmed.shape
            

            # Cut the current stack
            for ridx in range(0,int(trm_img_rows/cut_row_sz)):

                for cidx in range(0,int(trm_img_cols/cut_col_sz)):

                    # Extract each stack at a time
                    row_st     = ridx       * cut_sz[0]
                    row_en     = (ridx + 1) * cut_sz[0]
                    col_st     = cidx       * cut_sz[1]
                    col_en     = (cidx + 1) * cut_sz[1]
                    cut_img    = img_trimmed[row_st:row_en, col_st:col_en]

                    # Create current stack save location
                    cut_img_name  = str(ridx) + '_' + str(cidx) +'.tif'
                    cut_img_fpath = extraction_path + '/' + cut_img_name


                    skio.imsave(cut_img_fpath, cut_img)



    def extract_images(self, cut_sz):
        """
        Extracts images into directories having same name as the fits file stack.

        Parameters
        ----------
        cut_sz
            Size of each image extracted from the original image as
            [nrows, ncols], For now only perfect cutting is supported.

        Notes
        -----
            When extracting label stack user might get a Warning, 
            "is a low contrast image". This is okay to ignore.
        """
        for ctiff_path in self._tiff_lst:
            print("INFO: Extracting ", ctiff_path)
            
            # Create a directory to extract frames
            extraction_path = os.path.splitext(ctiff_path)[0] + '_' +\
                str(cut_sz[0]) + 'x' + str(cut_sz[1])
            create_dir(extraction_path)

            # Load fits file
            img = skio.imread(ctiff_path)

            # Cut edges to make image having size multiple of `cut_sz`
            cut_row_sz                 = cut_sz[0]
            cut_col_sz                 = cut_sz[1]

            img_rows, img_cols         = img.shape

            rem_rows                   = img_rows%cut_row_sz
            rem_cols                   = img_cols%cut_col_sz

            row_trim_idx0              = 0        + int(rem_rows/2)
            row_trim_idx1              = img_rows - (rem_rows - int(rem_rows/2))
            col_trim_idx0              = 0        + int(rem_cols/2)
            col_trim_idx1              = img_cols - (rem_cols - int(rem_cols/2))

            img_trimmed                = img[row_trim_idx0:row_trim_idx1,
                                             col_trim_idx0:col_trim_idx1]
            trm_img_rows, trm_img_cols = img_trimmed.shape
            

            # Cut the current stack
            for ridx in range(0,int(trm_img_rows/cut_row_sz)):

                for cidx in range(0,int(trm_img_cols/cut_col_sz)):

                    # Extract each stack at a time
                    row_st     = ridx       * cut_sz[0]
                    row_en     = (ridx + 1) * cut_sz[0]
                    col_st     = cidx       * cut_sz[1]
                    col_en     = (cidx + 1) * cut_sz[1]
                    cut_img    = img_trimmed[row_st:row_en, col_st:col_en]

                    # Create current stack save location
                    cut_img_name  = str(ridx) + '_' + str(cidx) +'.tif'
                    cut_img_fpath = extraction_path + '/' + cut_img_name


                    skio.imsave(cut_img_fpath, cut_img)


            
            
        
