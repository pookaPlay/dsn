import os
import pdb
import sys
import glob
import skimage.io as skio

# functions/methods from current module
from ..fd_ops import create_dir


class Snemi3d:

    
    def __init__(self, dir_path):
        """
        Initiates an instance having methods to perform various operations 
        on Braniac2-snemi3d dataset.

        Parameters
        ----------
        dir_path: str
            Path to dataset. This path is assumed to contain TIFF stacks
            (`.tiff` or `.tif`).

        Notes
        -----
        1. Does not load TIFF stacks recursively.
        2. The TIFF stacks are assumed to be depth first.
        """
        self._rdir    = dir_path

        # Load TIFF stacks as dictionary
        tif_flst      = glob.glob(self._rdir + '/*.tif*')
        self._tifstacks = {}
        for ctif in tif_flst:
            self._tifstacks[ctif] = skio.imread(ctif)


    def extract_images(self, cut_sz):
        """
        Extracts images into directories having same name as the TIFF stack.

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
        for key in self._tifstacks:
            print("INFO: Extracting ", key)
            
            # Create a directory to extract frames
            extraction_path = os.path.splitext(key)[0] + '_' +\
                str(cut_sz[0]) + 'x' + str(cut_sz[1])
            create_dir(extraction_path)
            
            # Load current stack
            cstack    = self._tifstacks[key]
            (num_stacks,h_stack,w_stack) = cstack.shape

            # Check for perfect cutting
            if(not(h_stack%cut_sz[0] == 0) or not(w_stack%cut_sz[1] == 0)):
                print("Perfect cutting is not possible")
                sys.exit(1)

            # Loop over current stack
            for idx in range(0,num_stacks):
                
                # Cut the current stack
                for ridx in range(0,int(h_stack/cut_sz[0])):
                    
                    for cidx in range(0,int(h_stack/cut_sz[1])):
                        
                        # Extract each stack at a time
                        row_st     = ridx       * cut_sz[0]
                        row_en     = (ridx + 1) * cut_sz[0]
                        col_st     = cidx       * cut_sz[1]
                        col_en     = (cidx + 1) * cut_sz[1]
                        c_2d_stack = cstack[idx, row_st:row_en, col_st:col_en]
                        
                        # Create current stack save location
                        c_2d_stack_name  = 'stack_' + str(idx) +\
                            '_' + str(ridx) + '_' + str(cidx) +'.tif'
                        c_2d_stack_fpath = extraction_path + '/' + c_2d_stack_name

                        # Save the stack
                        skio.imsave(c_2d_stack_fpath, c_2d_stack)


            
            
        
