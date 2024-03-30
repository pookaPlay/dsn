import os
import pdb
import glob
import skimage.io as skio

# functions/methods from current module
from ..fd_ops import create_dir


class ISBI:

    
    def __init__(self, dir_path):
        """
        Initiates an instance having methods to perform various operations 
        on Braniac2-ISBI dataset.

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


    def extract_images(self):
        """
        Extracts images into directories having same name as the TIFF stack.

        Notes
        -----
            When extracting label stack user might get a Warning, 
            "is a low contrast image". This is okay to ignore.
        """
        for key in self._tifstacks:
            print("INFO: Extracting ", key)
            
            # Create a directory to extract frames
            extraction_path = os.path.splitext(key)[0]
            create_dir(extraction_path)
            
            # Load current stack
            cstack    = self._tifstacks[key]
            (num_stacks, _, _) = cstack.shape

            # Loop over current stack
            for idx in range(0,num_stacks):
                print("\t",idx)
                # Extract each stack at a time
                c_2d_stack       = cstack[idx, :, :]

                # Create current stack save location
                c_2d_stack_name  = str(idx) + '.tif'
                c_2d_stack_fpath = extraction_path + '/' + c_2d_stack_name

                # Save the stack
                skio.imsave(c_2d_stack_fpath, c_2d_stack)


            
            
        
 
