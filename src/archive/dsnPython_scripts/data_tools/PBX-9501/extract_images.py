""" 
The following script extracts images 2D images TIFF images 
from 3D TIFF images.

Notes
-----
    When extracting label stack user might get a Warning, 
    'is a low contrast image'. This is okay to ignore.
"""
import pdb
from dsn.datasets import PBX9501

pbx9501 = PBX9501('../../../../../../data/PBX-9501',
                  '../../../../../../data/PBX-9501-gt')

# Extracting ground truth images
pbx9501.extract_gt_images([128,128])

# Extracting cuttings from original images
pbx9501.extract_images([128,128])
