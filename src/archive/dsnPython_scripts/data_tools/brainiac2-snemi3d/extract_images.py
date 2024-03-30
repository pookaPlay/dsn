""" 
The following script extracts images 2D images TIFF images 
from 3D TIFF images.

Notes
-----
    When extracting label stack user might get a Warning, 
    'is a low contrast image'. This is okay to ignore.
"""
import pdb
from dsn.datasets import Snemi3d

#snemi3d = Snemi3d('/home/vj/Dropbox/LosAlamos/dsn/data/snemi3d')
#snemi3d.extract_images([128,128])

snemi3d = Snemi3d('D:\\image_data\\snemi3d')
snemi3d.extract_images([256,256])
