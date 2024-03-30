""" 
The following script extracts images 2D images TIFF images 
from 3D TIFF images.

Notes
-----
    When extracting label stack user might get a Warning, 
    'is a low contrast image'. This is okay to ignore.
"""
import pdb
from dsn.datasets import ISBI

isbi = ISBI('/home/vj/Dropbox/LosAlamos/dsn/data/ISBIChallenge')
isbi.extract_images()

