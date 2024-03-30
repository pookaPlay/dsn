"""
Functions to manipulate directorie sand files.
"""
import os

def create_dir(fpath):
    """
    Create a directory if it does not exitst given its
    full path.

    Parameters
    ----------
    fpth: str
        Full path of directory.
    """
    if(not(os.path.isdir(fpath))):
        print("INFO: Creating ", fpath)
        os.mkdir(fpath)
    else:
        print("INFO: Directory",fpath," exists")
